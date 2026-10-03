"""Runs an agent over the golden set and records a scored, comparable run."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rich.table import Table

from brokerchat.expected import read_case
from brokerchat.glossary import Glossary
from brokerchat.pipeline import AgentResult, Pipeline, Processor, Usage, write_run_outputs
from brokerchat.scoring import TARGETS, CaseScore, EvalReport, meets_targets, score_case
from brokerchat.store import Store


@dataclass
class CaseRun:
    name: str
    store: Store
    results: list[AgentResult]
    score: CaseScore


def run_eval(
    golden_dir: Path,
    agent_factory: Callable[[], Processor],
    glossary: Glossary,
    case_names: list[str] | None = None,
    on_case: Callable[[CaseRun], None] | None = None,
) -> tuple[EvalReport, list[CaseRun]]:
    """Run every case. on_case is called as each case finishes, so paid results survive a later crash."""
    paths = sorted(Path(golden_dir).glob("*.jsonl"))
    if case_names:
        missing = set(case_names) - {p.stem for p in paths}
        if missing:
            raise ValueError(f"cases not found in {golden_dir}: {', '.join(sorted(missing))}")
        paths = [p for p in paths if p.stem in case_names]
    if not paths:
        raise ValueError(f"no cases found in {golden_dir}")

    runs = []
    for path in paths:
        case = read_case(path)
        store = Store(glossary)
        pipeline = Pipeline(store, agent_factory())
        pipeline.run(case.messages)
        run = CaseRun(case.expected.name, store, pipeline.results,
                      score_case(case.expected, store.board, set(store.messages)))
        runs.append(run)
        if on_case is not None:
            on_case(run)

    usage = Usage()
    for run in runs:
        for result in run.results:
            usage.add(result.usage)
    report = EvalReport(
        cases=[r.score for r in runs],
        messages=sum(len(r.results) for r in runs),
        unprocessed=sum(1 for r in runs for res in r.results if res.status != "ok"),
        usage=usage,
    )
    return report, runs


def new_run_dir(runs_dir: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_case_run(run: CaseRun, run_dir: Path) -> None:
    write_run_outputs(run.store, run.results, run_dir / run.name)
    (run_dir / run.name / "score.json").write_text(run.score.model_dump_json(indent=2), encoding="utf-8")


def write_summary(report: EvalReport, run_dir: Path) -> None:
    summary = report.summary()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "targets_met": meets_targets(summary),
                "messages": report.messages,
                "unprocessed": report.unprocessed,
                "usage": report.usage.model_dump(),
                "cost_usd": report.usage.cost_usd(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_eval_run(report: EvalReport, runs: list[CaseRun], runs_dir: Path, now: datetime | None = None) -> Path:
    run_dir = new_run_dir(runs_dir, now)
    for run in runs:
        write_case_run(run, run_dir)
    write_summary(report, run_dir)
    return run_dir


def report_table(report: EvalReport) -> Table:
    summary = report.summary()
    met = meets_targets(summary)
    table = Table(title=f"Eval: {len(report.cases)} cases, {report.messages} messages, "
                        f"{report.unprocessed} unprocessed, approx ${report.usage.cost_usd():.2f}")
    for column in ("Metric", "Value", "Target", "Met"):
        table.add_column(column)
    for name, value in summary.items():
        if name in TARGETS:
            op, target = TARGETS[name]
            table.add_row(name, f"{value:.3f}", f"{op} {target}", "yes" if met[name] else "[red]no[/red]")
        else:
            table.add_row(name, f"{value:.3f}", "", "")
    return table


def failure_lines(report: EvalReport) -> list[str]:
    lines = []
    for c in report.cases:
        lines += [f"{c.name}: missing quote {m}" for m in c.missing_quotes]
        lines += [f"{c.name}: spurious quote {m}" for m in c.spurious_quotes]
        lines += [f"{c.name}: {e}" for e in c.field_errors]
        lines += [f"{c.name}: no flag for ambiguous message {m}" for m in c.missed_flags]
    return lines
