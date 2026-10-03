"""Command-line entry point: brokerchat replay | generate | eval."""

import argparse
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from brokerchat.chatio import read_chat
from brokerchat.evalrun import failure_lines, new_run_dir, report_table, run_eval, write_case_run, write_summary
from brokerchat.expected import write_case
from brokerchat.generator.builder import load_scenario
from brokerchat.generator.random_day import random_day
from brokerchat.glossary import Glossary
from brokerchat.pipeline import NullAgent, Pipeline, Processor, write_run_outputs
from brokerchat.render import render_board
from brokerchat.store import Store

AGENT_CHOICES = ["claude", "null"]
EFFORT_CHOICES = ["low", "medium", "high", "xhigh", "max"]

console = Console()

# Where `ant auth login` stores its profiles; the Anthropic SDK reads them automatically.
ANT_PROFILE_DIR = Path.home() / ".config" / "anthropic"
CREDENTIAL_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
NO_CREDENTIALS_MESSAGE = (
    "[red]No Anthropic credentials found.[/red] Put ANTHROPIC_API_KEY=sk-ant-... in a .env file in this folder, "
    "set it in your shell, or run `ant auth login`. Use --agent null to run without the API."
)


def load_env(path: Path = Path(".env")) -> None:
    """Load KEY=value lines from a .env file for local development. Variables already set in the shell win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def has_credentials() -> bool:
    return any(os.environ.get(var) for var in CREDENTIAL_ENV_VARS) or ANT_PROFILE_DIR.is_dir()


def make_agent(name: str, glossary: Glossary, effort: str = "medium") -> Processor:
    if name == "null":
        return NullAgent()
    if name == "claude":
        import anthropic

        from brokerchat.agent.loop import ClaudeAgent

        return ClaudeAgent(anthropic.Anthropic(), glossary, effort=effort)
    raise ValueError(f"unknown agent {name!r}")


def cmd_replay(args: argparse.Namespace) -> int:
    glossary = Glossary.load()
    messages = read_chat(args.chat_file)
    store = Store(glossary)
    pipeline = Pipeline(store, make_agent(args.agent, glossary, args.effort))

    if args.no_live:
        pipeline.run(messages)
        console.print(render_board(store.board))
    else:
        with Live(render_board(store.board), console=console, refresh_per_second=8) as live:
            previous = None
            for message in messages:
                if args.speed > 0 and previous is not None:
                    time.sleep((message.ts - previous).total_seconds() / args.speed)
                pipeline.step(message)
                live.update(render_board(store.board))
                previous = message.ts

    not_ok = [r for r in pipeline.results if r.status != "ok"]
    console.print(
        f"{len(messages)} messages, {len(not_ok)} not fully processed, "
        f"approx ${pipeline.total_usage().cost_usd():.2f} API cost"
    )
    for result in not_ok:
        console.print(f"  [yellow]{result.message_id}: {result.status} ({result.reason})[/yellow]")
    if args.out is not None:
        write_run_outputs(store, pipeline.results, args.out)
        console.print(f"Outputs written to {args.out}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    glossary = Glossary.load()
    cases = [random_day(glossary, seed, n_steps=args.steps) for seed in args.seeds]
    cases += [load_scenario(glossary, path) for path in args.scenario]
    if not cases:
        console.print("[red]Nothing to generate: pass --seeds and/or --scenario[/red]")
        return 2
    for case in cases:
        path = write_case(case, args.out)
        console.print(
            f"{path}  {len(case.messages)} messages, {len(case.expected.quotes)} quotes, "
            f"{len(case.expected.ambiguous_message_ids)} ambiguous"
        )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    glossary = Glossary.load()
    run_dir = new_run_dir(args.runs_dir)
    report, _ = run_eval(
        args.golden,
        lambda: make_agent(args.agent, glossary, args.effort),
        glossary,
        case_names=args.case or None,
        on_case=lambda run: write_case_run(run, run_dir),
    )
    write_summary(report, run_dir)
    console.print(report_table(report))
    if args.show_failures:
        for line in failure_lines(report):
            console.print(f"  {line}")
    console.print(f"Run written to {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brokerchat", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="Replay a chat file through the agent and show the live board")
    replay.add_argument("chat_file", type=Path)
    replay.add_argument("--speed", type=float, default=0.0, help="Replay speed multiplier; 0 means no delays")
    replay.add_argument("--agent", choices=AGENT_CHOICES, default=AGENT_CHOICES[0])
    replay.add_argument("--effort", choices=EFFORT_CHOICES, default="medium")
    replay.add_argument("--out", type=Path, default=None, help="Directory for board.json, events.jsonl, results.jsonl")
    replay.add_argument("--no-live", action="store_true", help="Print the final board instead of a live view")
    replay.set_defaults(func=cmd_replay)

    generate = sub.add_parser("generate", help="Generate synthetic chat with its expected board")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--seeds", type=int, nargs="*", default=[], help="Random days to generate, one per seed")
    generate.add_argument("--steps", type=int, default=30, help="Messages per random day")
    generate.add_argument("--scenario", type=Path, action="append", default=[], help="Scenario YAML (repeatable)")
    generate.set_defaults(func=cmd_generate)

    evaluate = sub.add_parser("eval", help="Score the agent against the golden set")
    evaluate.add_argument("--golden", type=Path, default=Path("evals/golden"))
    evaluate.add_argument("--agent", choices=AGENT_CHOICES, default=AGENT_CHOICES[0])
    evaluate.add_argument("--effort", choices=EFFORT_CHOICES, default="medium")
    evaluate.add_argument("--runs-dir", type=Path, default=Path("evals/runs"))
    evaluate.add_argument("--case", action="append", default=[], help="Only run this case (repeatable)")
    evaluate.add_argument("--show-failures", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    if getattr(args, "agent", None) == "claude" and not has_credentials():
        console.print(NO_CREDENTIALS_MESSAGE)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
