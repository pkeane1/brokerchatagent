# brokerchat

A Claude agent that reads CDS broker chat one message at a time and keeps a live, structured quote board: levels, sizes, axes, client interest, and whether each quote is live, referred, traded or stale.

It's a local prototype that runs on **synthetic chat only**. The design and the reasons behind it are in the [PRD](docs/superpowers/specs/2026-10-02-broker-chat-parser-prd.md). The build steps are in the [implementation plan](docs/superpowers/plans/2026-10-02-broker-chat-parser.md).

## Quick start

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12+ and the dependencies for you.

```bash
uv sync                      # install dependencies
uv run pytest -q             # run the tests (no API calls)
```

These commands cost nothing because they use the dummy agent, which never calls the API:

```bash
uv run brokerchat replay examples/handwritten.jsonl --agent null --speed 600
uv run brokerchat eval --agent null
```

The real agent calls the Claude API (`claude-opus-5-5`) and **costs money**. You need credentials first: put
`ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file in the repo root (it is git-ignored and loaded automatically),
set it in your shell, or run `ant auth login`. Without credentials, `--agent claude` stops with a message.

```bash
uv run brokerchat replay evals/golden/day01.jsonl --speed 30   # watch the live board
uv run brokerchat eval --case basic_flow                       # score one case
uv run brokerchat eval --show-failures                         # score the full golden set
```

`claude` is the default agent. Pass `--agent null` whenever you don't want to spend anything.

## Commands

| Command | What it does |
|---------|--------------|
| `brokerchat replay FILE` | Feeds a chat file through the agent and shows the board updating live. `--speed 30` replays at 30x real time (0 means as fast as possible). `--no-live` prints only the final board. `--out DIR` saves the board, the event log and per-message results. |
| `brokerchat generate --out DIR` | Writes synthetic chat plus the board a perfect agent should end with. `--seeds 1 2 3` makes random days. `--scenario FILE.yaml` builds a hand-written scenario. `--steps` sets messages per random day. |
| `brokerchat eval` | Runs the agent over `evals/golden/`, scores it against the expected boards, and saves the run to `evals/runs/<timestamp>/`. `--case NAME` runs only that case. `--show-failures` lists every mistake. `--effort low/medium/high/...` changes how hard Claude thinks. |

## How one message flows through the system

```
 chat file (.jsonl)
      │  chatio.read_chat
      ▼
 Pipeline.step(message)                                  pipeline.py
      │
      ├─► agent.process(message, store)                  agent/loop.py
      │       │  builds context: board + last 20 messages in the room   agent/prompt.py
      │       │  calls Claude with 7 tools; loops until Claude stops calling tools (max 8 calls)
      │       ▼
      │   ToolExecutor validates each tool call         agent/tools.py
      │       │  bad call → error sent back to Claude, nothing written
      │       ▼
      │   store.emit(event) → event log + Board updated  store.py
      │
      ├─► sweep_stale: live quotes older than 30 min → stale       rules.py
      └─► detect_conflicts: crossed markets → conflict flag        rules.py
      ▼
 render_board → terminal                                 render.py
```

Three design rules hold throughout the code:

1. **Event-sourced.** Nothing changes the board directly. Every change is an event added to a log, and the board is rebuilt from that log. You can replay `events.jsonl` to get the same board again.
2. **Every entry cites its source.** Each quote, axe and interest lists the chat message ids it came from. The tools reject ids the pipeline hasn't seen, so Claude can't invent a source.
3. **Claude handles language, code handles rules.** Claude interprets the chat ("off that", "same in 10y"). Plain code looks up instrument aliases, marks quotes stale and spots crossed markets.

## What each file does

### `src/brokerchat/`: the application

| File | Responsibility |
|------|----------------|
| `models.py` | The data types: `ChatMessage`, `Instrument`, `Quote`, `Axe`, `Interest`, `Flag`, `Event`, and their enums (status, direction and so on). Levels are `Decimal` so prices like 104.625 stay exact. |
| `glossary.py` | Loads `domain/glossary.yaml`. Resolves aliases such as "Main", "XO" and "vw" to canonical names, and normalises tenors ("5y", "10s" → `5Y`, `10Y`). |
| `store.py` | `Store` holds the event log, the board and the messages seen so far. `Board` is the current market, rebuilt from events. |
| `rules.py` | Deterministic rules that run after every message: staleness (30 minutes) and crossed-market conflict flags. |
| `chatio.py` | Reads and writes chat files (JSONL). Bad lines fail with the file name and line number. |
| `render.py` | Draws the board in the terminal: quotes, axes, client interest and open flags. |
| `pipeline.py` | `Pipeline` runs message → agent → rules. Also holds `AgentResult`, `Usage` (token counts and cost), `NullAgent` (the dummy) and the run-output writer. |
| `cli.py` | The `brokerchat` command and its three subcommands. |

### `src/brokerchat/agent/`: the Claude part

| File | Responsibility |
|------|----------------|
| `loop.py` | `ClaudeAgent`, the hand-written tool-use loop. One fresh conversation per chat message. It handles API errors, refusals and the 8-call limit. |
| `tools.py` | The 7 tool definitions Claude sees (`lookup_instrument`, `get_quotes`, `upsert_quote`, `set_quote_status`, `record_axe`, `record_interest`, `raise_flag`). `ToolExecutor` validates each call and turns it into events. |
| `prompt.py` | The system prompt (desk conventions plus the glossary, cached across calls) and the per-message context (board, recent room history, new message). **This is the main file to edit when tuning the agent.** |

### `src/brokerchat/generator/` and evaluation

| File | Responsibility |
|------|----------------|
| `generator/builder.py` | `DayBuilder` writes chat one operation at a time (quote, two_markets, refer, trade, shift, unch, same_again, axe, interest, chatter, and three kinds of deliberately ambiguous message) and tracks the board a perfect agent should end with. Also loads scenario YAML files. |
| `generator/random_day.py` | Generates a random but realistic trading day from a seed. |
| `expected.py` | The ground-truth types (`ExpectedBoard` and related) and reading and writing a case: `name.jsonl` plus `name.expected.json`. |
| `scoring.py` | Compares the agent's board with the expected one and computes the PRD success metrics (S1–S4). |
| `evalrun.py` | Runs the agent over the golden set, then writes and prints the results table. |

### Data and other folders

| Path | What's there |
|------|--------------|
| `domain/glossary.yaml` | Instruments, aliases, quoting units and valid tenors. **Provisional: needs review by a real broker.** |
| `examples/handwritten.jsonl` | 20 hand-written chat messages covering most of the shorthand. |
| `scenarios/*.yaml` | Hand-written test scenarios (`basic_flow`, `context_and_ambiguity`). |
| `evals/golden/` | The golden set: 10 random days plus the 2 scenarios, each with its expected board. Regenerate it with the `generate` command in the plan, Task 8 Step 7. |
| `evals/runs/` | One folder per eval run (git-ignored). `summary.json` holds the scores and each case gets its own folder. |
| `tests/` | Mirrors `src/`. `tests/agent/fakes.py` is a fake Claude client so agent tests are free. `tests/live/` holds the only test that calls the real API; it is skipped unless `BROKERCHAT_LIVE=1`. |

## Reading eval results

| Metric | Meaning | Target |
|--------|---------|--------|
| `s1_quote_field_accuracy` | Share of quote fields (bid, offer, unit, sizes) that are correct | ≥ 0.95 |
| `s2_status_accuracy` | Share of quotes with the right final status | ≥ 0.95 |
| `s3_unsupported_entries` | Board entries citing a message that doesn't exist | 0 |
| `s4_flag_recall` | Share of deliberately ambiguous messages the agent flagged | ≥ 0.90 |
| `flag_precision` | How many of the agent's flags were warranted (approximate) | — |
| `spurious_quotes` | Quotes on the board that shouldn't be there | — |

To see why a case failed, open `evals/runs/<timestamp>/<case>/events.jsonl`. Each event names the chat message that caused it.

## Changing things

- **Teach it new shorthand:** add an alias to `domain/glossary.yaml`, or a convention to the system prompt in `agent/prompt.py`. Then run an eval to check nothing got worse.
- **Add a test scenario:** write a YAML file in `scenarios/` using the operations in `generator/builder.py` (see `OPS`), then regenerate the golden set.
- **Tune the agent:** change one thing at a time (prompt wording, a tool description, or `--effort`), run `brokerchat eval`, and record the result in `evals/RESULTS.md`.
