# Broker Chat Quote Parser — Prototype PRD

**Date:** 2026-10-02
**Status:** Draft for review
**Owner:** Pierce Keane

## 1. Summary

Build a Claude-powered agent that reads messy CDS broker chat in order and keeps a live, structured quote board. The board holds levels, sizes, axes, client interest, and the status of each quote. v1 is a local prototype that runs only on synthetic chat. It succeeds when it processes a realistic day of chat and a working CDS broker says the board is right.

The main goal is learning to build LLM agents: tool design, prompting, state handling and evaluation. A working demo for a broker and portfolio value are secondary goals.

## 2. Goals and non-goals

**Goals (v1)**

- Parse CDS chat shorthand into structured quotes: instrument, tenor, bid/offer, size, source, time.
- Resolve context across messages: "same again in 10y", "refer", "off that", "+1".
- Keep quote status current (live, referred, traded, stale) and flag conflicts.
- Ask instead of guessing: unclear messages produce a flag with a specific question.
- Trace every board entry back to the chat messages that produced it.
- Include a repeatable evaluation harness that scores the agent against ground truth.

**Non-goals (v1)**

- Real or confidential chat data. Synthetic only.
- Live connections to Bloomberg IB, Symphony or any other chat platform.
- Run generation, interest matching and trade-recap checking. These come later and build on the same data model.
- Multi-user deployment, auth or hosting.
- Pricing analytics (curves, fair value, spread-to-upfront conversion).

## 3. Users and success criteria

**Primary user (eventual):** a CDS broker who watches many chat rooms and needs a single accurate picture of the market.
**v1 user:** the builder (for development and eval) and one broker friend (for a demo review).

**Success criteria**

| # | Criterion | Measure |
|---|-----------|---------|
| S1 | Quotes extracted correctly | ≥ 95% field-level accuracy on the golden eval set |
| S2 | Status kept current | ≥ 95% of quotes have the correct final status at end of day |
| S3 | No invented quotes | 0 board entries with no supporting source message |
| S4 | Ambiguity is flagged, not guessed | ≥ 90% recall on messages labelled "ambiguous" in the golden set |
| S5 | Domain sign-off | Broker friend reviews a replayed synthetic day and says the board is right |

The S1–S4 targets are starting points. Revisit them once the first baseline eval run exists.

## 4. Domain primer

The agent must handle the shorthand below. **This list is the builder's best understanding. It should be checked with the broker friend before the golden set is final.**

| Chat | Meaning |
|------|---------|
| `ITRX Main 5y 58/58.5` | iTraxx Europe Main, 5-year: bid 58bp, offer 58.5bp |
| `XO 5y 302/304 10x10` | iTraxx Crossover 5y, 302/304bp, €10mm each side |
| `CDX IG 5y 52.25 bid` | One-sided bid |
| `VW 5y 120/125` | Single name (Volkswagen), 5y, in bp |
| `CDX HY 5y 104.5/104.625` | HY index quoted in price, not spread |
| `axed to buy Main 5y` / `who's axed XO?` | Axe (desire to buy or sell protection), or a question about axes |
| `refer` / `off that` | Pull the previous quote. It is no longer live |
| `done` / `lifted` / `hit` / `trades 58.5` | Quote traded |
| `same again in 10y` | Repeat the last quote structure for another tenor |
| `+1` / `tighter` / `unch` | Relative update to the previous level |
| `S42` / `on the run` | Index series |

Glossary data (aliases, index families, tenors, price-vs-spread quoting) lives in a version-controlled file (`domain/glossary.yaml`). It is not hard-coded in prompts.

## 5. Data model

Every change is stored as an append-only **event**. The **board** is a projection rebuilt from events. This makes the system auditable and replayable, and lets evaluation compare events as well as the final board.

**ChatMessage**: `id`, `ts`, `room`, `sender`, `text`

**Instrument**: `kind` (index | single_name), `name` (canonical, e.g. `ITRAXX_EUR_MAIN`), `series` (optional), `tenor` (e.g. `5Y`)

**Quote**

| Field | Type | Notes |
|-------|------|-------|
| `quote_id` | str | stable id |
| `instrument` | Instrument | |
| `bid`, `offer` | Decimal or null | one may be null |
| `unit` | `bp` or `price` | |
| `bid_size`, `offer_size` | Decimal or null | in mm, currency implied by instrument |
| `source` | str | sender or counterparty as shown |
| `status` | live, referred, traded, stale | |
| `updated_ts` | datetime | |
| `source_message_ids` | list[str] | required, never empty |
| `confidence` | high or low | low adds a flag |

**Axe**: `instrument`, `direction` (buy_protection, sell_protection), `size`, `source`, `source_message_ids`

**Interest**: `client`, `instrument`, `direction`, `note`, `source_message_ids`

**Flag**: `flag_id`, `kind` (ambiguous, conflict, unknown_instrument), `question`, `message_ids`, `resolved` (bool), `resolution`

**Event**: `event_id`, `ts`, `type` (quote_upserted, quote_status_changed, axe_recorded, interest_recorded, flag_raised, flag_resolved), `payload`, `caused_by_message_id`

## 6. Agent architecture

**Choice:** a hand-written tool-use loop over the Claude Messages API, in Python with the official `anthropic` SDK. A manual loop is chosen over the SDK Tool Runner because owning the loop is the main learning goal. Switching to the Tool Runner later is a small change.

**Model:** `claude-opus-5-5` with adaptive thinking, `tool_choice: auto`, and `strict: true` on every tool. Effort starts at `medium` and is tuned with the eval harness. Server-side refusal fallbacks are enabled.

**Per message flow**

1. **Ingest.** The next `ChatMessage` arrives from the replay feed.
2. **Build context.** Combine:
   - system prompt and glossary (stable, prompt-cached)
   - the last N messages from the same room (default 20)
   - the board slice relevant to the room, plus recently touched instruments
3. **Agent loop.** Claude calls tools until it ends its turn. Messages with no market content (for example "morning all") produce no tool calls.
4. **Apply.** Each tool call is validated and written as an event, then the board projection updates.
5. **Sweep.** A deterministic pass marks quotes `stale` once they pass a configurable age (default 30 minutes). Staleness is not decided by the LLM.

**Tools**

| Tool | Purpose |
|------|---------|
| `lookup_instrument(text)` | Resolve an alias ("Main", "XO", "VW") to a canonical instrument via the glossary. Deterministic. |
| `get_quotes(instrument?, room?)` | Read current board entries to resolve "off that" or "same again" |
| `upsert_quote(...)` | Create or update a quote. Must cite `source_message_ids` |
| `set_quote_status(quote_id, status, message_id)` | Refer, trade or re-activate |
| `record_axe(...)` | Record an axe |
| `record_interest(...)` | Record client interest |
| `raise_flag(kind, question, message_ids)` | Ask instead of guessing |

**Guardrails**

- Tool inputs are validated with Pydantic. Invalid input returns an `is_error` tool result so the model can correct itself.
- `upsert_quote` rejects calls with no cited message ids. This enforces S3.
- Each message has a cap on loop iterations (default 8). Hitting the cap raises a flag automatically.

## 7. Synthetic data and evaluation

**Generator.** A scenario script describes a trading day: instruments, a level path, participants, and events such as quotes, refers, trades, axes and ambiguous messages. Each scenario renders chat text from templates with controlled noise (typos, casing, abbreviations, interleaved rooms). Claude can optionally paraphrase lines for variety. **Because the scenario is the source of truth, every generated chat has an exact expected event log and final board.**

**Golden set.** About 10 hand-checked scenarios, roughly 300 messages in total, with a share deliberately labelled ambiguous. They are stored as `evals/golden/*.jsonl` with matching `*.expected.json` files.

**Scorer.** Compares the agent's output with the expected output and reports:

- quote field accuracy (S1) and final status accuracy (S2)
- unsupported entries (S3)
- flag recall and precision on ambiguous messages (S4)
- token usage and estimated cost per message

Every prompt or tool change gets an eval run. Results are written to `evals/runs/<timestamp>/` so runs can be compared.

## 8. Interface

- **Input:** a JSONL chat file, one `ChatMessage` per line.
- **CLI:**
  - `replay <file> [--speed 10x]`: streams messages and renders a live board in the terminal, with quotes, axes and open flags in separate panels
  - `eval [--set golden]`: runs the scorer
  - `generate <scenario>`: produces synthetic chat plus ground truth
- **Output:** a final board snapshot (JSON), the event log (JSONL) and a flag list.
- **Flag resolution in v1:** flags are listed for review but don't block the replay. Answering flags interactively is a stretch goal.

## 9. Ambiguity, conflicts and errors

| Situation | Behaviour |
|-----------|-----------|
| Unknown instrument alias | `raise_flag(unknown_instrument)`. No quote is created |
| Context reference with no clear target ("off that" with two live candidates) | `raise_flag(ambiguous)` naming the candidates |
| Two sources quote the same instrument at crossed levels | Both are kept, and `raise_flag(conflict)` is raised |
| Quote too old | Deterministic sweep sets it to `stale` |
| API error or rate limit | SDK retries. After that, the message is marked unprocessed and the replay continues |
| Refusal stop reason | Logged. The message is marked unprocessed |

## 10. Milestones and tech stack

**Stack:** Python 3.12, `uv`, `anthropic`, `pydantic`, `rich` (terminal board), `pytest`. The repo is on GitHub, and no real chat data is ever committed.

| # | Milestone | Done when |
|---|-----------|-----------|
| M0 | Repo setup and domain glossary | Glossary covers section 4. 20 hand-written example messages exist |
| M1 | Data model, event log, board projection and terminal render | Board renders from a hand-written event log. No LLM yet |
| M2 | Generator, golden set and scorer | Scorer runs end to end against a dummy agent |
| M3 | Agent loop with single-message parsing | Baseline eval scores recorded |
| M4 | Context resolution, flags and staleness | S1–S4 targets met on the golden set |
| M5 | Demo | Broker friend reviews a replayed day (S5) |

## 11. Open questions and risks

- **Real chat format.** Which platform (Bloomberg IB, Symphony, other) and what the shorthand really looks like. Check with the friend before finalising the golden set.
- **Domain accuracy.** The section 4 conventions may be wrong or incomplete, for example price-quoting rules or size conventions. Mitigation: a friend review of the glossary at M0.
- **Cost and latency per message.** Unknown until M3. Mitigations if needed: prompt caching, lower effort, smaller context windows, a cheaper model for trivial messages.
- **Hallucinated quotes.** Mitigated by required source citations (S3) and eval.
- **Compliance.** Any future use on real chat raises data-handling and compliance questions. These are out of scope for v1 but must be answered before any real data is touched.
