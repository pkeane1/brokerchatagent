# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git

Never run `git commit` (or push) in this repo. Leave all changes uncommitted for the user to review and commit themselves, even when a skill or workflow says to commit.

## Commands

Python 3.12+ via `uv` (installed at `~/.local/bin`; prefix `export PATH="$HOME/.local/bin:$PATH"` in bash if `uv` isn't found).

- Tests: `uv run pytest -q` · single test: `uv run pytest tests/agent/test_tools.py::test_upsert_creates_quote -v`
- Live API test (costs money): `BROKERCHAT_LIVE=1 uv run pytest tests/live -v -s`
- CLI: `uv run brokerchat replay|generate|eval` — default agent is `claude` (real API, costs money); pass `--agent null` for free runs. Ask the user before any command that calls the real API.

`README.md` maps every module; the PRD (`docs/superpowers/specs/`) is the source of truth for scope, the plan (`docs/superpowers/plans/`) for build steps.

## What this is

A Claude tool-use agent that reads CDS broker chat message by message and maintains a structured live quote board (quotes, axes, client interest, flags). Learning to build LLM agents is the primary goal, so the agent loop is hand-written over the Messages API rather than using the SDK Tool Runner.

## Architectural invariants (from the PRD)

These cut across modules and are easy to violate without reading the spec:

- **Event-sourced state.** Every change is an append-only event; the board is a projection rebuilt from the event log. Never mutate board state directly — tools emit events.
- **Provenance is mandatory.** Every quote/axe/interest carries `source_message_ids`; `upsert_quote` must reject calls without them. Zero unsupported board entries is a success criterion.
- **LLM vs deterministic split.** Claude handles language and context resolution ("same again in 10y", "off that"). Instrument alias lookup (`domain/glossary.yaml`) and staleness marking are deterministic code, not LLM decisions.
- **Flag, don't guess.** Ambiguous references, unknown instruments, and crossed quotes produce a `Flag` via `raise_flag` rather than a best guess.
- **Model config.** `claude-opus-5-5`, adaptive thinking, `tool_choice: auto`, `strict: true` on all tools, effort tuned via the eval harness. Tool inputs validated with Pydantic; invalid input returns an `is_error` tool result.
- **Eval-driven changes.** Prompt or tool changes should be followed by an eval run against the golden set (`evals/golden/`), with results in `evals/runs/<timestamp>/`.

## Data rules

Synthetic chat only. Never commit real broker chat or anything resembling confidential market data. Domain conventions in PRD §4 are unverified and pending review by a practising CDS broker — treat them as provisional.
