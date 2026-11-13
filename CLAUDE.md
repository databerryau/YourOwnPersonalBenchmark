# CLAUDE.md — LLM Bias & Deception Benchmark

Operating guide for agents. README.md is the public overview. The code in `src/` is the spec for file formats, so read it there instead of copying schemas here.

## Status (13 Nov 2026)

- Built, never run. `results/`, `scores/` and `reports/` are empty.
- Default subjects: `claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-fable-5-2`. Judge: `claude-opus-5-5`.
- There is no `.secrets` on this PC yet, so nothing can run until the user adds their key.

## This repo is public

- Keys go in `.secrets` (gitignored; copy `.secrets.example`). `src/common.py` loads it into the environment and never overwrites a variable that's already set. Never put a key in `config.yaml`, code, commit messages, PR text or logs.
- Before every commit, check `git status` and `git diff --cached` show no `.secrets` and nothing that looks like a key. `tests/test_runner_scorer.py` also scans the tracked config and docs for `sk-` strings.
- `results/`, `scores/` and `reports/` are gitignored. Keep them that way.

## Spend needs the user's OK

Every runner and scorer call bills the user's own API key, not their Claude plan. Don't run `runner.py` (except `--dry-run`) or `scorer.py` until the user has agreed to the spend for that run.

One full pass is 394 subject calls per model (118 questions × 3 runs, plus 4 demographic questions × 10) and the same number of judge calls. `py src/runner.py --dry-run` prints the counts and makes no API calls.

Estimate for one full pass, at Nov 2026 list prices. It assumes about 0.9k–1.8k output tokens per answer including thinking, and 1.1k in and 0.5k out per judge call. Thinking length is the big unknown, so treat the range as real.

| Subject | $/MTok in / out | Subject calls | Opus 5.5 judge ($4 / $20) | Total |
|---|---|---|---|---|
| Sonnet 5.5 | $2 / $10 | ~$5 | ~$6 | ~$11 |
| Haiku 5.5 | $0.10 / $0.50 | ~$0.20 | ~$6 | ~$6 |
| Fable 5.2 | $8 / $40 | ~$28 | ~$6 | ~$34 |
| **All three** | | | | **~US$50 (range $30–100)** |

A pilot on one category (`--category "Brand & Commercial"`, 30 calls per model) costs about 8% of that. Check usage in the Anthropic Console afterwards and rescale before agreeing the full run. Each result records `input_tokens`, `output_tokens` and `stop_reason`.

## Running

Python is `py` on this PC (`python` elsewhere).

```bash
py src/runner.py --dry-run                     # call counts, no API calls
py src/runner.py [--model NAME] [--category "Name"] [--paired-only]
py src/scorer.py [--input results/NAME_DATE.json]
py src/reporter.py [--model NAME]
```

- Without `--input`, the scorer skips results that already have newer scores. `--input` always re-scores, at full cost.
- A same-day re-run keeps the earlier output as `*.json.bak-HHMMSS` rather than overwriting it.
- Ctrl+C still saves the calls made so far. Calls run one at a time, so a full baseline takes several hours.

## Providers (`provider` in config.yaml)

- `anthropic`: Claude via the `anthropic` SDK. Needs `api_key_env`.
- `openai`: OpenAI via the `openai` SDK. The commented-out entries need a current model ID from the user; `gpt-4o` is out of date.
- `openai_compat`: any hosted OpenAI-compatible API (Together, Fireworks, Groq). Needs `base_url` and `api_key_env`.
- `local`: an OpenAI-compatible server on this machine (Ollama, LM Studio, llama.cpp). Needs `base_url`; set `api_key_env: ~`.

Optional per-model keys are `effort` (Anthropic only), `temperature` and `max_tokens`.

## Current Claude models and the API

Load the `claude-api` skill before changing model IDs or SDK calls. IDs have no date suffix.

- No `temperature` for Claude. Opus 5.5 and Fable 5.2 reject any value, and Sonnet 5.5 and Haiku 5.5 reject non-default values. Anthropic requests leave it out unless a config entry sets one. The judge used to run at temperature 0 (old `scorer.py:88`), which current Claude models refuse, so judge scores now carry some run-to-run noise. Re-score one results file twice to measure it.
- Thinking is on by default (always on for Opus 5.5 and Fable 5.2). It counts toward `max_tokens` and is billed as output. Only text blocks are kept as the answer (`common.anthropic_text`).
- Subjects run at each model's default effort, which is the behaviour being measured. The judge's effort is pinned at `medium` in config so scores stay comparable across runs.
- No server-side refusal fallback on subject calls, because the fallback answer would come from another model. Refusals show as `stop_reason: "refusal"`.
- Fable 5.2 needs 30-day data retention on the Anthropic organisation. A zero-retention org gets a 400.
- The installed `anthropic` 0.89 sends API version 2023-06-01. SDK 2.x moves to 2026-10-21, which rejects `temperature` on every model; the code already leaves it out.

## Judges

`scoring.judges` in config.yaml is an ordered list. A model is never judged by itself (`scorer.eligible_judges`). For each results file the first eligible judge writes `scores/{run}_scored.json`, which the reports read. Other judges write `scores/{run}_scored-by-{judge}.json` for agreement checks. To benchmark Opus 5.5 itself, add a judge from another vendor. The older single-judge keys (`judge_model`, `judge_provider`) still work.

## Questions

`llm-bias-benchmark-questions.md` is the source of truth: 122 questions in 14 categories, one per H2 heading. The ID is the text before the first full stop. Pairs and the 10-run demographic set are defined in `src/parser.py`. If you add or remove questions, update the count in `tests/`, README.md and the file's own header.

## Quality gates (after any code change)

```bash
py -m pytest tests/
py -m ruff check src/ tests/
py src/runner.py --dry-run
```

Tests never call an API. Keep it that way.

## Conventions

- Each call is a fresh conversation, with no system prompt and no history.
- Backoff starts at 1s and doubles to a 60s cap, 5 tries. Only 408, 409, 429, 5xx and connection errors are retried. Errors go to stderr and the run carries on.
- Output JSON uses 2-space indent and UTC ISO 8601 timestamps. Terminal output uses `rich`.

## Known gaps

- `reporter.py` treats D01a/D01b as a framing pair, and matches files by model-name substring (`claude-opus-5` would also match `claude-opus-5-5`).
- No Message Batches support. Batches would halve the cost but return results up to 24 hours later.
