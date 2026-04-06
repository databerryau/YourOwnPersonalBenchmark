"""Send benchmark questions to LLM APIs and collect raw responses.

Supports:
  provider: anthropic       — Anthropic API (claude-*)
  provider: openai          — OpenAI API
  provider: openai_compat   — Any OpenAI-compatible hosted API (Together, Fireworks, etc.)
  provider: local           — Local OpenAI-compatible server (Ollama, LM Studio, llama.cpp)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from parser import parse_questions

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
SECRETS_PATH = PROJECT_ROOT / ".secrets"
QUESTIONS_PATH = PROJECT_ROOT / "llm-bias-benchmark-questions.md"


# ── Secrets loader ────────────────────────────────────────────────────────────

def load_secrets(path: Path = SECRETS_PATH) -> None:
    """Load KEY=value pairs from .secrets into os.environ (does not overwrite existing vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Client factories ──────────────────────────────────────────────────────────

def _get_api_key(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    return os.environ.get(api_key_env)


def _make_client(model_cfg: dict):
    """Return (client, call_fn) for the given model config."""
    provider = model_cfg["provider"]
    api_key_env = model_cfg.get("api_key_env")
    base_url = model_cfg.get("base_url")

    if provider == "anthropic":
        api_key = _get_api_key(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Anthropic API key not found. Set {api_key_env!r} in .secrets or environment."
            )
        import anthropic
        return anthropic.Anthropic(api_key=api_key), _call_anthropic

    elif provider in ("openai", "openai_compat", "local"):
        import openai
        api_key = _get_api_key(api_key_env) or "local"  # local servers often ignore the key
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        elif provider == "local":
            raise ValueError(
                f"Model {model_cfg['name']!r} has provider=local but no base_url in config.yaml"
            )
        if provider in ("openai", "openai_compat", "local") and api_key_env and not _get_api_key(api_key_env):
            raise EnvironmentError(
                f"API key not found. Set {api_key_env!r} in .secrets or environment."
            )
        return openai.OpenAI(**kwargs), _call_openai

    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ── API callers ───────────────────────────────────────────────────────────────

def _call_anthropic(
    client,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    t0 = time.monotonic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": question}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    response_text = msg.content[0].text
    tokens_used = msg.usage.input_tokens + msg.usage.output_tokens
    return response_text, tokens_used, latency_ms


def _call_openai(
    client,
    model: str,
    question: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, int]:
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": question}],
    )
    latency_ms = int((time.monotonic() - t0) * 1000)
    response_text = resp.choices[0].message.content
    tokens_used = getattr(resp.usage, "total_tokens", 0) or 0
    return response_text, tokens_used, latency_ms


def _call_with_backoff(call_fn, *args, max_retries: int = 5):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return call_fn(*args)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            console.print(
                f"[yellow]Retry {attempt + 1}/{max_retries}: {exc}. Waiting {delay:.0f}s...[/yellow]"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_model(
    model_cfg: dict,
    questions: list[dict],
    settings: dict,
    date_str: str,
) -> Path:
    model_name = model_cfg["name"]
    runs_per_q = settings.get("runs_per_question", 3)
    demo_runs = settings.get("demographic_runs", 10)
    temperature = settings.get("temperature", 1.0)
    max_tokens = settings.get("max_tokens", 2048)

    client, call_fn = _make_client(model_cfg)

    tasks = []
    for q in questions:
        n_runs = demo_runs if q["is_demographic"] else runs_per_q
        for run_num in range(1, n_runs + 1):
            tasks.append((q, run_num))

    results = []
    safe_model = model_name.replace("/", "-").replace(":", "-")
    out_path = RESULTS_DIR / f"{safe_model}_{date_str}.json"

    console.rule(f"[bold blue]Running {model_name} ({model_cfg['provider']})")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(model_name, total=len(tasks))

        for q, run_num in tasks:
            progress.update(
                task,
                description=f"[cyan]{q['id']}[/cyan] run {run_num}",
                advance=0,
            )
            try:
                response_text, tokens_used, latency_ms = _call_with_backoff(
                    call_fn,
                    client,
                    model_name,
                    q["question"],
                    max_tokens,
                    temperature,
                )
            except Exception as exc:
                console.print(
                    f"[red]ERROR {q['id']} run {run_num}: {exc}[/red]",
                    file=sys.stderr,
                )
                response_text = ""
                tokens_used = 0
                latency_ms = 0

            results.append(
                {
                    "question_id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "model": model_name,
                    "provider": model_cfg["provider"],
                    "run_number": run_num,
                    "response": response_text,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tokens_used": tokens_used,
                    "latency_ms": latency_ms,
                }
            )
            progress.advance(task)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved {len(results)} results → {out_path}[/green]")
    return out_path


def main():
    load_secrets()

    parser = argparse.ArgumentParser(description="Run LLM bias benchmark")
    parser.add_argument("--model", help="Run only this model name")
    parser.add_argument("--paired-only", action="store_true", help="Only run paired questions")
    parser.add_argument("--category", help="Only run questions in this category")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    questions = parse_questions(args.questions)

    if args.paired_only:
        questions = [q for q in questions if q["paired_with"] is not None]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]

    console.print(f"[bold]Loaded {len(questions)} questions[/bold]")

    models = cfg["models"]
    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            console.print(f"[red]Model {args.model!r} not found in config[/red]")
            sys.exit(1)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for model_cfg in models:
        run_model(model_cfg, questions, cfg["settings"], date_str)


if __name__ == "__main__":
    main()
