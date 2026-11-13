"""Helpers shared by runner.py and scorer.py: secrets, retries, response text, file safety."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
SECRETS_PATH = PROJECT_ROOT / ".secrets"

# HTTP statuses worth retrying. Other 4xx errors (bad model ID, bad parameter,
# auth) fail the same way every time, so retrying only wastes time.
RETRYABLE_STATUS = {408, 409, 429}


def load_secrets(path: Path = SECRETS_PATH) -> None:
    """Load KEY=value pairs from .secrets into os.environ (never overwrites existing vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def is_retryable(exc: Exception) -> bool:
    """True for rate limits, server errors and connection problems."""
    status = getattr(exc, "status_code", None)
    if status is None:
        # Connection errors have no status. ValueError covers bad judge JSON,
        # which a retry would only pay for again.
        return not isinstance(exc, ValueError)
    return status in RETRYABLE_STATUS or status >= 500


def call_with_backoff(call_fn, *args, max_retries: int = 5, label: str = "Retry", **kwargs):
    """Exponential backoff: 1s doubling to a 60s cap, at most `max_retries` attempts."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return call_fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries - 1 or not is_retryable(exc):
                raise
            console.print(
                f"[yellow]{label} {attempt + 1}/{max_retries}: {exc}. Waiting {delay:.0f}s...[/yellow]"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def anthropic_text(msg) -> str:
    """Join the text blocks of a Messages API response.

    Current Claude models think by default, so content[0] is often a thinking
    block with no `.text`. Only the visible answer is kept.
    """
    return "".join(
        block.text for block in msg.content if getattr(block, "type", None) == "text"
    )


def same_model(a: str | None, b: str | None) -> bool:
    return bool(a and b) and a.strip().lower() == b.strip().lower()


def safe_name(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def backup_if_exists(path: Path) -> None:
    """Rename an existing output file so a same-day re-run never overwrites paid-for results."""
    if path.exists():
        stamp = datetime.now().strftime("%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        path.rename(backup)
        console.print(f"[yellow]Kept the earlier file as {backup.name}[/yellow]")
