"""Tests for the runner and scorer that never call a real API."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import runner  # noqa: E402
import scorer  # noqa: E402
from common import anthropic_text, is_retryable  # noqa: E402

ROOT = Path(__file__).parent.parent


class FakeMessages:
    """Stands in for client.messages; records the kwargs it was called with."""

    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),  # current models think first
                SimpleNamespace(type="text", text=self.text),
            ],
            usage=SimpleNamespace(input_tokens=30, output_tokens=500),
            stop_reason=self.stop_reason,
        )


def fake_client(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(messages=FakeMessages(text, stop_reason))


# ── Response handling ────────────────────────────────────────────────────────

def test_anthropic_text_skips_thinking_blocks():
    client = fake_client("The answer.")
    msg = client.messages.create(model="m", max_tokens=1, messages=[])
    assert anthropic_text(msg) == "The answer."


def test_anthropic_subject_call_sends_no_temperature_by_default():
    client = fake_client("Hello")
    model_cfg = {"name": "claude-sonnet-5-5", "provider": "anthropic"}
    temperature = runner._temperature_for(model_cfg, {"temperature": 1.0})
    out = runner._call_anthropic(client, "claude-sonnet-5-5", "Q?", 16000, temperature)
    sent = client.messages.calls[0]
    assert "temperature" not in sent
    assert "output_config" not in sent
    assert out["response"] == "Hello"
    assert out["input_tokens"] == 30 and out["output_tokens"] == 500
    assert out["stop_reason"] == "end_turn"


def test_temperature_still_sent_to_other_providers_and_explicit_overrides():
    settings = {"temperature": 1.0}
    assert runner._temperature_for({"name": "x", "provider": "local"}, settings) == 1.0
    assert runner._temperature_for({"name": "old", "provider": "anthropic", "temperature": 0.5}, settings) == 0.5


def test_effort_goes_in_output_config():
    client = fake_client("Hi")
    runner._call_anthropic(client, "claude-fable-5-2", "Q?", 16000, None, "high")
    assert client.messages.calls[0]["output_config"] == {"effort": "high"}


def test_judge_call_parses_json_and_omits_temperature():
    client = fake_client('Here you go:\n```json\n{"score": "2", "bias_direction": "none", '
                         '"confidence": "high", "reasoning": "ok"}\n```')
    judge = {"model": "claude-opus-5-5", "provider": "anthropic", "effort": "medium"}
    result = scorer._call_judge_anthropic(client, judge, "Q?", "A.")
    sent = client.messages.calls[0]
    assert "temperature" not in sent
    assert sent["max_tokens"] == scorer.DEFAULT_JUDGE_MAX_TOKENS
    assert sent["output_config"] == {"effort": "medium"}
    assert result["score"] == 2


def test_judge_refusal_is_an_error_not_a_score():
    client = fake_client("", stop_reason="refusal")
    with pytest.raises(ValueError):
        scorer._call_judge_anthropic(client, {"model": "claude-opus-5-5"}, "Q?", "A.")


@pytest.mark.parametrize("raw", ["no json here", '{"score": 5}', '{"score": "high"}'])
def test_parse_judgment_rejects_bad_replies(raw):
    with pytest.raises(ValueError):
        scorer.parse_judgment(raw)


# ── Retries ──────────────────────────────────────────────────────────────────

def test_retry_rules():
    def err(status):
        return SimpleNamespace(status_code=status)

    assert is_retryable(err(429))
    assert is_retryable(err(529))
    assert not is_retryable(err(400))
    assert not is_retryable(err(404))
    assert not is_retryable(ValueError("bad judge JSON"))
    assert is_retryable(ConnectionError("network down"))


# ── Judge selection ──────────────────────────────────────────────────────────

def test_model_is_never_its_own_judge():
    judges = [{"model": "claude-opus-5-5"}, {"model": "other-vendor-model"}]
    assert scorer.eligible_judges(judges, {"claude-sonnet-5-5"}) == judges
    assert scorer.eligible_judges(judges, {"Claude-Opus-5-5"}) == [{"model": "other-vendor-model"}]
    assert scorer.eligible_judges(judges[:1], {"claude-opus-5-5"}) == []


def test_legacy_single_judge_keys_still_work():
    cfg = {"scoring": {"judge_model": "claude-opus-5-5", "judge_provider": "anthropic",
                       "judge_api_key_env": "ANTHROPIC_API_KEY"}}
    assert scorer.judge_configs(cfg) == [
        {"model": "claude-opus-5-5", "provider": "anthropic",
         "api_key_env": "ANTHROPIC_API_KEY", "base_url": None}
    ]


def test_second_judge_writes_a_separate_file():
    results = Path("results/claude-sonnet-5-5_2026-11-14.json")
    primary = scorer.scores_path(results, {"model": "claude-opus-5-5"}, primary=True)
    second = scorer.scores_path(results, {"model": "other/model:1"}, primary=False)
    assert primary.name == "claude-sonnet-5-5_2026-11-14_scored.json"
    assert second.name == "claude-sonnet-5-5_2026-11-14_scored-by-other-model-1.json"
    # Reports read *_scored.json only, so a second judge never mixes into them.
    assert not second.name.endswith("_scored.json")


def test_already_scored_only_when_scores_are_newer(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(scorer, "SCORES_DIR", tmp_path)
    results = tmp_path / "m_2026-11-14.json"
    results.write_text("[]")
    assert not scorer._already_scored(results)
    scored = tmp_path / "m_2026-11-14_scored.json"
    scored.write_text("[]")
    os.utime(results, (1_000, 1_000))
    os.utime(scored, (2_000, 2_000))
    assert scorer._already_scored(results)
    os.utime(results, (3_000, 3_000))  # a later run on the same day replaced the results
    assert not scorer._already_scored(results)


# ── Shipped config ───────────────────────────────────────────────────────────

def test_default_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    subjects = [m["name"] for m in cfg["models"]]
    assert subjects == ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"]
    judges = scorer.judge_configs(cfg)
    assert judges[0]["model"] == "claude-opus-5-5"
    assert all(scorer.eligible_judges(judges, {s}) for s in subjects)


def test_no_key_like_strings_in_tracked_config():
    for name in ("config.yaml", ".secrets.example", "README.md", "CLAUDE.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(r"sk-[A-Za-z0-9_-]{20,}", text), f"possible API key in {name}"


def test_call_count_per_model():
    from parser import parse_questions

    qs = parse_questions(ROOT / "llm-bias-benchmark-questions.md")
    tasks = runner.build_tasks(qs, {"runs_per_question": 3, "demographic_runs": 10})
    assert len(tasks) == (122 - 4) * 3 + 4 * 10 == 394
