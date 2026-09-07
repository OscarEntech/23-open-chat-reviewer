"""Pricing configuration behaviour for the token cost panel."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from chatreview.token_panel import (
    DEFAULT_PRICING,
    load_pricing,
    pricing_age_days,
    resolve_timezone,
)


def _write_pricing(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "token-pricing.json"
    path.write_text(payload, encoding="utf-8")
    return path


def test_load_pricing_falls_back_when_the_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chatreview.token_panel.PRICING_PATH", tmp_path / "missing.json")

    config = load_pricing()

    assert config["loaded_from"] == "built in defaults"
    assert "load_error" not in config
    assert config["prices_usd_per_mtok"] == DEFAULT_PRICING["prices_usd_per_mtok"]


def test_load_pricing_reports_a_malformed_file_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pricing(tmp_path, "{not valid json")
    monkeypatch.setattr("chatreview.token_panel.PRICING_PATH", path)

    config = load_pricing()

    assert "load_error" in config
    assert config["loaded_from"] == "built in defaults"
    assert config["rate_per_usd"] == DEFAULT_PRICING["rate_per_usd"]


def test_load_pricing_merges_operator_values_over_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_pricing(
        tmp_path,
        json.dumps(
            {
                "as_at": "2026-01-15",
                "rate_per_usd": 1.5,
                "prices_usd_per_mtok": {
                    "claude-opus-5": {"inp": 1.0, "out": 2.0, "cw5m": 3.0, "cw1h": 4.0, "read": 0.1}
                },
            }
        ),
    )
    monkeypatch.setattr("chatreview.token_panel.PRICING_PATH", path)

    config = load_pricing()

    assert config["loaded_from"] == str(path)
    assert config["rate_per_usd"] == 1.5
    assert config["as_at"] == "2026-01-15"
    assert config["prices_usd_per_mtok"]["claude-opus-5"]["inp"] == 1.0
    assert config["source"] == DEFAULT_PRICING["source"]


def test_pricing_age_days_measures_staleness_and_tolerates_a_bad_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATREVIEW_TIMEZONE", "UTC")
    stamped = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=3)

    assert pricing_age_days({"as_at": stamped.isoformat()}) == 3
    assert pricing_age_days({"as_at": "not-a-date"}) is None
    assert pricing_age_days({}) is None


def test_resolve_timezone_prefers_the_project_variable_and_falls_back_to_utc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHATREVIEW_TIMEZONE", "Australia/Perth")
    assert resolve_timezone() == "Australia/Perth"

    monkeypatch.setenv("CHATREVIEW_TIMEZONE", "Not/AZone")
    assert resolve_timezone() == "UTC"

    monkeypatch.delenv("CHATREVIEW_TIMEZONE", raising=False)
    monkeypatch.delenv("TZ", raising=False)
    assert resolve_timezone() == "UTC"
