"""
Unit tests for dce_orchestrator's channel-exclusion filter.

Regression coverage for a real incident: EXCLUDED_CHANNELS is documented as
matching by channel ID *or* name (docker-compose.yml passes the name the user
actually sets, e.g. "bot-zone"), but the filter briefly regressed to
ID-only matching and silently exported an excluded channel into the training
archive. These tests exercise evaluate_and_export's filter directly against
both an excluded name and an excluded ID, with DCE and disk I/O mocked out.
"""
import os
import tempfile
from unittest.mock import patch

import pytest

import dce_orchestrator


CHANNELS = [
    {"id": "1504309927651573771", "name": "bot-zone"},
    {"id": "426225897163587589", "name": "fartposting"},
]


@pytest.fixture
def temp_state_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("channel_state.ARCHIVE_STATE_DIR", tmpdir)
        monkeypatch.setattr(
            "channel_state.CHANNEL_STATE_FILE", os.path.join(tmpdir, "channel_state.json")
        )
        yield tmpdir


def _run_evaluate(monkeypatch, excluded, temp_state_dir):
    monkeypatch.setattr(dce_orchestrator, "EXCLUDED_CHANNELS", excluded)
    exported_channel_ids = []

    def fake_export_channel(channel_id, after=None, before=None):
        exported_channel_ids.append(channel_id)
        return "/out/fake"

    with patch.object(dce_orchestrator, "export_channel", side_effect=fake_export_channel), \
         patch.object(dce_orchestrator, "parse_dce_export_directory", return_value={}), \
         patch.object(dce_orchestrator, "append_messages", return_value=0):
        dce_orchestrator.evaluate_and_export(CHANNELS)

    return exported_channel_ids


class TestExcludedChannelsFilter:
    def test_excludes_by_name(self, monkeypatch, temp_state_dir):
        """EXCLUDED_CHANNELS=['bot-zone'] must exclude by channel name, not just id."""
        exported = _run_evaluate(monkeypatch, ["bot-zone"], temp_state_dir)
        assert "1504309927651573771" not in exported
        assert "426225897163587589" in exported

    def test_excludes_by_id(self, monkeypatch, temp_state_dir):
        exported = _run_evaluate(monkeypatch, ["1504309927651573771"], temp_state_dir)
        assert "1504309927651573771" not in exported
        assert "426225897163587589" in exported

    def test_no_exclusions_exports_everything(self, monkeypatch, temp_state_dir):
        exported = _run_evaluate(monkeypatch, [], temp_state_dir)
        assert set(exported) == {"1504309927651573771", "426225897163587589"}

    def test_unrelated_exclusion_leaves_channel_untouched(self, monkeypatch, temp_state_dir):
        exported = _run_evaluate(monkeypatch, ["some-other-channel"], temp_state_dir)
        assert set(exported) == {"1504309927651573771", "426225897163587589"}
