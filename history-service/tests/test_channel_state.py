"""
Unit tests for channel_state module.
"""
import json
import os
import tempfile
import pytest
from unittest.mock import patch

from channel_state import (
    load_state,
    save_state,
    get_channel,
    update_channel,
    channels_needing_export,
    _ensure_state_dir,
    CHANNEL_STATE_FILE,
)


@pytest.fixture
def temp_state_dir(monkeypatch):
    """Create a temporary state directory and patch ARCHIVE_STATE_DIR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("channel_state.ARCHIVE_STATE_DIR", tmpdir)
        monkeypatch.setattr("channel_state.CHANNEL_STATE_FILE", os.path.join(tmpdir, "channel_state.json"))
        yield tmpdir


class TestLoadState:
    def test_load_empty_state(self, temp_state_dir):
        """Test loading state when no file exists."""
        state = load_state()
        assert state == {}

    def test_load_existing_state(self, temp_state_dir, monkeypatch):
        """Test loading an existing state file."""
        state_file = os.path.join(temp_state_dir, "channel_state.json")
        expected = {
            "111": {"channel_id": "111", "channel_name": "general", "last_export_at": "2026-01-01T00:00:00Z"},
            "222": {"channel_id": "222", "channel_name": "random", "last_export_at": "2026-01-02T00:00:00Z"},
        }
        with open(state_file, "w") as f:
            json.dump(expected, f)

        state = load_state()
        assert len(state) == 2
        assert state["111"]["channel_name"] == "general"

    def test_load_invalid_json(self, temp_state_dir):
        """Test loading a corrupted state file."""
        state_file = os.path.join(temp_state_dir, "channel_state.json")
        with open(state_file, "w") as f:
            f.write("not valid json")

        state = load_state()
        assert state == {}


class TestSaveState:
    def test_save_state(self, temp_state_dir):
        """Test saving state to disk."""
        state = {
            "111": {"channel_id": "111", "channel_name": "general", "last_export_at": "2026-01-01T00:00:00Z"},
        }
        save_state(state)

        state_file = os.path.join(temp_state_dir, "channel_state.json")
        assert os.path.exists(state_file)
        with open(state_file) as f:
            loaded = json.load(f)
        assert loaded == state


class TestGetChannel:
    def test_get_existing_channel(self):
        """Test getting an existing channel record."""
        state = {"111": {"channel_id": "111", "channel_name": "general"}}
        result = get_channel(state, "111")
        assert result is not None
        assert result["channel_name"] == "general"

    def test_get_nonexistent_channel(self):
        """Test getting a nonexistent channel record."""
        state = {"111": {"channel_id": "111"}}
        result = get_channel(state, "999")
        assert result is None


class TestUpdateChannel:
    def test_update_new_channel(self):
        """Test adding a new channel to state."""
        state = {}
        record = update_channel(
            state,
            channel_id="111",
            channel_name="general",
            last_export_at="2026-01-01T00:00:00Z",
            total_messages_exported=100,
        )
        assert "111" in state
        assert record["channel_id"] == "111"
        assert record["total_messages_exported"] == 100

    def test_update_existing_channel(self):
        """Test updating an existing channel."""
        state = {"111": {"channel_id": "111", "channel_name": "old_name"}}
        update_channel(
            state,
            channel_id="111",
            channel_name="new_name",
            last_export_at="2026-02-01T00:00:00Z",
        )
        assert state["111"]["channel_name"] == "new_name"
        assert state["111"]["last_export_at"] == "2026-02-01T00:00:00Z"

    def test_update_channel_defaults_last_message_at(self):
        """Test that last_message_at defaults to last_export_at when not provided."""
        state = {}
        record = update_channel(
            state,
            channel_id="111",
            channel_name="test",
            last_export_at="2026-01-01T00:00:00Z",
        )
        assert record["last_message_at"] == "2026-01-01T00:00:00Z"


class TestChannelsNeedingExport:
    def test_new_channel_needs_export(self):
        """Test that a new channel (no prior state) needs export."""
        state = {}
        channels = [{"id": "111", "name": "general"}]

        result = channels_needing_export(state, channels)
        assert len(result) == 1
        assert result[0]["should_export"] is True
        assert result[0]["reason"] == "new channel (no prior export)"
        assert result[0]["last_export_at"] is None

    def test_channel_with_activity_needs_export(self):
        """Test that a channel with activity since last export needs export."""
        state = {
            "111": {
                "channel_id": "111",
                "channel_name": "general",
                "last_export_at": "2026-01-01T00:00:00Z",
            }
        }
        channels = [
            {"id": "111", "name": "general", "last_message_timestamp": "2026-02-01T00:00:00Z"}
        ]

        result = channels_needing_export(state, channels)
        assert len(result) == 1
        assert result[0]["should_export"] is True
        assert "activity since" in result[0]["reason"]

    def test_channel_without_activity_skipped(self):
        """Test that a channel with no new activity is skipped."""
        state = {
            "111": {
                "channel_id": "111",
                "channel_name": "general",
                "last_export_at": "2026-02-01T00:00:00Z",
            }
        }
        channels = [
            {"id": "111", "name": "general", "last_message_timestamp": "2026-01-15T00:00:00Z"}
        ]

        result = channels_needing_export(state, channels)
        assert len(result) == 0

    def test_multiple_channels_mixed(self):
        """Test evaluating multiple channels with mixed states."""
        state = {
            "111": {
                "channel_id": "111",
                "last_export_at": "2026-02-01T00:00:00Z",
            },
            "333": {
                "channel_id": "333",
                "last_export_at": "2026-03-01T00:00:00Z",
            }
        }
        channels = [
            {"id": "111", "name": "general", "last_message_timestamp": "2026-01-15T00:00:00Z"},  # No activity (old message)
            {"id": "222", "name": "random", "last_message_timestamp": "2026-03-01T00:00:00Z"},  # New channel (needs export)
            {"id": "333", "name": "offtopic", "last_message_timestamp": "2026-02-15T00:00:00Z"},  # No activity (message before export)
        ]

        result = channels_needing_export(state, channels)
        assert len(result) == 1  # Only channel 222 needs export (new channel)
        assert result[0]["channel_id"] == "222"
