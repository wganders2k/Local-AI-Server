"""
Unit tests for dce_parser module.

Tests use the actual DCE output format (lowercase fields, top-level channel
object, "messages" array) to ensure the parser handles real data correctly.
"""
import json
import os
import tempfile
import pytest

from dce_parser import (
    parse_dce_message,
    _parse_attachments,
    _is_image,
    _is_system_message,
    _extract_channel_info,
    parse_dce_export_file,
    parse_dce_export_directory,
)


# ──────────────────────────────────────────────────────────────
# Helper: build a realistic DCE export object
# ──────────────────────────────────────────────────────────────

def _build_dce_export(messages, channel_id="111222333", channel_name="general"):
    """Build a full DCE-style export JSON object."""
    return {
        "guild": {"id": "419718473233465355", "name": "Test Server"},
        "channel": {"id": channel_id, "name": channel_name},
        "messages": messages,
    }


def _build_message(
    msg_id="1234567890",
    author_id="9876543210",
    username="testuser",
    content="Hello!",
    timestamp="2026-03-15T14:23:01+00:00",
    msg_type="Default",
    is_bot=False,
    nickname=None,
    attachments=None,
):
    """Build a single DCE message dict (lowercase fields)."""
    msg = {
        "id": msg_id,
        "type": msg_type,
        "timestamp": timestamp,
        "content": content,
        "author": {
            "id": author_id,
            "name": username,
            "isBot": is_bot,
        },
        "attachments": attachments or [],
    }
    if nickname:
        msg["author"]["nickname"] = nickname
    return msg


# ──────────────────────────────────────────────────────────────
# parse_dce_message
# ──────────────────────────────────────────────────────────────

class TestParseDCEMessage:
    def test_parse_basic_message(self):
        """Test parsing a basic DCE message with lowercase fields."""
        raw = _build_message()
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None
        assert record["message_id"] == "1234567890"
        assert record["user_id"] == "9876543210"
        assert record["username"] == "testuser"
        assert record["content"] == "Hello!"
        assert record["channel_id"] == "111"
        assert record["channel_name"] == "general"
        assert record["is_bot"] is False
        assert "attachments" not in record

    def test_parse_message_with_nickname(self):
        """Test that nickname (display name) is preferred over plain name."""
        raw = _build_message(username="pcow", nickname="Walter Hartwell White")
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None
        assert record["username"] == "Walter Hartwell White"

    def test_parse_message_without_id(self):
        """Test that messages without id are skipped."""
        raw = {"author": {"id": "123", "name": "user"}, "content": "No ID"}
        record = parse_dce_message(raw)
        assert record is None

    def test_parse_message_with_null_content(self):
        """Test parsing a message with null content."""
        raw = _build_message(content=None)
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None
        assert record["content"] == ""

    def test_parse_message_with_attachments(self):
        """Test parsing a message with image attachments."""
        raw = _build_message(
            content="Check this out",
            attachments=[
                {
                    "url": "http://example.com/image.png",
                    "content_type": "image/png",
                    "filename": "test.png",
                    "size": 2048,
                }
            ],
        )
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None
        assert "attachments" in record
        assert len(record["attachments"]) == 1
        att = record["attachments"][0]
        assert att["url"] == "http://example.com/image.png"
        assert att["content_type"] == "image/png"
        assert att["file_size_bytes"] == 2048
        assert att["caption_status"] == "pending"
        assert att["caption_excluded_from_training"] is True

    def test_bot_message_skipped(self):
        """Test that bot messages are skipped."""
        raw = _build_message(is_bot=True)
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is None

    def test_system_message_type_skipped(self):
        """Test that system message types (ThreadCreated, type 20) are skipped."""
        raw = _build_message(msg_type="ThreadCreated")
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is None

    def test_interaction_confirmation_skipped(self):
        """Test that interaction confirmation (type 20) is skipped."""
        raw = _build_message(msg_type="20")
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is None

    def test_default_message_type_kept(self):
        """Test that Default type messages are kept."""
        raw = _build_message(msg_type="Default")
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None

    def test_type_zero_kept(self):
        """Test that type 0 (normal text) messages are kept."""
        raw = _build_message(msg_type=0)
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None

    def test_type_zero_string_kept(self):
        """Test that type '0' (string) messages are kept."""
        raw = _build_message(msg_type="0")
        record = parse_dce_message(raw, channel_id="111", channel_name="general")
        assert record is not None


# ──────────────────────────────────────────────────────────────
# _is_system_message
# ──────────────────────────────────────────────────────────────

class TestIsSystemMessage:
    def test_default_not_skipped(self):
        assert _is_system_message("Default") is False
        assert _is_system_message(0) is False
        assert _is_system_message("0") is False
        assert _is_system_message(None) is False

    def test_system_types_skipped(self):
        for t in ["1", "2", "3", "5", "6", "7", "8", "9", "10", "11", "12",
                   "13", "14", "15", "16", "17", "19", "20", "21", "22"]:
            assert _is_system_message(t) is True, f"Type {t} should be skipped"

    def test_thread_created_skipped(self):
        assert _is_system_message("ThreadCreated") is True


# ──────────────────────────────────────────────────────────────
# _extract_channel_info
# ──────────────────────────────────────────────────────────────

class TestExtractChannelInfo:
    def test_extract_from_dce_object(self):
        data = {"channel": {"id": "123", "name": "general"}}
        channel_id, channel_name = _extract_channel_info(data)
        assert channel_id == "123"
        assert channel_name == "general"

    def test_empty_channel(self):
        data = {}
        channel_id, channel_name = _extract_channel_info(data)
        assert channel_id == ""
        assert channel_name == ""


# ──────────────────────────────────────────────────────────────
# _parse_attachments
# ──────────────────────────────────────────────────────────────

class TestParseAttachments:
    def test_parse_image_attachment(self):
        """Test parsing an image attachment with lowercase fields."""
        raw = [{"url": "http://img.png", "content_type": "image/png", "filename": "img.png", "size": 1024}]
        result = _parse_attachments(raw)
        assert len(result) == 1
        assert result[0]["caption_status"] == "pending"

    def test_parse_non_image_attachment(self):
        """Test parsing a non-image attachment."""
        raw = [{"url": "http://file.pdf", "content_type": "application/pdf", "filename": "doc.pdf", "size": 4096}]
        result = _parse_attachments(raw)
        assert len(result) == 1
        assert result[0]["caption_status"] == "skipped"

    def test_parse_empty_attachments(self):
        """Test parsing an empty attachment list."""
        result = _parse_attachments([])
        assert result == []


# ──────────────────────────────────────────────────────────────
# _is_image
# ──────────────────────────────────────────────────────────────

class TestIsImage:
    def test_image_types(self):
        """Test image content type detection."""
        assert _is_image("image/png") is True
        assert _is_image("image/jpeg") is True
        assert _is_image("image/gif") is True
        assert _is_image("image/webp") is True

    def test_non_image_types(self):
        """Test non-image content type detection."""
        assert _is_image("application/pdf") is False
        assert _is_image("text/plain") is False
        assert _is_image("") is False
        assert _is_image(None) is False


# ──────────────────────────────────────────────────────────────
# parse_dce_export_file
# ──────────────────────────────────────────────────────────────

class TestParseDCEExportFile:
    def test_parse_valid_file_with_messages_key(self, tmp_path):
        """Test parsing a DCE export with top-level object and 'messages' key."""
        messages = [
            _build_message(msg_id="1", author_id="u1", username="user1", content="Msg 1"),
            _build_message(msg_id="2", author_id="u2", username="user2", content="Msg 2"),
        ]
        data = _build_dce_export(messages, channel_id="ch1", channel_name="general")
        filepath = tmp_path / "export.json"
        filepath.write_text(json.dumps(data))

        records = parse_dce_export_file(str(filepath))
        assert len(records) == 2
        assert records[0]["message_id"] == "1"
        assert records[1]["user_id"] == "u2"
        assert records[0]["channel_id"] == "ch1"
        assert records[0]["channel_name"] == "general"

    def test_parse_file_filters_bot_messages(self, tmp_path):
        """Test that bot messages are filtered from export files."""
        messages = [
            _build_message(msg_id="1", author_id="u1", username="user1", content="Human msg"),
            _build_message(msg_id="2", author_id="bot1", username="Bot", content="Bot msg", is_bot=True),
        ]
        data = _build_dce_export(messages)
        filepath = tmp_path / "export.json"
        filepath.write_text(json.dumps(data))

        records = parse_dce_export_file(str(filepath))
        assert len(records) == 1
        assert records[0]["message_id"] == "1"

    def test_parse_file_filters_system_messages(self, tmp_path):
        """Test that system message types are filtered from export files."""
        messages = [
            _build_message(msg_id="1", author_id="u1", username="user1", content="Normal msg"),
            _build_message(msg_id="2", author_id="u1", username="user1", content="Started a thread.", msg_type="ThreadCreated"),
            _build_message(msg_id="3", author_id="bot", username="Bot", content="✅ Chat thread created", msg_type="20"),
        ]
        data = _build_dce_export(messages)
        filepath = tmp_path / "export.json"
        filepath.write_text(json.dumps(data))

        records = parse_dce_export_file(str(filepath))
        assert len(records) == 1
        assert records[0]["message_id"] == "1"

    def test_parse_invalid_json(self, tmp_path):
        """Test parsing a file with invalid JSON."""
        filepath = tmp_path / "bad.json"
        filepath.write_text("not valid json")

        records = parse_dce_export_file(str(filepath))
        assert records == []

    def test_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        filepath = tmp_path / "empty.json"
        filepath.write_text("")

        records = parse_dce_export_file(str(filepath))
        assert records == []


# ──────────────────────────────────────────────────────────────
# parse_dce_export_directory
# ──────────────────────────────────────────────────────────────

class TestParseDCEExportDirectory:
    def test_parse_directory_with_files(self, tmp_path):
        """Test parsing a directory with multiple DCE export files."""
        channel1 = _build_dce_export([
            _build_message(msg_id="1", author_id="u1", username="user1", content="Ch1 Msg1"),
        ], channel_id="ch1", channel_name="general")

        channel2 = _build_dce_export([
            _build_message(msg_id="2", author_id="u1", username="user1", content="Ch2 Msg1"),
            _build_message(msg_id="3", author_id="u2", username="user2", content="Ch2 Msg2"),
        ], channel_id="ch2", channel_name="random")

        (tmp_path / "channel1.json").write_text(json.dumps(channel1))
        (tmp_path / "channel2.json").write_text(json.dumps(channel2))

        result = parse_dce_export_directory(str(tmp_path))
        assert "u1" in result
        assert "u2" in result
        assert len(result["u1"]) == 2
        assert len(result["u2"]) == 1

    def test_parse_empty_directory(self, tmp_path):
        """Test parsing an empty directory."""
        result = parse_dce_export_directory(str(tmp_path))
        assert result == {}

    def test_parse_nonexistent_directory(self):
        """Test parsing a nonexistent directory."""
        result = parse_dce_export_directory("/nonexistent/path")
        assert result == {}
