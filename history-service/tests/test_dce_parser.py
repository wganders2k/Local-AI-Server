"""
Unit tests for dce_parser module.
"""
import json
import os
import tempfile
import pytest

from dce_parser import (
    parse_dce_message,
    _parse_attachments,
    _is_image,
    parse_dce_export_file,
    parse_dce_export_directory,
)


class TestParseDCEMessage:
    def test_parse_basic_message(self):
        """Test parsing a basic DCE message."""
        raw = {
            "Id": "123456789012345678",
            "Author": {"Id": "987654321098765432", "Name": "testuser"},
            "Content": "Hello, world!",
            "Timestamp": "2026-03-15T14:23:01Z",
            "ChannelId": "111222333444555666",
            "ChannelName": "general",
            "Attachments": [],
        }

        record = parse_dce_message(raw)
        assert record is not None
        assert record["message_id"] == "123456789012345678"
        assert record["user_id"] == "987654321098765432"
        assert record["username"] == "testuser"
        assert record["content"] == "Hello, world!"
        assert record["channel_id"] == "111222333444555666"
        assert record["channel_name"] == "general"
        assert "attachments" not in record  # Empty attachments should be omitted

    def test_parse_message_without_id(self):
        """Test that messages without Id are skipped."""
        raw = {
            "Author": {"Id": "123", "Name": "user"},
            "Content": "No ID",
            "Timestamp": "2026-01-01T00:00:00Z",
        }
        record = parse_dce_message(raw)
        assert record is None

    def test_parse_message_with_null_content(self):
        """Test parsing a message with null content."""
        raw = {
            "Id": "123",
            "Author": {"Id": "456", "Name": "user"},
            "Content": None,
            "Timestamp": "2026-01-01T00:00:00Z",
        }
        record = parse_dce_message(raw)
        assert record is not None
        assert record["content"] == ""

    def test_parse_message_with_attachments(self):
        """Test parsing a message with image attachments."""
        raw = {
            "Id": "123",
            "Author": {"Id": "456", "Name": "user"},
            "Content": "Check this out",
            "Timestamp": "2026-01-01T00:00:00Z",
            "Attachments": [
                {
                    "Url": "http://example.com/image.png",
                    "ContentType": "image/png",
                    "Filename": "test.png",
                    "Size": 2048,
                }
            ],
        }

        record = parse_dce_message(raw)
        assert record is not None
        assert "attachments" in record
        assert len(record["attachments"]) == 1
        att = record["attachments"][0]
        assert att["url"] == "http://example.com/image.png"
        assert att["content_type"] == "image/png"
        assert att["file_size_bytes"] == 2048
        assert att["caption_status"] == "pending"
        assert att["caption_excluded_from_training"] is True


class TestParseAttachments:
    def test_parse_image_attachment(self):
        """Test parsing an image attachment."""
        raw = [{"Url": "http://img.png", "ContentType": "image/png", "Filename": "img.png", "Size": 1024}]
        result = _parse_attachments(raw)
        assert len(result) == 1
        assert result[0]["caption_status"] == "pending"

    def test_parse_non_image_attachment(self):
        """Test parsing a non-image attachment."""
        raw = [{"Url": "http://file.pdf", "ContentType": "application/pdf", "Filename": "doc.pdf", "Size": 4096}]
        result = _parse_attachments(raw)
        assert len(result) == 1
        assert result[0]["caption_status"] == "skipped"

    def test_parse_empty_attachments(self):
        """Test parsing an empty attachment list."""
        result = _parse_attachments([])
        assert result == []


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


class TestParseDCEExportFile:
    def test_parse_valid_file(self, tmp_path):
        """Test parsing a valid DCE export file."""
        messages = [
            {"Id": "1", "Author": {"Id": "u1", "Name": "user1"}, "Content": "Msg 1", "Timestamp": "2026-01-01T00:00:00Z"},
            {"Id": "2", "Author": {"Id": "u2", "Name": "user2"}, "Content": "Msg 2", "Timestamp": "2026-01-02T00:00:00Z"},
        ]
        filepath = tmp_path / "export.json"
        filepath.write_text(json.dumps(messages))

        records = parse_dce_export_file(str(filepath))
        assert len(records) == 2
        assert records[0]["message_id"] == "1"
        assert records[1]["user_id"] == "u2"

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


class TestParseDCEExportDirectory:
    def test_parse_directory_with_files(self, tmp_path):
        """Test parsing a directory with multiple export files."""
        # Create two channel export files
        channel1 = [
            {"Id": "1", "Author": {"Id": "u1", "Name": "user1"}, "Content": "Ch1 Msg1", "Timestamp": "2026-01-01T00:00:00Z"},
        ]
        channel2 = [
            {"Id": "2", "Author": {"Id": "u1", "Name": "user1"}, "Content": "Ch2 Msg1", "Timestamp": "2026-01-02T00:00:00Z"},
            {"Id": "3", "Author": {"Id": "u2", "Name": "user2"}, "Content": "Ch2 Msg2", "Timestamp": "2026-01-03T00:00:00Z"},
        ]

        (tmp_path / "channel1.json").write_text(json.dumps(channel1))
        (tmp_path / "channel2.json").write_text(json.dumps(channel2))

        result = parse_dce_export_directory(str(tmp_path))
        assert "u1" in result
        assert "u2" in result
        assert len(result["u1"]) == 2  # One from each channel
        assert len(result["u2"]) == 1

    def test_parse_empty_directory(self, tmp_path):
        """Test parsing an empty directory."""
        result = parse_dce_export_directory(str(tmp_path))
        assert result == {}

    def test_parse_nonexistent_directory(self):
        """Test parsing a nonexistent directory."""
        result = parse_dce_export_directory("/nonexistent/path")
        assert result == {}
