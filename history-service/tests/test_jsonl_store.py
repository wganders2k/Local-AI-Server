"""
Unit tests for jsonl_store module.
"""
import json
import os
import tempfile
import pytest

from jsonl_store import (
    append_message,
    append_messages,
    get_message_count,
    get_all_user_ids,
    load_all_records,
    get_pending_captions,
    rewrite_archive,
    refresh_fields,
    update_records,
    _user_file_path,
    _user_id_cache,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the in-memory ID cache before each test."""
    _user_id_cache.clear()
    yield
    _user_id_cache.clear()


@pytest.fixture
def temp_archive_dir(monkeypatch):
    """Create a temporary archive directory and patch ARCHIVE_DIR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr("jsonl_store.ARCHIVE_DIR", tmpdir)
        yield tmpdir


class TestAppendMessage:
    def test_append_single_message(self, temp_archive_dir, monkeypatch):
        """Test appending a single message to a new user file."""
        record = {
            "message_id": "123456789",
            "user_id": "user1",
            "username": "test_user",
            "content": "Hello, world!",
            "timestamp": "2026-03-15T14:23:01Z",
        }

        result = append_message("user1", record)
        assert result is True

        # Verify file was created with correct content
        user_file = os.path.join(temp_archive_dir, "user1.jsonl")
        assert os.path.exists(user_file)
        with open(user_file) as f:
            line = f.readline().strip()
            saved_record = json.loads(line)
        assert saved_record["message_id"] == "123456789"
        assert saved_record["content"] == "Hello, world!"

    def test_append_duplicate_message(self, temp_archive_dir):
        """Test that duplicate messages are not appended."""
        record = {
            "message_id": "123456789",
            "user_id": "user1",
            "content": "Hello!",
        }

        result1 = append_message("user1", record)
        assert result1 is True

        result2 = append_message("user1", record)
        assert result2 is False  # Duplicate should be rejected

    def test_append_message_without_id(self, temp_archive_dir):
        """Test that messages without message_id are skipped."""
        record = {
            "user_id": "user1",
            "content": "No ID message",
        }

        result = append_message("user1", record)
        assert result is False


class TestAppendMessages:
    def test_append_multiple_messages(self, temp_archive_dir):
        """Test appending multiple messages at once."""
        records = [
            {"message_id": "1", "user_id": "user1", "content": "Msg 1"},
            {"message_id": "2", "user_id": "user1", "content": "Msg 2"},
            {"message_id": "3", "user_id": "user1", "content": "Msg 3"},
        ]

        count = append_messages("user1", records)
        assert count == 3

    def test_append_with_duplicates(self, temp_archive_dir):
        """Test appending when some messages already exist."""
        initial_records = [
            {"message_id": "1", "user_id": "user1", "content": "Msg 1"},
            {"message_id": "2", "user_id": "user1", "content": "Msg 2"},
        ]
        append_messages("user1", initial_records)

        new_records = [
            {"message_id": "2", "user_id": "user1", "content": "Msg 2 (duplicate)"},
            {"message_id": "3", "user_id": "user1", "content": "Msg 3"},
        ]

        count = append_messages("user1", new_records)
        assert count == 1  # Only one new message

    def test_append_empty_list(self, temp_archive_dir):
        """Test appending an empty list of messages."""
        count = append_messages("user1", [])
        assert count == 0


class TestGetMessageCount:
    def test_count_existing_user(self, temp_archive_dir):
        """Test counting messages for a user with existing archive."""
        records = [
            {"message_id": "1", "content": "Msg 1"},
            {"message_id": "2", "content": "Msg 2"},
            {"message_id": "3", "content": "Msg 3"},
        ]
        append_messages("user1", records)

        count = get_message_count("user1")
        assert count == 3

    def test_count_nonexistent_user(self, temp_archive_dir):
        """Test counting messages for a user with no archive."""
        count = get_message_count("nonexistent_user")
        assert count == 0


class TestGetAllUserIds:
    def test_list_user_ids(self, temp_archive_dir):
        """Test listing all user IDs with archives."""
        append_messages("user1", [{"message_id": "1", "content": "Msg"}])
        append_messages("user2", [{"message_id": "2", "content": "Msg"}])

        users = get_all_user_ids()
        assert set(users) == {"user1", "user2"}

    def test_empty_archive(self, temp_archive_dir):
        """Test listing user IDs when no archives exist."""
        users = get_all_user_ids()
        assert users == []


class TestLoadAllRecords:
    def test_load_records(self, temp_archive_dir):
        """Test loading all records for a user."""
        records = [
            {"message_id": "1", "content": "Msg 1", "timestamp": "2026-01-01T00:00:00Z"},
            {"message_id": "2", "content": "Msg 2", "timestamp": "2026-01-02T00:00:00Z"},
        ]
        append_messages("user1", records)

        loaded = load_all_records("user1")
        assert len(loaded) == 2
        assert loaded[0]["message_id"] == "1"
        assert loaded[1]["content"] == "Msg 2"

    def test_load_nonexistent_user(self, temp_archive_dir):
        """Test loading records for a user with no archive."""
        records = load_all_records("nonexistent")
        assert records == []


class TestGetPendingCaptions:
    def test_get_pending_captions(self, temp_archive_dir):
        """Test getting pending captions from attachments."""
        record = {
            "message_id": "1",
            "user_id": "user1",
            "attachments": [
                {
                    "url": "http://example.com/image.png",
                    "content_type": "image/png",
                    "filename": "test.png",
                    "file_size_bytes": 1024,
                    "caption_status": "pending",
                    "caption_excluded_from_training": True,
                }
            ],
        }
        append_messages("user1", [record])

        pending = get_pending_captions(limit=10)
        assert len(pending) == 1
        user_id, rec, att_idx, attachment = pending[0]
        assert user_id == "user1"
        assert att_idx == 0
        assert attachment["filename"] == "test.png"

    def test_skip_done_captions(self, temp_archive_dir):
        """Test that completed captions are not returned as pending."""
        record = {
            "message_id": "1",
            "user_id": "user1",
            "attachments": [
                {
                    "url": "http://example.com/image.png",
                    "caption_status": "done",
                    "caption": "Already captioned",
                }
            ],
        }
        append_messages("user1", [record])

        pending = get_pending_captions(limit=10)
        assert len(pending) == 0


class TestRewriteArchive:
    def test_rewrite_archive(self, temp_archive_dir):
        """Test rewriting an archive file."""
        initial_records = [
            {"message_id": "1", "content": "Original"},
            {"message_id": "2", "content": "Original 2"},
        ]
        append_messages("user1", initial_records)

        updated_records = [
            {"message_id": "1", "content": "Updated"},
            {"message_id": "2", "content": "Updated 2"},
            {"message_id": "3", "content": "New record"},
        ]
        rewrite_archive("user1", updated_records)

        loaded = load_all_records("user1")
        assert len(loaded) == 3
        assert loaded[0]["content"] == "Updated"
        assert loaded[2]["message_id"] == "3"


class TestUserFilePath:
    def test_path_generation(self, temp_archive_dir, monkeypatch):
        """Test that user file path is generated correctly."""
        path = _user_file_path("12345")
        assert path.endswith("12345.jsonl")
        assert temp_archive_dir in path


class TestRefreshFields:
    FIELDS = ("type", "reference", "is_pinned", "mentions")

    def _archived(self):
        return {
            "message_id": "1", "user_id": "u1", "content": "look",
            "attachments": [{"filename": "a.png", "caption": "a cat", "caption_status": "done"}],
            "is_pinned": True,
        }

    def test_parser_fields_refresh_and_captions_survive(self, temp_archive_dir):
        append_messages("u1", [self._archived()])
        parsed = {"message_id": "1", "user_id": "u1", "content": "look", "type": "Reply",
                  "reference": {"message_id": "9"}, "attachments": [{"filename": "a.png"}]}

        assert refresh_fields("u1", [parsed], self.FIELDS) == 1

        rec = load_all_records("u1")[0]
        assert rec["type"] == "Reply"
        assert rec["reference"] == {"message_id": "9"}
        assert "is_pinned" not in rec  # the parse no longer says it is pinned
        assert rec["attachments"][0]["caption"] == "a cat"

    def test_unchanged_records_are_not_rewritten(self, temp_archive_dir, monkeypatch):
        rec = dict(self._archived(), type="Default")
        append_messages("u1", [rec])
        monkeypatch.setattr("jsonl_store.rewrite_archive", lambda *a: pytest.fail("rewrote an unchanged file"))

        assert refresh_fields("u1", [dict(rec)], self.FIELDS) == 0

    def test_messages_not_archived_are_ignored(self, temp_archive_dir):
        append_messages("u1", [self._archived()])

        assert refresh_fields("u1", [{"message_id": "2", "type": "Default"}], self.FIELDS) == 0
        assert len(load_all_records("u1")) == 1

    def test_update_records_sees_writes_made_after_an_earlier_read(self, temp_archive_dir):
        append_messages("u1", [{"message_id": "1", "user_id": "u1"}])
        stale = load_all_records("u1")  # a long job's snapshot
        append_messages("u1", [{"message_id": "2", "user_id": "u1"}])

        def mark(records):
            records[0]["seen"] = True
            return True

        update_records("u1", mark)
        assert [r["message_id"] for r in load_all_records("u1")] == ["1", "2"]
        assert len(stale) == 1


class TestRewriteKeepsPermissions:
    def test_rewrite_keeps_the_existing_mode(self, temp_archive_dir):
        append_messages("u1", [{"message_id": "1"}])
        path = _user_file_path("u1")
        os.chmod(path, 0o644)

        rewrite_archive("u1", [{"message_id": "1", "type": "Default"}])

        assert os.stat(path).st_mode & 0o777 == 0o644

    def test_a_new_file_is_world_readable(self, temp_archive_dir):
        rewrite_archive("u2", [{"message_id": "1"}])

        assert os.stat(_user_file_path("u2")).st_mode & 0o777 == 0o644
