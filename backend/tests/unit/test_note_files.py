"""笔记文件测试：权限、版本身份、重复写入和异内容冲突分别验证。"""
import tempfile
import unittest
from pathlib import Path

from research_loop.domain.notes import CreateNoteInput
from research_loop.domain.tasks import SavePolicy
from research_loop.services.notes import NoteService
from research_loop.storage.files import FileConflictError, FileStore
from research_loop.storage.mysql import Database


class NoteFileTests(unittest.TestCase):
    """在临时目录验证文件事实与当前数据库草稿的独立性。"""

    def test_imported_draft_projection_does_not_choose_last_saved_version(self):
        """没有创建工具动作时，旧保存不能冒充新稿；新稿保存后正确显示且不跨任务。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database("sqlite:///" + str(root / "test.db")); db.ensure_schema()
            service = NoteService(db, FileStore(root / "data", root / "out", root))
            old = service.create("task", CreateNoteInput(title="旧稿", markdown="old"))
            service.save("task", SavePolicy.allow, old.draft_id, old.content_hash)
            current = service.create("task", CreateNoteInput(title="新稿", markdown="new"))
            self.assertFalse(service.artifacts("task")[0]["is_current_draft"])
            saved = service.save("task", SavePolicy.allow, current.draft_id, current.content_hash)
            foreign = service.create("other", CreateNoteInput(title="其他任务", markdown="foreign"))
            service.save("other", SavePolicy.allow, foreign.draft_id, foreign.content_hash)
            rows = {row["draft_id"]: row for row in service.artifacts("task")}
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[current.draft_id]["is_current_draft"])
            self.assertFalse(rows[old.draft_id]["is_current_draft"])
            Path(saved.path).write_text("用户改文")
            changed = next(row for row in service.artifacts("task") if row["draft_id"] == current.draft_id)
            self.assertEqual(changed["integrity"], "missing_or_changed")

    def test_same_content_is_idempotent_and_different_content_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = FileStore(root / "data", root / "out", root)
            db = Database("sqlite:///" + str(root / "test.db"))
            db.ensure_schema()
            service = NoteService(db, store)
            draft = service.create("task", CreateNoteInput(title="T", markdown="# Draft"))
            first = service.save("task", SavePolicy.allow, draft.draft_id, draft.content_hash)
            second = service.save("task", SavePolicy.allow, draft.draft_id, draft.content_hash)
            self.assertEqual(first.status, "saved")
            self.assertEqual(second.status, "already_saved")
            with self.assertRaises(FileConflictError):
                store.write_bytes_atomic(Path(first.path), b"different")

    def test_unknown_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database("sqlite:///" + str(root / "test.db"))
            db.ensure_schema()
            service = NoteService(db, FileStore(root / "data", root / "out", root))
            with self.assertRaises(ValueError):
                service.create("task", CreateNoteInput(title="T", markdown="x", evidence_ids=["not-visible"]))
