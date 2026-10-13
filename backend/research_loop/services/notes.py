"""笔记服务：保留原始 Markdown，校验身份/版本并按授权保存；审校只作为独立记录。"""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

from research_loop.domain.notes import CreateNoteInput, Draft, ReviewResult, SavedNote
from research_loop.domain.tasks import SavePolicy
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database


class NoteService:
    def __init__(self, db: Database, files: FileStore):
        self.db = db
        self.files = files

    def create(self, task_id: str, payload: CreateNoteInput) -> Draft:
        """校验内部证据权限并原样落库 Markdown；不追加 Sources、hash 或核验提示。"""
        unknown = [evidence_id for evidence_id in payload.evidence_ids
                   if not self.db.task_has_evidence(task_id, evidence_id)]
        if unknown:
            raise ValueError("evidence not returned to this task: " + str(unknown))
        markdown = payload.markdown
        content_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        draft = Draft(draft_id=str(uuid4()), task_id=task_id, title=payload.title,
                      markdown=markdown, content_hash=content_hash, evidence_ids=payload.evidence_ids)
        self.db.save_draft(draft.model_dump(mode="json"))
        self.db.update_task(task_id, save_status="draft_created", quality_status="unreviewed")
        return draft

    def save(self, task_id: str, save_policy: SavePolicy, draft_id: str, content_hash: str) -> SavedNote:
        """按授权保存指定版本的笔记，并校验已有文件是否被修改。

        Args:
            task_id: 拥有草稿的根任务 ID。
            save_policy: 后端保存权限，不由模型自行提升。
            draft_id: 待保存的草稿 ID。
            content_hash: 调用方确认的当前稿内容 hash。

        Returns:
            包含实际路径、导出文件 hash 和核验状态的保存收据。

        Raises:
            PermissionError: 任务不允许写文件。
            ValueError: 草稿不属于任务、版本过期或已有文件已被修改。
        """
        if save_policy != SavePolicy.allow:
            raise PermissionError("save_note denied by task save_policy")
        draft = self.db.get_draft(draft_id)
        if not draft or draft["task_id"] != task_id:
            raise ValueError("draft is missing or belongs to another task")
        if draft["content_hash"] != content_hash:
            raise ValueError("content_hash does not match current draft")
        review = self.db.get_review_result(task_id, draft_id, content_hash)
        has_collaboration = bool(self.db.list_subtasks(task_id))
        review_status = review.get("status") if review else ("not_run" if has_collaboration else None)
        prior = self.db.get_receipt(draft_id, content_hash)
        if prior:
            export = self.db.fetchone("SELECT * FROM note_exports WHERE draft_id=? AND content_hash=?", (draft_id, content_hash))
            from pathlib import Path
            actual = hashlib.sha256(Path(prior["path"]).read_bytes()).hexdigest()
            expected = export["file_sha256"] if export else content_hash
            if actual != expected:
                raise ValueError("saved file changed; refusing to claim an unchanged save")
            return SavedNote(draft_id=draft_id, path=prior["path"], content_hash=content_hash,
                             file_sha256=actual, receipt_id=prior["receipt_id"], status="already_saved",
                             review_status=review_status)
        for child in self.db.list_subtasks(task_id):
            raw = child.get("result") or {}
            if isinstance(raw, str):
                raw = json.loads(raw)
            child_result = raw.get("result", raw)
            if child_result.get("deep_reading_status") == "partial":
                # 对齐 DPR 严格阅读发布入口；不是追加 Reviewer 或逐项事实门。
                raise ValueError("论文精读总结未完整生成")
        exported = draft["markdown"]
        file_hash = hashlib.sha256(exported.encode("utf-8")).hexdigest()
        path, digest = self.files.save_markdown(task_id, draft_id, exported, file_hash)
        receipt_id = str(uuid4())
        receipt_status = "saved"
        self.db.execute("INSERT OR REPLACE INTO note_exports(draft_id,content_hash,file_sha256,review_status) VALUES(?,?,?,?)",
                        (draft_id, content_hash, digest, review_status))
        self.db.save_receipt(receipt_id, draft_id, content_hash, str(path), receipt_status)
        self.db.update_task(task_id, save_status=receipt_status)
        return SavedNote(draft_id=draft_id, path=str(path), content_hash=content_hash,
                         file_sha256=digest, receipt_id=receipt_id, status=receipt_status, review_status=review_status)

    def save_review(self, result: ReviewResult) -> ReviewResult:
        """审校必须绑定当前任务的准确草稿 hash；过期结论拒绝落到另一版本上。"""
        draft = self.db.get_draft(result.draft_id)
        if not draft or draft["task_id"] != result.task_id or draft["content_hash"] != result.content_hash:
            raise ValueError("review is bound to an expired or different draft hash")
        self.db.save_review_result(result.model_dump(mode="json"))
        self.db.update_task(result.task_id, quality_status=result.status)
        return result

    def artifacts(self, task_id: str) -> list[dict]:
        """返回本任务实际保存的版本；答复正文不等于附件，旧稿也不冒充当前稿。

        Returns:
            带文件 hash、实际完整性和独立核验状态的附件，不修改文件或收据。
        """
        from pathlib import Path
        rows = self.db.fetchall("SELECT d.draft_id,d.content_hash,d.title,s.path,e.file_sha256 FROM drafts d JOIN save_receipts s ON s.draft_id=d.draft_id AND s.content_hash=d.content_hash LEFT JOIN note_exports e ON e.draft_id=d.draft_id AND e.content_hash=d.content_hash WHERE d.task_id=?", (task_id,))
        actions = self.db.list_actions(task_id)
        current = next((a["observation"]["data"].get("draft_id") for a in reversed(actions)
                        if a["status"] == "succeeded" and a["tool_name"] in {"create_note", "revise_note"}), None)
        if current is None:
            # 已导入/恢复的草稿可能没有本任务的创建动作；与 Worker 的交付回退一致，
            # 从本任务的最新数据库版本取身份，不能把“最近保存的旧稿”当成当前稿。
            latest = self.db.fetchone("SELECT draft_id FROM drafts WHERE task_id=? ORDER BY created_at DESC,draft_id DESC LIMIT 1", (task_id,))
            current = latest["draft_id"] if latest else None
        output = []
        for row in rows:
            path = Path(row["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            review = self.db.get_review_result(task_id, row["draft_id"], row["content_hash"])
            output.append({**row, "integrity": "verified" if actual == (row["file_sha256"] or row["content_hash"]) else "missing_or_changed",
                           "is_current_draft": row["draft_id"] == current,
                           "review_status": review["status"] if review else "not_run"})
        return output
