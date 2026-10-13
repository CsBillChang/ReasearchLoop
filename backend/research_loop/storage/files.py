"""文件适配器：管理 PDF/解析缓存和项目笔记；限制目录，同内容复用，异内容拒绝覆盖。"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


class FileConflictError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FileStore:
    def __init__(self, data_dir: str | Path, output_dir: str | Path, project_root: str | Path):
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.cache_dir = self.data_dir / "cache"
        self.pdf_dir = self.cache_dir / "pdf"
        self.parse_dir = self.cache_dir / "parsed"
        for path in (self.cache_dir, self.pdf_dir, self.parse_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _inside(self, path: Path, root: Path) -> Path:
        """解析真实路径后检查根目录边界，阻止路径穿越到配置目录之外。"""
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise PermissionError(f"path is outside configured root: {resolved}") from exc
        return resolved

    def cache_pdf_path(self, content_hash: str) -> Path:
        return self._inside(self.pdf_dir / f"{content_hash}.pdf", self.pdf_dir)

    def cache_parse_path(self, cache_key: str) -> Path:
        return self._inside(self.parse_dir / f"{cache_key}.json", self.parse_dir)

    def write_bytes_atomic(self, path: Path, content: bytes) -> str:
        """返回内容 hash；同内容复用，异内容抛冲突；新文件先写临时文件并 fsync，再原子替换目标。"""
        path = path.resolve()
        try:
            path.relative_to(self.data_dir)
        except ValueError:
            path = self._inside(path, self.output_dir)
        digest = sha256_bytes(content)
        if path.exists():
            existing = path.read_bytes()
            if sha256_bytes(existing) == digest:
                return digest
            raise FileConflictError(f"refusing to overwrite different content: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return digest

    def save_markdown(self, task_id: str, draft_id: str, content: str, content_hash: str) -> tuple[Path, str]:
        """按 task_id/draft_id 构造项目内路径并校验内容 hash；不接受模型指定任意输出路径。"""
        safe_task = "".join(ch for ch in task_id if ch.isalnum() or ch in "-_")
        safe_draft = "".join(ch for ch in draft_id if ch.isalnum() or ch in "-_")
        path = self._inside(self.output_dir / safe_task / f"{safe_draft}.md", self.output_dir)
        digest = sha256_bytes(content.encode("utf-8"))
        if digest != content_hash:
            raise ValueError("content_hash does not match markdown")
        self.write_bytes_atomic(path, content.encode("utf-8"))
        return path, digest
