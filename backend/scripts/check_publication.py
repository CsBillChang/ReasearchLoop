"""检查Git暂存内容的公开边界；不联网、不调用模型、不输出敏感值。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT_FILES = {".gitignore", ".env.example", ".python-version",
              "README.md", "Makefile", "compose.yaml"}
SOURCE_ROOTS = {"backend", "frontend", "config"}
BLOCKED_PARTS = {"node_modules", "__pycache__", ".venv", ".vscode", ".idea",
                 "archive", "backups", "data-restores", ".pytest_cache"}
BLOCKED_SUFFIXES = {".pdf", ".onnx", ".sqlite", ".db", ".log", ".pem", ".key",
                    ".pyc", ".png", ".jpg", ".jpeg", ".zip", ".gz"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider_key": re.compile(r"\bsk-[A-Za-z0-9_-]{18,}"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    "personal_home": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
}
# 仅放行逐图检查过的合成展示截图；换图须重新核对像素内容和字节身份。
# hash 白名单不等于自动隐私识别，其他图片仍按原发布边界拒绝。
REVIEWED_SCREENSHOTS = {
    "frontend/assets/screenshots/chat.jpg": "538c4266b661a555558352ff04fbee1058a1f3da9f5a7ce2ba5474d2834ed1a2",
    "frontend/assets/screenshots/profile.jpg": "4a59bb6d3611a87e4b6cefe2559cdbea1088f25c43aa31ccee346bb2abf81e90",
}


def check_file(name: str, content: bytes, mode: str = "100644") -> list[dict]:
    """检查单个暂存文件，只返回路径、规则和行号。

    Args:
        name: 仓库内的POSIX相对路径。
        content: 暂存blob原始字节，不使用可能不同的工作区内容。
        mode: Git文件模式；拒绝符号链接与子模块。

    Returns:
        命中的公开边界问题；空列表表示本检查器未发现问题。
    """
    path = PurePosixPath(name)
    findings = []
    allowed = name in ROOT_FILES
    allowed = allowed or (bool(path.parts) and path.parts[0] in SOURCE_ROOTS)
    if (not allowed or path.is_absolute() or ".." in path.parts
            or any(part in BLOCKED_PARTS for part in path.parts)
            or (path.name.startswith(".env") and name != ".env.example")
            or (path.suffix.lower() in BLOCKED_SUFFIXES and name not in REVIEWED_SCREENSHOTS)):
        findings.append({"path": name, "rule": "excluded_path"})
    if mode not in {"100644", "100755"}:
        findings.append({"path": name, "rule": "unsupported_file_mode"})
    if name in REVIEWED_SCREENSHOTS:
        if mode != "100644":
            findings.append({"path": name, "rule": "image_mode_not_regular"})
        if (len(content) > 1_000_000 or not content.startswith(b"\xff\xd8\xff")
                or hashlib.sha256(content).hexdigest() != REVIEWED_SCREENSHOTS[name]):
            findings.append({"path": name, "rule": "unreviewed_image_bytes"})
        return findings
    if len(content) > 1_000_000 or b"\x00" in content:
        findings.append({"path": name, "rule": "large_or_binary"})
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return findings + [{"path": name, "rule": "non_utf8"}]
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"path": name, "rule": rule,
                             "line": text.count("\n", 0, match.start()) + 1})
    return findings


def inspect_index(root: Path) -> dict:
    """检查整个暂存树，拒绝冲突阶段，返回不含原文的汇总。"""
    def git(*args: str) -> bytes:
        """从指定仓库读取Git索引，不修改引用或工作区。"""
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True).stdout

    entries = git("ls-files", "--stage", "-z").split(b"\x00")
    findings, count, size = [], 0, 0
    for entry in entries:
        if not entry:
            continue
        meta, raw_name = entry.split(b"\t", 1)
        mode, sha, stage = meta.decode().split()
        name = raw_name.decode("utf-8")
        if stage != "0":
            findings.append({"path": name, "rule": "unmerged_index"})
            continue
        blob = git("cat-file", "blob", sha)
        count += 1
        size += len(blob)
        findings.extend(check_file(name, blob, mode))
    if not count:
        findings.append({"path": ".", "rule": "empty_index"})
    return {"files": count, "bytes": size, "passed": not findings,
            "findings": findings, "scope": "staged_blobs_only"}


def main() -> None:
    """运行发布检查；可写本机报告，失败返回非零，不输出敏感原文。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = inspect_index(Path(__file__).resolve().parents[2])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
