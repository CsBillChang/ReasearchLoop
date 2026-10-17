"""上传服务：校验 PDF 内容，生成内容身份，并解析受控上传引用。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pymupdf

from research_loop.config import Settings
from research_loop.storage.files import FileStore


def resolve_upload(settings, attachment):
    """将服务器生成的上传身份绑定到已验证PDF，拒绝调用方替换路径或链接。"""
    if not attachment.input_ref.startswith('upload-'):
        return attachment
    digest=attachment.input_ref[7:]
    if not re.fullmatch('[a-f0-9]{64}',digest):
        raise ValueError('无效上传身份')
    path=settings.data_dir/'uploads'/f'{digest}.pdf'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('上传文件不存在或已改变，请重新上传')
    if attachment.url or (attachment.local_path and Path(attachment.local_path).resolve()!=path.resolve()):
        raise ValueError('上传身份不能替换为其他路径')
    return attachment.model_copy(update={'local_path':str(path)})


def store_pdf(settings: Settings, data: bytes, filename: str) -> dict:
    """验证已接收的 PDF 并以内容 hash 原子保存。

    Args:
        settings: 后端配置的项目及数据根目录。
        data: HTTP 层已完成大小检查的原始字节。
        filename: 仅作为展示名称，不参与服务器路径拼接。

    Returns:
        上传身份、文件名、页数和字节数，不返回服务器文件路径。

    Raises:
        ValueError: 字节不是 PDF，或文档损坏、为空、需要密码。
    """
    if not data.startswith(b'%PDF-'):
        raise ValueError('不是PDF文件')
    try:
        with pymupdf.open(stream=data, filetype='pdf') as pdf:
            if pdf.needs_pass or not pdf.page_count:
                raise ValueError('PDF为空或需要密码')
            pages = pdf.page_count
    except Exception as exc:
        raise ValueError('无法打开PDF，可能损坏或已加密') from exc
    digest = hashlib.sha256(data).hexdigest()
    files = FileStore(settings.data_dir, settings.output_dir, settings.project_root)
    files.write_bytes_atomic(settings.data_dir / 'uploads' / f'{digest}.pdf', data)
    return {
        'input_ref': 'upload-' + digest,
        'filename': filename.replace('\\', '/').split('/')[-1],
        'sha256': digest, 'pages': pages, 'bytes': len(data),
    }
