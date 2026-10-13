"""DPR 图表提取适配：可选 PaperCropper，失败回退 PyMuPDF 内嵌图片。

调用参数、降级次序及图片尺寸过滤改编自 daily-paper-reader/src/paper_figures.py，
MIT 许可见 docs/licenses/daily-paper-reader.txt。宿主新增 PDF/hash/页授权绑定，
保留原始 PNG/图片字节，不引入 Pillow/WebP、图像理解模型或 Docling。
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import fitz

from research_loop.storage.files import FileStore


class PaperMediaProvider:
    """使用已下载同版本 PDF，只提取本次授权页；图片文件不是语义证据。"""

    def __init__(self, files: FileStore):
        """注入项目缓存根目录；不接受模型指定输出位置。"""
        self.files = files

    def extract(self, pdf_path: Path, pdf_hash: str, pages: list[int]) -> dict:
        """提取裁剪图表或内嵌图片，并返回真实位置、hash 和覆盖限制。

        Args:
            pdf_path: 已授权且下载完成的缓存 PDF。
            pdf_hash: 源文件字节 hash。
            pages: 允许处理的物理页，1 起算。

        Returns:
            assets、extractor、fallback_reason 及未验证能力；不将 page=0 当作物理页。
        """
        if sha256(pdf_path.read_bytes()).hexdigest() != pdf_hash:
            raise ValueError('PDF hash mismatch')
        script = os.getenv('PAPERCROPPER_SCRIPT', '')
        model = os.getenv('PAPERCROPPER_MODEL', '')
        enabled = os.getenv('PAPERCROPPER_DISABLE', '').lower() not in {'1', 'true', 'yes'}
        configured = enabled and Path(script).is_file() and Path(model).is_file()
        options = {'conf': '0.4', 'imgsz': '1024', 'dpi': '200', 'png-dpi': '260', 'batch-size': '4', 'padding': '2.0'}
        options = {k: os.getenv('PAPERCROPPER_' + k.upper().replace('-', '_'), v) for k, v in options.items()}
        runtime = {'script': sha256(Path(script).read_bytes()).hexdigest(), 'model': sha256(Path(model).read_bytes()).hexdigest()} if configured else None
        key = sha256(json.dumps([pdf_hash, pages, options, runtime, fitz.VersionBind, 'media-v1'], sort_keys=True).encode()).hexdigest()
        folder = self.files.cache_dir / 'media' / key
        manifest = folder / 'manifest.json'
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if all(Path(a['path']).is_file() and sha256(Path(a['path']).read_bytes()).hexdigest() == a['sha256'] for a in result['assets']):
                return {**result, 'cache_hit': True}
            raise ValueError('media cache integrity failure; preserve it for diagnosis')
        assets = []; fallback = 'PaperCropper not configured or disabled'
        with fitz.open(pdf_path) as document:
            if not pages or any(n < 1 or n > len(document) for n in pages):
                raise ValueError('invalid media physical page range')
            if configured:
                with tempfile.TemporaryDirectory(prefix='researchloop-media-') as temp:
                    # 外部提取器只能接触已授权页，不能读取用户禁止的其他页。
                    subset = Path(temp) / 'authorized.pdf'
                    with fitz.open() as selected:
                        for n in pages:
                            selected.insert_pdf(document, from_page=n-1, to_page=n-1)
                        selected.save(subset)
                    command = [os.getenv('PAPERCROPPER_PYTHON', sys.executable), script, '--pdf', str(subset), '--model', model,
                               '--output', temp, '--formats', 'png', '--targets', 'figure,table']
                    for k, v in options.items():
                        command.extend(['--' + k, v])
                    try:
                        result = subprocess.run(command, capture_output=True, text=True, check=False,
                            timeout=max(30, int(os.getenv('PAPERCROPPER_TIMEOUT_SECONDS', '360'))))
                        fallback = 'PaperCropper returned no assets' if result.returncode == 0 else f'PaperCropper exit {result.returncode}'
                        if result.returncode == 0:
                            for directory, kind in [('Figures_png', 'figure'), ('Tables_png', 'table')]:
                                for image in sorted((Path(temp) / 'authorized' / directory).glob('*.png')):
                                    payload = image.read_bytes()
                                    assets.append(self._asset(folder, payload, 'png', kind, None, None))
                    except (subprocess.TimeoutExpired, OSError) as exc:
                        fallback = 'PaperCropper ' + type(exc).__name__
            if not assets:
                seen = set()
                for n in pages:
                    page = document[n-1]
                    for spec in page.get_images(full=True):
                        xref = spec[0]
                        extracted = document.extract_image(xref)
                        width, height = extracted['width'], extracted['height']
                        if width < 240 or height < 180 or width * height < 120000:
                            continue
                        payload = extracted['image']; image_hash = sha256(payload).hexdigest()
                        if (n, image_hash) in seen:
                            continue
                        seen.add((n, image_hash))
                        rects = page.get_image_rects(xref)
                        assets.append(self._asset(folder, payload, extracted['ext'], 'embedded_image', n,
                                                  list(rects[0]) if len(rects) == 1 else None))
                extractor = 'pymupdf_embedded_images'
            else:
                extractor = 'papercropper_doclayout_yolo'; fallback = None
        result = {'pdf_sha256': pdf_hash, 'authorized_pages': pages, 'extractor': extractor, 'assets': assets,
                  'fallback_reason': fallback, 'cache_hit': False, 'model_received_images': False,
                  'limitations': ['只提取文件；未恢复表格单元格，也未进行图像语义核验。',
                      'PyMuPDF 回退不保证提取矢量表格/公式；PaperCropper未知页码和bbox为null。']}
        self.files.write_bytes_atomic(manifest, json.dumps(result, ensure_ascii=False).encode())
        return result

    def _asset(self, folder: Path, payload: bytes, extension: str, kind: str, page: int | None, bbox: list | None) -> dict:
        """按内容 hash 原子缓存图片；页码仅取可信 PDF API，不从文件排序推断。"""
        image_hash = sha256(payload).hexdigest()
        path = folder / (image_hash + '.' + extension)
        self.files.write_bytes_atomic(path, payload)
        return {'path': str(path), 'sha256': image_hash, 'kind': kind, 'page_number': page, 'bbox': bbox}
