"""真实 PDF 组件配合提取器替身，验证降级、授权范围与定位真实性。"""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import fitz

from research_loop.providers.paper_media import PaperMediaProvider
from research_loop.storage.files import FileStore


class PaperMediaTests(unittest.TestCase):
    """图像提取不等于视觉理解或 DocLayout 模型已验收。"""

    def setUp(self):
        """生成两页含位图和矢量线条的可控 PDF。"""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = FileStore(self.root/'data', self.root/'notes', self.root)
        pix = fitz.Pixmap(fitz.csRGB, (0, 0, 500, 300)); pix.clear_with(150)
        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page(); page.insert_image(fitz.Rect(20, 20, 520, 320), stream=pix.tobytes('png'))
            page.draw_line((20, 400), (400, 400))
        self.pdf = self.root/'fixture.pdf'; doc.save(self.pdf); doc.close()
        self.hash = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        self.provider = PaperMediaProvider(self.files)

    def test_embedded_fallback_scope_hash_and_cache(self):
        """仅授权第二页，回退提取保留真实页和bbox，重复命中缓存。"""
        with patch.dict(os.environ, {'PAPERCROPPER_DISABLE':'1'}):
            result = self.provider.extract(self.pdf, self.hash, [2])
            self.assertEqual([a['page_number'] for a in result['assets']], [2])
            self.assertEqual(result['assets'][0]['bbox'], [20, 20, 520, 320])
            self.assertFalse(result['model_received_images'])
            self.assertTrue(self.provider.extract(self.pdf, self.hash, [2])['cache_hit'])
        with self.assertRaises(ValueError): self.provider.extract(self.pdf, '0'*64, [1])

    def test_cropper_unknown_locator_and_timeout_fallback(self):
        """裁剪器替身输出不含定位时不能从图片排序猜页；超时则回退。"""
        script = self.root/'crop.py'; script.write_text('# stub')
        model = self.root/'model.pt'; model.write_bytes(b'fixture')
        def crop(command, **kwargs):
            folder = Path(command[command.index('--output')+1]) / 'authorized' / 'Tables_png'
            folder.mkdir(parents=True)
            pix = fitz.Pixmap(fitz.csRGB, (0, 0, 500, 300)); pix.clear_with(80); pix.save(folder/'table1.png')
            return subprocess.CompletedProcess(command, 0, '', '')
        with patch.dict(os.environ, {'PAPERCROPPER_SCRIPT':str(script), 'PAPERCROPPER_MODEL':str(model), 'PAPERCROPPER_DISABLE':'0'}):
            with patch('subprocess.run', side_effect=crop):
                result = self.provider.extract(self.pdf, self.hash, [1])
            self.assertEqual(result['extractor'], 'papercropper_doclayout_yolo')
            self.assertIsNone(result['assets'][0]['page_number']); self.assertIsNone(result['assets'][0]['bbox'])
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('fixture', 30)):
                result = self.provider.extract(self.pdf, self.hash, [2])
            self.assertEqual(result['extractor'], 'pymupdf_embedded_images')
            self.assertIn('TimeoutExpired', result['fallback_reason'])
