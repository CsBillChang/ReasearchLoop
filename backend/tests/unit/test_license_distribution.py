"""验证第三方改编模块随源码保留完整许可，不依赖未分发的根目录说明。"""
import ast
from hashlib import sha256
from pathlib import Path
import unittest


class LicenseDistributionTests(unittest.TestCase):
    """固定已迁入文件头的完整声明，避免公开精简误删许可内容。"""

    def test_adapted_modules_keep_complete_license_and_origin(self):
        """六个适配模块各自包含完整 MIT 原文和来源信息。"""
        folder = Path(__file__).resolve().parents[2] / 'research_loop' / 'providers'
        modules = ['paper_acquisition.py', 'paper_sources.py', 'paper_summary.py',
                   'discovery.py', 'pdf.py', 'paper_media.py']
        expected = 'f581afc6633041334cd746e21873f41626dac5fc7a83d7d62876fda462fb61c8'
        for name in modules:
            with self.subTest(module=name):
                doc = ast.get_docstring(ast.parse((folder / name).read_text()))
                self.assertIn('https://github.com/ziwenhahaha/daily-paper-reader', doc)
                self.assertIn('Copyright (c) 2026 ziwenhahaha', doc)
                license_text = 'MIT License\n' + doc.split('MIT License\n', 1)[1].rstrip() + '\n'
                self.assertEqual(sha256(license_text.encode()).hexdigest(), expected)
