# PDF 缓存测试：内容、解析器和配置身份决定复用，读取页范围单独验证。
import os
import json
import tempfile
import unittest
from pathlib import Path

from research_loop.providers.pdf import PDFProvider
from research_loop.storage.files import FileStore


class PaperCacheTests(unittest.TestCase):
    def test_pdf_hash_parser_config_cache_and_page_ranges(self):
        source = (Path(os.environ.get("RESEARCHLOOP_TEST_PDF_DIR", str(Path.home() / "Downloads"))) / "main.pdf")
        if not source.exists():
            self.skipTest("fixed PDF sample unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            provider = PDFProvider(FileStore(root / "data", root / "out", root))
            first = provider.ingest(str(source), "main")
            pages = provider.pages(first, 1, 2)
            second = provider.ingest(str(source), "main")
            self.assertEqual(first.content_sha256, second.content_sha256)
            self.assertEqual(first.cache_path, second.cache_path)
            self.assertEqual([1, 2], [page.page_number for page in pages])
            self.assertTrue(all(page.evidence_id for page in pages))
            self.assertTrue(Path(first.cache_path).exists())
            cached = json.loads(Path(first.cache_path).read_text())
            self.assertEqual(cached["total_pages"], first.total_pages)

