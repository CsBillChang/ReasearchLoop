# 论文工具集成测试：检索授权、PDF 读取与页级证据应通过真实服务边界衔接。
import os
import tempfile
import unittest
from pathlib import Path

from research_loop.domain.tasks import SavePolicy
from research_loop.domain.evidence import PaperCandidate
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry


class PaperToolIntegrationTests(unittest.TestCase):
    def test_authorized_local_pdf_and_page_visibility(self):
        source = (Path(os.environ.get("RESEARCHLOOP_TEST_PDF_DIR", str(Path.home() / "Downloads"))) / "RARR.pdf")
        if not source.exists():
            self.skipTest("fixed PDF sample unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database("sqlite:///" + str(root / "test.db"))
            db.ensure_schema()
            files = FileStore(root / "data", root / "out", root)
            papers = PaperService(db, DiscoveryProvider(), PDFProvider(files))
            ctx = ToolContext(task_id="task", save_policy=SavePolicy.deny,
                              attachments={"rarr": {"local_path": str(source)}}, services={"papers": papers})
            result = ToolRegistry(specs()).execute("read_paper", {"target_ref": "rarr", "start_page": 1, "page_count": 2}, ctx)
            self.assertTrue(result.ok, result.message)
            self.assertEqual(result.data["returned_pages"], [1, 2])
            self.assertEqual(len(result.data["pages"]), 2)
            self.assertTrue(all(db.task_has_evidence("task", item["evidence_id"]) for item in result.data["pages"]))

    def test_arbitrary_local_path_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database("sqlite:///" + str(root / "test.db"))
            db.ensure_schema()
            papers = PaperService(db, DiscoveryProvider(), PDFProvider(FileStore(root / "data", root / "out", root)))
            ctx = ToolContext(task_id="task", save_policy=SavePolicy.deny, services={"papers": papers})
            result = ToolRegistry(specs()).execute("read_paper", {"target_ref": str(Path("/tmp/anything.pdf"))}, ctx)
            self.assertFalse(result.ok)
            self.assertEqual(result.kind, "execution_error")

    def test_fixed_paper_attachment_keeps_identity_and_requires_task_input(self):
        source = (Path(os.environ.get("RESEARCHLOOP_TEST_PDF_DIR", str(Path.home() / "Downloads"))) / "RARR.pdf")
        if not source.exists():
            self.skipTest("fixed PDF sample unavailable")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database("sqlite:///" + str(root / "test.db")); db.ensure_schema()
            files = FileStore(root / "data", root / "out", root)
            paper_id = "paper-fixed"
            db.save_paper(paper_id, PaperCandidate(paper_id=paper_id, title="Fixed", source="fixture",
                                                   pdf_url=str(source)).model_dump(mode="json"))
            db.grant_input("task", "r2-paper-a", "task")
            papers = PaperService(db, DiscoveryProvider(), PDFProvider(files))
            ctx = ToolContext(task_id="task", save_policy=SavePolicy.deny,
                              attachments={"r2-paper-a": {"local_path": str(source), "paper_id": paper_id}},
                              services={"papers": papers})
            result = ToolRegistry(specs()).execute("read_paper", {"target_ref": "r2-paper-a", "start_page": 1, "page_count": 1}, ctx)
            self.assertTrue(result.ok, result.message)
            self.assertEqual(db.fetchone("SELECT paper_id FROM documents WHERE document_id=?",
                                         (result.data["document_id"],))["paper_id"], paper_id)
