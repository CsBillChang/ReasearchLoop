"""隔离SQLite验证取消贯通真实取文链与父线程池；角色生成边界为合成替身。"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from research_loop.config import Settings
from research_loop.domain.evidence import Document, PageEvidence, PaperCandidate
from research_loop.domain.tasks import SavePolicy, TaskStatus
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from research_loop.workers.executor import TaskExecutor


class ControlledStream(httpx.SyncByteStream):
    """在已有首块后等待测试显式释放，模拟尚未返回的socket。"""

    def __init__(self, entered, release, before_second=lambda: None):
        """取消信号和流返回分别受控，不能把状态修改冒充线程退出。"""
        self.entered, self.release, self.before_second = entered, release, before_second
        self.closed, self.overread = False, False

    def __iter__(self):
        """只有测试释放后才返回下一块，使父子等待可观察。"""
        yield b"%PDF-first-block"
        self.entered.set()
        if not self.release.wait(5):
            raise AssertionError("controlled stream was never released")
        self.before_second()
        yield b"cancel-observable-block"
        self.overread = True
        yield b"must-not-read"

    def close(self):
        """记录流真正关闭，区别于仅数据库终态。"""
        self.closed = True


class LegacyPDF:
    """无取消作用域且只接受原有两位置参数的本地替身。"""

    def ingest(self, source, paper_id):
        """固定合成文档，验证兼容旧适配签名。"""
        return Document(document_id="legacy-doc", paper_id=paper_id, content_sha256="a" * 64,
                        parser_name="fixture", parser_version="1", config_sha256="b" * 64,
                        total_pages=1, cache_path=source)

    def pages(self, document, start, count):
        """返回一页合成证据，不读取真实文件。"""
        return [PageEvidence(evidence_id="legacy-page", document_id=document.document_id,
                             page_number=1, text="synthetic evidence")]


class ReadCancellationTests(unittest.TestCase):
    """实际工具、服务、获取与隔离存储，不发送模型或外部HTTP。"""

    def setUp(self):
        """仅使用临时SQLite与本地MockTransport。"""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = Settings(project_root=self.root, data_dir=self.root / "data",
                                 output_dir=self.root / "notes", researcher_concurrency=3,
                                 database_url="sqlite:///" + str(self.root / "test.db"))
        self.db = Database(self.settings.database_url)
        self.db.ensure_schema()
        self.files = FileStore(self.settings.data_dir, self.settings.output_dir, self.root)
        self.discovery = DiscoveryProvider(backends={})
        self.addCleanup(self.discovery.close)
        self.registry = ToolRegistry(specs())
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection")):
            blocker = patch.object(owner, name, side_effect=AssertionError("network prohibited"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def task(self):
        """创建隔离根任务，不调用HTTP或模型。"""
        tid = uuid4().hex
        self.db.create_task(tid, None, uuid4().hex, "读取合成论文，不保存", "deny", [])
        return tid

    def paper(self, tid, name="paper"):
        """授予本测试合成材料，不借生产身份。"""
        p = PaperCandidate(paper_id=name, title=name, source="fixture",
                           paper_url="https://fixture.invalid/" + name + ".pdf")
        self.db.save_paper(name, p.model_dump(mode="json"))
        self.db.grant_paper(tid, name, tid)

    def service(self, stream):
        """用真实PDFProvider承接合成字节流。"""
        self.calls = []
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: self.calls.append(request) or httpx.Response(200, stream=stream)))
        self.addCleanup(client.close)
        return PaperService(self.db, self.discovery, PDFProvider(self.files, client=client))

    def assert_no_evidence(self, tid):
        """未完成流不提交新文档、页或任务授权。"""
        self.assertEqual(self.db.fetchall("SELECT * FROM documents"), [])
        self.assertEqual(self.db.fetchall("SELECT * FROM page_evidence"), [])
        self.assertEqual(self.db.list_task_evidence(tid), [])
        self.assertEqual(list(self.root.rglob("*.pdf")), [])

    def test_tool_callback_reaches_stream_and_returns_cancelled(self):
        """ctx.cancelled在流中触发，不能当作来源失败或继续解析。"""
        tid, flag = self.task(), threading.Event()
        self.paper(tid)
        entered, release = threading.Event(), threading.Event()
        release.set()
        stream = ControlledStream(entered, release, flag.set)
        service = self.service(stream)
        ctx = ToolContext(tid, SavePolicy.deny, cancelled=flag.is_set, services={"papers": service})
        obs = self.registry.execute("read_paper", {"target_ref": "paper", "page_count": 1}, ctx)
        self.assertFalse(obs.ok)
        self.assertEqual(obs.kind, "cancelled")
        self.assertFalse(obs.retryable)
        self.assertEqual(obs.data["error_code"], "PAPER_READ_CANCELLED")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(stream.closed)
        self.assertFalse(stream.overread)
        self.assert_no_evidence(tid)

    def test_root_cancel_is_checked_when_caller_callback_is_false(self):
        """子读取也观察根状态，不依赖调用方重复传相同取消标记。"""
        root = self.task()
        child = self.db.create_subtask(root, root, 0, "synthetic read", {}, "fixture")
        tid = child["subtask_id"]
        self.paper(tid)
        entered, release = threading.Event(), threading.Event()
        release.set()
        stream = ControlledStream(entered, release, lambda: self.db.update_task(root, cancel_requested=1))
        service = self.service(stream)
        obs = self.registry.execute("read_paper", {"target_ref": "paper", "page_count": 1},
            ToolContext(tid, SavePolicy.deny, services={"papers": service}))
        self.assertEqual(obs.kind, "cancelled")
        self.assertTrue(stream.closed)
        self.assert_no_evidence(tid)

    def test_legacy_ingest_signature_remains_compatible(self):
        """无新作用域接口的旧替身仍可正常读回并授权。"""
        tid = self.task()
        self.paper(tid)
        service = PaperService(self.db, self.discovery, LegacyPDF())
        obs = self.registry.execute("read_paper", {"target_ref": "paper"},
            ToolContext(tid, SavePolicy.deny, services={"papers": service}))
        self.assertTrue(obs.ok, obs.message)
        self.assertTrue(self.db.task_has_evidence(tid, "legacy-page"))

    def test_actual_parent_pool_returns_after_stream_cancel_and_retains_two_results(self):
        """不替换父循环：两子先完成，第三读取真实关闭后父批次才返回。"""
        root = self.task()
        claimed = self.db.claim_task("fixture-worker", task_id=root)
        lease = claimed["lease_token"]
        for name in ("done-A", "done-B", "blocked"):
            self.paper(root, name)
        entered, release = threading.Event(), threading.Event()
        stream = ControlledStream(entered, release)
        service = self.service(stream)
        case = self

        class NoModel:
            """角色图为固定替身，不允许模型请求。"""

            def close(self):
                """无网络资源。"""

        class ChildGraph:
            """只替换子角色生成边界，真实parent批次、取文与存储仍运行。"""

            def __init__(self, services):
                self.services = services

            def invoke(self, state, config=None):
                name = self.services["scope"]["paper_ids"][0]
                if name != "blocked":
                    return {**state, "final_answer": name + " retained finding",
                            "structured_result": {"status": "completed", "findings": [name + " retained finding"],
                                                  "paper_ids": [name]}}
                ctx = ToolContext(state["task_id"], SavePolicy.deny,
                    cancelled=lambda: bool(case.db.get_task(root).get("cancel_requested")),
                    services=self.services)
                obs = case.registry.execute("read_paper", {"target_ref": name, "page_count": 1}, ctx)
                case.assertEqual(obs.kind, "cancelled")
                return {**state, "cancelled": True}

        worker = TaskExecutor(self.settings, self.db, lambda _: NoModel())
        tasks = [{"objective": "read " + name, "mode": "investigate", "paper_ids": [name]}
                 for name in ("done-A", "done-B", "blocked")]
        with patch("research_loop.workers.executor.build_researcher_graph",
                   side_effect=lambda provider, services, *args, **kwargs: ChildGraph(services)), \
                ThreadPoolExecutor(max_workers=1) as parent_pool:
            parent = parent_pool.submit(worker._run_research_batch, root, tasks,
                                        "fixture-worker", lease, self.settings, service)
            try:
                self.assertTrue(entered.wait(3))
                deadline = time.monotonic() + 3
                while sum(c["status"] == "completed" for c in self.db.list_subtasks(root)) < 2:
                    if time.monotonic() > deadline:
                        self.fail("two completed child results not committed")
                    time.sleep(0.01)
                TaskService(self.db).cancel(root)
                self.assertFalse(parent.done(), "database cancellation must not pretend live stream stopped")
            finally:
                release.set()
            results = parent.result(timeout=5)
        self.assertTrue(stream.closed)
        self.assertFalse(stream.overread)
        self.assertEqual([r["status"] for r in results], ["completed", "completed", "cancelled"])
        delivered = worker._deliverable_result(root, {}, "用户取消，保留已有成果")
        self.assertIn("done-A retained finding", delivered.answer_markdown)
        self.assertIn("done-B retained finding", delivered.answer_markdown)
        TaskService(self.db).complete(root, TaskStatus.cancelled, delivered,
                                      owner=("fixture-worker", lease))
        self.assertEqual(self.db.get_task(root)["status"], "cancelled")
        self.assertEqual(self.db.fetchall("SELECT * FROM model_calls"), [])
        self.assertEqual(len(self.calls), 1)
        self.assert_no_evidence(root)
