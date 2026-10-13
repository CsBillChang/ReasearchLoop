"""缺证、明确反证与读取范围回归；使用合成来源，不宣称模型语义正确率。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from research_loop.agents.researcher import FinishResearchArgs, _finish
from research_loop.agents.reviewer import FinishReviewArgs
from research_loop.domain.evidence import ClaimAssessment, Document, PageEvidence
from research_loop.domain.tasks import TaskCreate
from research_loop.services.papers import PaperService
from research_loop.services.tasks import TaskService
from research_loop.services.verification import validate_assessments, quote_in_text
from research_loop.storage.mysql import Database


class VerificationTests(unittest.TestCase):
    """同一校验器接受直接支持/反证，并拒绝缺证升级和错误引用。"""

    def setUp(self):
        """构建临时任务和部分可读文档，不访问网络。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.db = Database('sqlite:///' + str(Path(temp.name) / 'db.sqlite')); self.db.ensure_schema()
        self.tid = TaskService(self.db).create(TaskCreate(prompt='核验实验条件')).task_id
        self.doc = Document(document_id=uuid4().hex, paper_id='fixture', content_sha256='a'*64,
            parser_name='fixture', parser_version='1', config_sha256='b'*64, total_pages=3, cache_path='unused', gaps=['部分公式无法解析'])
        self.db.save_document(self.doc.model_dump())
        self.ref = uuid4().hex
        self.text = 'The encoder uses four layers. The reported memory is 8 GB.'
        self.db.save_page(PageEvidence(evidence_id=self.ref, document_id=self.doc.document_id, page_number=1,
            text=self.text, assets=[{'asset_id':'image-1','page_number':1}], gaps=['图像未理解']).model_dump())
        self.db.mark_task_evidence(self.tid, self.ref)
        self.papers = PaperService(self.db, None, None)

    def assessment(self, **changes):
        """生成与授权来源绑定的判断。"""
        return ClaimAssessment.model_validate({'claim':'编码器有四层','status':'supported','basis':'direct_statement',
            'reason':'原文直接说明层数','evidence_ids':[self.ref],'source_quote':'The encoder uses four layers.', **changes})

    def test_actual_pdf_word_wrap_is_not_a_new_claim(self):
        """处理混合断词；不消除普通连字符、数字符号或否定词。"""
        text = 'The query-\nevidence model is fine-tuned fol-\nlowing prior work.'
        self.assertTrue(quote_in_text('The query-evidence model is fine-tuned following prior work.',text))
        self.assertFalse(quote_in_text('The query-evidence model is not fine-tuned following prior work.',text))
        self.assertFalse(quote_in_text('recovered data','re-covered data'))
        self.assertFalse(quote_in_text('value 24','value 2-\n4'))
        self.assertFalse(quote_in_text('value 2e4','value 2e-\n4'))
        self.assertFalse(quote_in_text('one phrase ... another','one phrase between another'))

    def test_missing_and_partial_are_not_contradictions(self):
        """负面检索/解析缺口不能被同一结构标为已证伪。"""
        for basis in ['not_found','incomplete_or_ambiguous']:
            for status in ['supported','contradicted']:
                with self.subTest(basis=basis, status=status), self.assertRaises(ValueError):
                    self.assessment(basis=basis, status=status)
            validate_assessments(self.tid, [self.assessment(status='insufficient',basis=basis,source_quote=None)], self.papers)

    def test_acronym_wrap_preserves_text_and_ordinary_hyphens(self):
        """大写缩写跨行可保持连字符；数字、否定词和非换行连字符仍不得改写。"""
        for quote, source in [('H-CLIP uses Det-Pro', 'H-\nCLIP uses Det-\nPro'),
                              ('CLIP uses text', 'CL-\nIP uses text')]:
            self.assertTrue(quote_in_text(quote, source))
        for quote, source in [('H CLIP', 'H-\nCLIP'), ('CLIP', 'CL-IP'),
                              ('2E4', '2E-\n4'), ('It works', 'It does not work')]:
            self.assertFalse(quote_in_text(quote, source))

    def test_review_reports_all_errors_without_changing_verdict(self):
        """一次指出结论冲突及多个坏引句；不替模型删除疑点或升级为通过。"""
        from research_loop.agents.reviewer import _finish as finish_review
        unresolved = self.assessment(status='insufficient', basis='not_found', source_quote=None)
        bad = [self.assessment(source_quote='The encoder uses eight layers.'),
               self.assessment(source_quote='The memory is 16 GB.')]
        args = FinishReviewArgs(status='supported', findings=[unresolved, *bad])
        original = args.model_dump()
        with self.assertRaises(ValueError) as error:
            finish_review(args, SimpleNamespace(task_id=self.tid, services={'papers':self.papers}))
        self.assertIn('supported verdict conflicts', str(error.exception))
        self.assertIn('assessment[1]', str(error.exception))
        self.assertIn('assessment[2]', str(error.exception))
        self.assertEqual(args.model_dump(), original)

    def test_positive_and_explicit_counterevidence_remain_usable(self):
        """直接反证仍可交付，不将全部问题强制降为未知。"""
        for item in [self.assessment(), self.assessment(claim='编码器有八层',status='contradicted')]:
            validate_assessments(self.tid, [item], self.papers)
        # PDF 换行只规范空白，不修改数字、否定词或来源关系。
        validate_assessments(self.tid,[self.assessment(source_quote='The encoder\nuses four layers.')],self.papers)
        for quote in ['The encoder uses eight layers.', 'The encoder uses four layers. memory is 8 GB.']:
            with self.assertRaises(ValueError):
                validate_assessments(self.tid,[self.assessment(source_quote=quote)],self.papers)
        with self.assertRaises(PermissionError):
            validate_assessments(self.tid,[self.assessment(evidence_ids=['foreign'])],self.papers)

    def test_read_scope_does_not_claim_unread_pages_or_images(self):
        """授权一页不代表全部三页，全部文字也不等于视觉理解。"""
        page = self.papers.evidence_bundle(self.tid,[self.ref])[0]
        self.assertEqual(page['read_scope']['included_physical_pages'],[1])
        self.assertFalse(page['read_scope']['all_physical_pages_included'])
        self.assertFalse(page['read_scope']['visual_content_interpreted'])
        self.assertEqual(len(page['assets']),1)
        self.assertIn('图像未理解', page['gaps']); self.assertIn('部分公式无法解析', page['gaps'])
        self.assertEqual(self.papers.evidence_scope(self.tid)[0]['asset_pages'],[1])

    def test_jina_unknown_pages_not_full_pdf_coverage(self):
        """无页码正文保留 null，不将未知映射写成全页覆盖。"""
        doc = self.doc.model_copy(update={'total_pages':None}); self.db.save_document(doc.model_dump())
        scope = self.papers.evidence_bundle(self.tid,[self.ref])[0]['read_scope']
        self.assertIsNone(scope['all_physical_pages_included'])
        self.assertEqual(scope['locator_kind'],'document_text')

    def test_research_handoff_requires_each_assigned_claim(self):
        """不能只交自由文本绕过已明确委派的核验问题；判断和范围同时交接。"""
        ctx = SimpleNamespace(task_id=self.tid, services={'papers':self.papers,
            'scope':{'mode':'investigate','verification_claims':['测试环境的版本为99']}})
        args = {'status':'partial','findings':['已读文字无法确认'],'evidence_ids':[self.ref]}
        with self.assertRaises(ValueError): _finish(FinishResearchArgs(**args),ctx)
        result = _finish(FinishResearchArgs(**args, assessments=[self.assessment(claim='测试环境的版本为99',
            status='insufficient',basis='not_found',source_quote=None)]),ctx)
        self.assertEqual(result.data['assessments'][0]['status'],'insufficient')
        self.assertEqual(result.data['evidence_scope'][0]['included_physical_pages'],[1])
        with self.assertRaises(ValueError):
            FinishReviewArgs(status='issues',findings=[{**self.assessment().model_dump(),'basis':'not_found'}])

    def test_frozen_legacy_review_schema_is_available_for_replay(self):
        """旧 checkpoint/成功收据不因新增字段被强制重新请求供应商。"""
        from research_loop.agents.reviewer import LegacyFinishReviewArgs, finish_spec
        old = {'status':'supported','findings':[{'claim':'已知条件','status':'supported','reason':'历史结论','evidence_ids':[self.ref]}]}
        self.assertIs(finish_spec(False).args_model, LegacyFinishReviewArgs)
        LegacyFinishReviewArgs.model_validate(old)
        with self.assertRaises(ValueError): FinishReviewArgs.model_validate(old)
