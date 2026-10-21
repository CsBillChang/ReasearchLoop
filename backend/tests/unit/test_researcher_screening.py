"""筛选模式的交付边界；合成输入与服务替身，不调用模型、网络或数据库。"""
import copy
import unittest
from unittest.mock import Mock, patch

from research_loop.agents.researcher import finish_spec, researcher_state
from research_loop.domain.tasks import SavePolicy
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs as paper_specs
from research_loop.tools.registry import ToolRegistry


class ResearcherScreeningTests(unittest.TestCase):
    """验证模式优先、未核验披露和原有调查/授权边界。"""

    def setUp(self):
        """只模拟任务已授权身份及进展投影，原文核验仍使用实际合同。"""
        self.db = Mock()
        self.db.list_actions.return_value = []
        self.db.task_has_paper.side_effect = lambda task, ref: ref == 'candidate'
        self.db.task_has_evidence.side_effect = lambda task, ref: ref == 'page'
        self.papers = Mock(db=self.db)
        self.papers.evidence_scope.return_value = []
        self.papers.evidence_bundle.return_value = [{'text': 'A synthetic source statement.'}]
        self.context = ToolContext('child', SavePolicy.deny, services={
            'papers': self.papers, 'scope': {'mode': 'screen'}}, role='researcher')
        self.registry = ToolRegistry([finish_spec()])
        progress = patch('research_loop.agents.researcher.ResearchProgressService.prepare', return_value={})
        progress.start()
        self.addCleanup(progress.stop)
        self.payload = {'status': 'completed', 'findings': [
            'Synthetic Candidate，https://fixture.invalid/p：摘要描述语言引导定位；训练规模尚不能确认。'],
            'paper_ids': ['candidate'], 'evidence_ids': [], 'gaps': []}

    def finish(self, **updates):
        """执行与schema共用的正式工具入口，不直接绕过参数校验。"""
        return self.registry.execute('finish_research', {**self.payload, **updates}, self.context)

    def test_screen_with_pending_claims_needs_no_fulltext_assessment(self):
        """旧关注点不把摘要推荐变成逐条全文核验。"""
        claims = ['语言是否作用于框回归分支？']
        self.context.services['scope']['verification_claims'] = claims
        result = self.finish()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['status'], 'completed')
        self.assertEqual(result.data['pending_verification_claims'], claims)
        self.assertEqual(result.data['evidence_ids'], [])
        self.assertEqual(result.data['assessments'], [])
        self.papers.evidence_bundle.assert_not_called()

    def test_screen_abstract_assessment_does_not_downgrade_recommendation(self):
        """U1形状的摘要supported无页证据不批准，也不吞掉合法筛选。"""
        assessments = [{'claim': '摘要描述语言引导定位', 'status': 'supported',
                        'basis': 'direct_statement', 'reason': '摘要明确描述',
                        'source_quote': 'Language guides localization.'}]
        before = copy.deepcopy(assessments)
        result = self.finish(assessments=assessments)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['status'], 'completed')
        self.assertEqual(result.data['findings'], self.payload['findings'])
        self.assertEqual(result.data['assessments'], [])
        self.assertEqual(result.data['assessment_errors'], [])
        self.assertEqual(result.data['gaps'], [])
        self.assertIn('未作为原文科学核验结论接收', result.message)
        self.assertIn('可选后续关注点，不要求继续核验', result.message)
        self.assertEqual(assessments, before)
        self.papers.evidence_bundle.assert_not_called()

    def test_screen_does_not_accept_even_well_formed_scientific_verdict(self):
        """screen结果不借完整核验形状夹带未经执行的科学结论。"""
        result = self.finish(assessments=[{'claim': '候选结论', 'status': 'supported',
            'basis': 'direct_statement', 'reason': '输入自称支持',
            'evidence_ids': ['unverified-page'], 'source_quote': 'A synthetic source statement.'}])
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['assessments'], [])
        self.papers.evidence_bundle.assert_not_called()

    def test_screen_preserves_explicit_partial_and_gaps(self):
        """筛选仍可因候选不足或真实资料缺口主动报告partial。"""
        result = self.finish(status='partial', gaps=['仅找到一个候选。'])
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['status'], 'partial')
        self.assertIn('仅找到一个候选。', result.data['gaps'])

    def test_completed_screen_still_requires_findings_and_selected_candidates(self):
        """模式放宽不能制造没有候选或内容的completed。"""
        for updates in ({'findings': []}, {'paper_ids': []}):
            with self.subTest(updates=updates):
                result = self.finish(**updates)
                self.assertFalse(result.ok)
                self.assertIn('findings and selected paper_ids', result.message)

    def test_screen_rejects_unauthorized_selected_paper(self):
        """摘要筛选也必须使用本子任务已授权题录。"""
        result = self.finish(paper_ids=['foreign-candidate'])
        self.assertFalse(result.ok)
        self.assertIn('authorized to this Researcher', result.message)

    def test_screen_rejects_unauthorized_read_evidence(self):
        """允许空原文证据不允许伪造或越权引用页。"""
        result = self.finish(evidence_ids=['foreign-page'])
        self.assertFalse(result.ok)
        self.assertIn('must be read evidence IDs', result.message)

    def test_investigate_claims_still_require_exact_assessments(self):
        """显式调查主张不被screen收尾规则放宽。"""
        self.context.services['scope'] = {'mode': 'investigate', 'verification_claims': ['指定主张']}
        result = self.finish(status='partial')
        self.assertFalse(result.ok)
        self.assertIn('each exact verification_claim', result.message)
        result = self.finish(status='partial', assessments=[{'claim': '指定主张', 'status': 'supported'}])
        self.assertFalse(result.ok)
        self.assertIn('required assessment', result.message)

    def test_investigate_still_checks_actual_quote_and_permission(self):
        """调查支持判断仍核对真实引句，越权来源保持拒绝。"""
        self.context.services['scope'] = {'mode': 'investigate', 'verification_claims': ['指定主张']}
        assessment = {'claim': '指定主张', 'status': 'supported', 'basis': 'direct_statement',
                      'reason': '原文说明', 'evidence_ids': ['page'], 'source_quote': 'Invented quote.'}
        result = self.finish(status='partial', assessments=[assessment])
        self.assertFalse(result.ok)
        self.assertIn('source_quote must occur', result.message)
        self.papers.evidence_bundle.side_effect = PermissionError('evidence outside current task')
        result = self.finish(status='partial', assessments=[assessment])
        self.assertFalse(result.ok)
        self.assertIn('evidence outside current task', result.message)
        self.assertFalse(result.retryable)

    def test_investigate_can_return_insufficient_without_inventing_evidence(self):
        """无法核实可明确交回insufficient，不能把缺证当反证。"""
        self.context.services['scope'] = {'mode': 'investigate', 'verification_claims': ['指定主张']}
        result = self.finish(status='partial', assessments=[{'claim': '指定主张', 'status': 'insufficient',
            'basis': 'not_found', 'reason': '未取得所需正文。'}])
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['assessments'][0]['status'], 'insufficient')

    def test_deep_read_keeps_claims_pending_without_preverification(self):
        """精读取文仍保持既有的后续关注点交接。"""
        self.context.services['scope'] = {'mode': 'deep_read', 'verification_claims': ['指定主张']}
        result = self.finish()
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['pending_verification_claims'], ['指定主张'])
        self.assertEqual(result.data['assessments'], [])

    def test_mode_prompt_precedes_shared_source_rules_without_forced_verification(self):
        """筛选先收到本模式合同；全文调查与精读取文保留各自说明。"""
        screen = researcher_state('child', '推荐候选', brief={'mode': 'screen'})['messages'][0]['content']
        self.assertTrue(screen.startswith('本次mode=screen'))
        self.assertIn('不逐条核验', screen)
        self.assertNotIn('须用assessments逐项返回', screen)
        investigate = researcher_state('child', '比较方法', brief={'mode': 'investigate'})['messages'][0]['content']
        self.assertIn('调查任务默认可读取全文', investigate)
        self.assertIn('须用assessments逐项返回', investigate)
        names = {item['function']['name'] for item in ToolRegistry(paper_specs()).schemas('researcher')}
        self.assertIn('read_paper', names)


if __name__ == '__main__':
    unittest.main()
