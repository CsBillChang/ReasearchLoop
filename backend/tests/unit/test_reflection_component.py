"""Reflection组件评测的真实服务接线与有界失败测试；模型替身不代表语义通过。"""
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.providers.deepseek import RequestNotSent
from scripts.evaluate_reflection_component import (execute_case, ComponentLedger, FrozenPDF, PHASE_CAPS)


def fixture(kind='contradicted'):
    """创建两主张/一页原文合成样例，不依赖论文名称或网络。"""
    claim = {'correct': '实际方法训练全部组件。', 'contradicted': '实际方法冻结全部组件。',
             'insufficient': '驱动版本是535.104.05。'}[kind]
    return ({'id': kind, 'material': 'fixture', 'markdown': '# 短稿\n\n'+claim+'\n\n第二句正确。'},
        {'key': 'fixture', 'material': {'path': '/tmp/controlled.pdf'}, 'document': {
            'document_id': 'doc-fixture', 'paper_id': 'paper-fixture', 'content_sha256': 'a'*64,
            'config_sha256': 'b'*64, 'parser_name': 'frozen', 'parser_version': '1',
            'cache_path': '/tmp/no-cache-read', 'total_pages': 1, 'gaps': []},
         'pages': [{'evidence_id': 'ev-fixture', 'document_id': 'doc-fixture', 'page_number': 1,
             'text': 'Actual method trains all components. The second statement is correct.',
             'parsed_markdown': '', 'assets': [], 'gaps': []}]})


class ScriptedProvider:
    """只构造供应商结构；计数/落盘/工具执行/校验/新hash复核均由产品代码完成。"""
    def __init__(self, kind, bad_quote=False, bad_patch=False, unknown=False, case=None, material=None):
        """配置已冻结的替身分支；显式输入非真实语义判定。"""
        self.kind, self.bad_quote, self.bad_patch, self.unknown = kind, bad_quote, bad_patch, unknown
        self.messages = []; self.case = case; self.material = material

    def close(self):
        """替身无网络连接。"""

    def send(self, messages, tools=None, **kwargs):
        """依据合成测试步骤返回现有工具合同，绝不在真实评测中启用。"""
        self.messages.append(messages)
        if self.unknown:
            raise TimeoutError('synthetic unknown after send')
        source = self.material or fixture(self.kind)[1]
        case = self.case or fixture(self.kind)[0]
        ref = source['pages'][0]['evidence_id']; quote = source['pages'][0]['text'][:60]
        tool_names = {t['function']['name'] for t in tools or []}
        if 'revise_note' in tool_names:
            data = json.loads(next(m['content'] for m in messages if m.get('role') == 'user' and m['content'].startswith('{"current_draft"')))
            draft, review = data['current_draft'], data['current_review']
            old = case['markdown'].split('\n\n')[1]
            replacement = '在所提供原文中未能确认该具体细节。' if self.kind == 'insufficient' else '实际方法训练全部组件。'
            args = {'draft_id': draft['draft_id'], 'content_hash': draft['content_hash'], 'review_id': review['review_id'],
                'patches': [{'finding_index': 0, 'old_text': old, 'new_text': replacement,
                    'reason': 'synthetic-local-reason-do-not-leak-to-reviewer', 'evidence_ids': ['wrong-ref'] if self.bad_patch else [ref]}]}
            name = 'revise_note'
        elif 'finish_review' in tool_names:
            user = next(m['content'] for m in messages if '稿件:\n' in m.get('content', ''))
            initial = case['markdown'] in user
            status = self.kind if initial and self.kind != 'correct' else 'supported'
            first = case['markdown'].split('\n\n')[1] if initial else '修订后的第一句'
            finding = {'claim': first, 'status': status, 'reason': 'synthetic-review-reason-do-not-leak',
                'basis': 'not_found' if status == 'insufficient' else 'direct_statement', 'evidence_ids': [ref],
                'source_quote': None if status == 'insufficient' else ('fabricated absent quotation' if self.bad_quote else quote)}
            second = {**finding, 'claim': case['markdown'].split('\n\n')[2].strip(), 'status': 'supported',
                'basis': 'direct_statement', 'source_quote': quote}
            args = {'status': 'supported' if status == 'supported' else 'issues', 'findings': [finding, second], 'gaps': []}
            name = 'finish_review'
        else:
            return {'id': str(uuid4()), 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
                'choices': [{'message': {'role': 'assistant', 'content': '未完成，保留已有成果'}, 'finish_reason': 'stop'}]}
        return {'id': str(uuid4()), 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': str(uuid4()), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]},
                'finish_reason': 'tool_calls'}]}


class ComponentTests(unittest.TestCase):
    """完整调用产品Reflection、Reviewer、patch与持久化收尾。"""
    def run_case(self, kind, **options):
        """返回隔离组件轨迹，临时库和文件在测试后清理。"""
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = Settings(project_root=root, database_url='sqlite:///'+str(root/'test.db'),
            data_dir=root/'data', output_dir=root/'notes', retrieval_enabled=False,
            max_model_requests=11,max_tool_actions=30,reserve_model_requests=0)
        db = Database(config.database_url); db.ensure_schema()
        case, material = fixture(kind)
        provider = ScriptedProvider(kind, **options)
        result = execute_case(db, config, case, material, lambda settings: provider)
        # 组件固定11/30/0在新根创建时冻结，JSON与数据库额度必须一致。
        tid=result['task']['task_id']
        db.assert_budget_policy(tid,config)
        frozen=db.execution_policy(tid)
        self.assertEqual((frozen['max_model_requests'],frozen['max_tool_actions'],frozen['reserve_model_requests']),(11,30,0))
        self.assertEqual(frozen['mode'],'enforce')
        self.assertFalse(list(root.rglob('*.md')))
        self.assertTrue(json.loads(result['task']['result_json'])['answer_markdown'])
        self.assertEqual(db.fetchone("SELECT COUNT(*) AS n FROM tasks WHERE status IN ('running','queued')")['n'], 0)
        return result, provider

    def test_correct_preserved(self):
        """正确稿核验一次即结束，不调用patch或额外复审。"""
        result, provider = self.run_case('correct')
        self.assertEqual(result['baseline']['markdown'], result['current']['markdown'])
        self.assertEqual(len(provider.messages), 1)
        self.assertEqual(result['task']['status'], 'completed')

    def test_patch_and_independent_review(self):
        """修订完整落库；复审绑定新hash，未混入旧意见，非目标文字逐字保留。"""
        result, provider = self.run_case('contradicted')
        self.assertEqual(len(provider.messages), 3)
        self.assertEqual(result['task']['status'], 'completed')
        self.assertNotEqual(result['baseline']['content_hash'], result['current']['content_hash'])
        self.assertTrue(result['checks']['non_target_text_preserved'])
        final = json.dumps(provider.messages[-1], ensure_ascii=False)
        self.assertNotIn('synthetic-review-reason-do-not-leak', final)
        self.assertNotIn('synthetic-local-reason-do-not-leak', final)
        self.assertIn(result['current']['content_hash'], final)
        self.assertEqual(len(result['reflection']['patches']), 1)

    def test_missing_fact_is_disclosed(self):
        """缺证可披露，不被程序替换为反证或新的猜测数值。"""
        result, _ = self.run_case('insufficient')
        self.assertIn('未能确认', result['current']['markdown'])
        self.assertNotIn('535.104.05', result['current']['markdown'])

    def test_invalid_review_is_failure_not_quality(self):
        """不合法引句有界结束；不得把审校执行失败记为正确或启动修订。"""
        result, provider = self.run_case('contradicted', bad_quote=True)
        self.assertLessEqual(len(provider.messages), 4)
        self.assertEqual(result['final_review']['status'], 'execution_failed')
        self.assertEqual(result['reflection']['patches'], [])
        self.assertEqual(result['task']['status'], 'partial')

    def test_invalid_patch_keeps_original(self):
        """未授权证据patch被拒；停止并交付原稿，不重写整篇或偷用复审额度。"""
        result, provider = self.run_case('contradicted', bad_patch=True)
        self.assertEqual(result['baseline']['markdown'], result['current']['markdown'])
        self.assertLessEqual(len(provider.messages), 4)
        self.assertEqual(len(result['reflection']['attempts']), 1)

    def test_unknown_stops_without_patch(self):
        """供应商结果未知单列，不能自动再发或将unknown算零。"""
        result, provider = self.run_case('correct', unknown=True)
        self.assertEqual(len(provider.messages), 1)
        self.assertEqual(result['usage']['unknown'], 1)
        self.assertEqual(result['task']['status'], 'partial')

    def test_source_and_page_boundaries(self):
        """缓存适配拒绝不同材料/越界页，不解析或下载新PDF。"""
        _, material = fixture(); pdf = FrozenPDF([material])
        with self.assertRaises(PermissionError): pdf.ingest('/other.pdf', 'paper-fixture')
        doc = pdf.ingest('/tmp/controlled.pdf', 'paper-fixture')
        with self.assertRaises(ValueError): pdf.pages(doc, 1, 2)

    def test_phase_cap_and_unknown_usage(self):
        """阶段预算不挪用；缺usage在下次发送前阻断。"""
        with tempfile.TemporaryDirectory() as directory:
            ledger = ComponentLedger(Path(directory)/'calls.json', {'physical_cap': 44, 'token_cap': 1200000,
                'wall_seconds': 1800, 'cases': [{'id': 'F', 'request_cap': 11}]})
            ledger.case = 'F'; ledger.phase = 'revision'
            messages = [{'role': 'user', 'content': '核验'}]
            for _ in range(PHASE_CAPS['revision']):
                n = ledger.reserve('send', messages, None)
                ledger.settle(n, {'id': str(uuid4()), 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
            with self.assertRaises(RequestNotSent): ledger.reserve('send', messages, None)
            ledger.phase = 'final_review'
            n = ledger.reserve('send', messages, None); ledger.settle(n, {'id': 'missing', 'usage': None})
            with self.assertRaises(RequestNotSent): ledger.reserve('send', messages, None)
