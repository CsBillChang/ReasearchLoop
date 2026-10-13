"""AG6评测器合同：预算、固定分母、隔离和真实API/Worker接线，默认不联网。"""
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from research_loop.config import Settings
from scripts.run_ag6 import (ROOT, EvaluationLedger, EvaluationRunner,
                            summarize_usage, validate_isolation, validate_manifest)


class EvaluationContractTests(unittest.TestCase):
    """防止重复计费、删失败分母和把程序完成误报为语义通过。"""

    def setUp(self):
        """每例创建独立账本；只读取真实场景配置。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.manifest = json.loads((ROOT/'backend/tests/acceptance/ag6_cases.json').read_text())

    def test_exact_denominator_and_permissions(self):
        """删题、重复题、错误授权必须拒绝，完整清单有效。"""
        validate_manifest(self.manifest)
        for mutation in ('remove', 'duplicate', 'permission'):
            value = copy.deepcopy(self.manifest)
            if mutation == 'remove': value['cases'].pop()
            elif mutation == 'duplicate': value['cases'][1]['id'] = 'E01'
            else: value['cases'][1]['save_policy'] = 'yes'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_manifest(value)

    def test_production_or_escaping_destination_rejected(self):
        """数据库/文件隔离在创建资源前验证，不能影响生产或外部路径。"""
        good = 'mysql://user:password@127.0.0.1:33316/researchloop_test_ag6_fixture'
        self.assertTrue(validate_isolation(good, ROOT/'artifacts/test').startswith('researchloop_test_ag6_'))
        for url, path in [(good.replace('_test_ag6_fixture', ''), ROOT/'artifacts/test'),
                          (good.replace('127.0.0.1', 'external.example'), ROOT/'artifacts/test'),
                          (good, ROOT/'data/../outside')]:
            with self.assertRaises(ValueError): validate_isolation(url, path)

    def test_usage_deduplicates_resume_and_keeps_unknown(self):
        """同task澄清前后快照不重复加tokens，未知不是零。"""
        row = {'call_id': 'a', 'status': 'succeeded', 'usage_json': json.dumps({'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6})}
        unknown = {'call_id': 'b', 'status': 'unknown', 'usage_json': '{}'}
        result = summarize_usage([row, row, unknown])
        self.assertEqual(result['known_tokens']['total_tokens'], 6)
        self.assertEqual(result['unknown'], 1)
        self.assertEqual(result['missing_usage_call_ids'], ['b'])

    def test_concurrent_cap_before_network(self):
        """并行Researcher共用一把额度锁，多余申请不会被计为已发送。"""
        self.manifest['physical_cap'] = 3
        ledger = EvaluationLedger(self.out/'ledger.json', self.manifest)
        ledger.case = 'E01'
        def reserve(_):
            """只模拟申请，无网络和伪造论文结果。"""
            try: return ledger.reserve('json', [{'role':'user','content':'test'}], 100)
            except RuntimeError: return None
        with ThreadPoolExecutor(max_workers=8) as pool: values = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(v is not None for v in values), 3)
        self.assertEqual(len(ledger.records), 3)

    def test_unknown_blocks_new_request_and_ledger_overwrite(self):
        """远端未知后不允许再发请求，也不通过新建账本丢失旧记录。"""
        path = self.out/'ledger.json'
        ledger = EvaluationLedger(path, self.manifest); ledger.case = 'E01'
        n = ledger.reserve('stream', [{'role':'user','content':'test'}], 100)
        ledger.settle(n, None, RuntimeError('connection lost'))
        with self.assertRaises(RuntimeError): ledger.reserve('json', [], 100)
        with self.assertRaises(ValueError): EvaluationLedger(path, self.manifest)

    def test_token_and_time_admission(self):
        """输入估算超过余额或批次超时，不能开始下一次发送。"""
        for kind in ['tokens', 'time']:
            value = copy.deepcopy(self.manifest)
            if kind == 'tokens': value['token_cap'] = 10
            ledger = EvaluationLedger(self.out/(kind+'.json'), value); ledger.case='E01'
            if kind == 'time': ledger.started -= value['wall_seconds']+1
            from research_loop.providers.deepseek import RequestNotSent
            with self.assertRaises(RequestNotSent): ledger.reserve('json', [{'role':'user','content':'large'}], 100)
            self.assertEqual(ledger.records, [])

    def test_local_rejection_is_not_unknown_usage(self):
        """发送前拒绝无供应商usage，不混入已发送缺用量/远端未知。"""
        local = {'call_id':'local', 'status':'not_sent', 'sent_state':'not_sent', 'usage_json':'{}'}
        remote = {'call_id':'remote', 'status':'unknown', 'sent_state':'sent', 'usage_json':'{}'}
        result = summarize_usage([local, remote])
        self.assertEqual(result['unknown'], 1)
        self.assertEqual(result['missing_usage_call_ids'], ['remote'])
        self.assertEqual(result['not_sent_call_ids'], ['local'])

    def test_explicit_rejection_stops_next_step_before_send(self):
        """HTTP明确拒绝也应立即阻断后续步骤；未知usage不伪填零。"""
        from research_loop.providers.deepseek import ProviderResponseError
        ledger=EvaluationLedger(self.out/'rejected.json',self.manifest);ledger.case='E11'
        n=ledger.reserve('stream',[{'role':'user','content':'test'}],100)
        ledger.settle(n,None,ProviderResponseError(400,'missing required field',None))
        with self.assertRaises(RuntimeError):ledger.reserve('stream',[{'role':'user','content':'next step'}],100)
        self.assertEqual(len(ledger.records),1)
        result=summarize_usage([{'call_id':'rejected','status':'rejected','sent_state':'sent','usage_json':'{}'}])
        self.assertEqual(result['provider_failures'],1)
        self.assertEqual(result['unknown'],0)
        self.assertEqual(result['missing_usage_call_ids'],['rejected'])

    def test_real_api_worker_evidence_and_no_semantic_auto_pass(self):
        """实际API→隔离数据库→Worker；供应商明确使用替身，核对结果不可自动通过。"""
        from tests.integration.test_memory import MemoryProvider
        from research_loop.main import app
        out = self.out/'run'; out.mkdir()
        url = os.getenv('AG6_CONTRACT_DATABASE_URL') or 'sqlite:///'+str(out/'test.db')
        if not url.startswith('sqlite:'): validate_isolation(url, ROOT/'artifacts/test')
        settings = Settings(project_root=ROOT, data_dir=out/'data', output_dir=out/'notes', database_url=url)
        runner = EvaluationRunner(settings, out, self.manifest)
        provider = MemoryProvider(None)
        runner.provider = lambda _: provider
        case = copy.deepcopy(self.manifest['cases'][0])
        case['steps'] = [{'kind':'task','prompt':'请回顾有效偏好，不搜索论文，不保存文件。'}]
        with ExitStack() as stack:
            for module in ['tasks','product','notes']:
                stack.enter_context(patch('research_loop.api.'+module+'.get_settings', return_value=settings))
            client = stack.enter_context(TestClient(app))
            runner.case_started = time.monotonic()
            runner.scenario(client, case)
        result = runner.results[0]
        self.assertNotIn('execution_error', result, result.get('execution_error'))
        self.assertEqual(result['tasks'][0]['view']['status'], 'completed')
        self.assertEqual(len(result['tasks'][0]['receipts']), 3)
        self.assertEqual(result['semantic_verdict'], 'not_reviewed')
        summary = runner.summary()
        self.assertEqual(summary['planned_scenarios'], 12)
        self.assertEqual(len(summary['unexecuted']), 11)
        self.assertFalse(summary['stage_complete'])
        self.assertEqual(list((out/'notes').rglob('*.md')), [])


if __name__ == '__main__':
    unittest.main()
