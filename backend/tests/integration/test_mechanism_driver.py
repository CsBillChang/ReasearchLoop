"""独立A/B业务进程→原生Worker→共享预算→Mock HTTP→所属组收尾。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from experiments.mechanism_pilot.contract import RunContract, settings_profile
from research_loop.config import Settings
from experiments.mechanism_pilot.lifecycle import Deadline
from experiments.mechanism_pilot.runner import execute_prepared_case, launch_case
from experiments.mechanism_pilot.resources import runtime_namespace


class DriverTests(unittest.TestCase):
    """需要显式本轮A/B源码目录；未配置时不冒称完成双版本串联。"""

    def setUp(self):
        paths = [os.getenv('MECHANISM_BASELINE_SOURCE'), os.getenv('MECHANISM_CANDIDATE_SOURCE')]
        if not all(paths):
            self.skipTest('explicit isolated A/B source roots required')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.batch = self.root/'runtime'
        self.batch.mkdir()
        self.sources = dict(zip(('A', 'B'), paths))
        material = {'case_id': 'M1', 'history': [],
            'prompt': '只依据本条数据说明保留6条，不找论文、不保存文件。'}
        self.input = self.root/'input.json'
        self.input.write_text(json.dumps(material))
        self.manifest = {'files': [{'path': str(self.input),
            'sha256': hashlib.sha256(self.input.read_bytes()).hexdigest()}],
            'runtime_root': str(self.batch), 'business_roots': self.sources,
            'model': 'deepseek-flash', 'context_window_tokens': 1000000,
            'phases': {'M': {'input_token_limit': 1048576, 'max_output_tokens': 4096}},
            'limits': {'tokens': 3000000, 'requests': 96, 'nano_yuan': 12000000000,
                       'admission_seconds': 2400, 'total_seconds': 2700},
            'egress': ['synthetic_fixture'],
            'cases': {'M1': {'phase': 'M', 'input_path': str(self.input),
                'input_sha256': hashlib.sha256(self.input.read_bytes()).hexdigest()}},
            'cells': [[f'R{i:02}', 'B'] for i in range(1, 7)] + [[f'M{i}', v] for i in range(1, 5) for v in ('A', 'B')]}
        self.manifest_path = self.root/'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.auth = self.root/'authorization.json'
        self.auth.write_text(json.dumps({'status': 'draft'}))
        self.freeze = RunContract(self.manifest, self.auth).freeze_sha256
        self.deadline = Deadline(time.time(), time.monotonic())
        self.manifest['settings_profiles'] = {'M': settings_profile(Settings(**self.payload('B')['settings']))}
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.freeze = RunContract(self.manifest, self.auth).freeze_sha256

    def payload(self, variant):
        root = self.batch/variant/'case'
        payload = {'phase': 'M', 'case_id': 'M1', 'variant': variant,
            'manifest_path': str(self.manifest_path), 'authorization_path': str(self.auth),
            'freeze_sha256': self.freeze, 'ledger_path': str(self.batch/'budget.sqlite3'),
            'started_at': self.deadline.started_wall,
            'execution_kind': 'substitute', 'input_path': str(self.input),
            'input_sha256': hashlib.sha256(self.input.read_bytes()).hexdigest(),
            'settings': {'project_root': str(root), 'data_dir': str(root/'data'),
                'output_dir': str(root/'output'), 'database_url': 'sqlite:///'+str(root/'runtime.sqlite3'),
                'deepseek_model': 'deepseek-flash', 'deepseek_base_url': 'https://fixture.invalid',
                'deepseek_api_key': None, 'context_window_tokens': 1000000,
                'context_output_reserve': 4096, 'model_max_output_tokens': 4096,
                'web_search_enabled': False, 'web_fetch_enabled': False,
                'retrieval_enabled': True, 'qdrant_url': 'http://127.0.0.1:16391',
                'retrieval_namespace': runtime_namespace(root),
                'reserve_model_requests': 0, 'max_model_requests': 8}}
        calls = [('get_profile', {}), ('set_research_plan', {'task_kind': 'memory',
            'goal': '说明现有数据', 'goals': [{'goal_id': 'derive', 'description': '说明保留条数', 'min_papers': 0}]}),
            ('finish_task', {'answer_markdown': '保留6条；合成资格结果，不表示真实模型效果。',
             'goals': [{'goal_id': 'derive', 'status': 'satisfied', 'reason': '按当前数据'}]})]
        payload['substitute_responses'] = [{'id': f'synthetic-{i}',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant', 'content': '',
                'reasoning_content': '合成推理哨兵', 'tool_calls': [{'id': f'call-{i}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}
            for i, (name, args) in enumerate(calls)]
        return payload

    def test_separate_ab_workers_share_physical_ledger_and_cleanup(self):
        bodies = []
        for variant in ('A', 'B'):
            payload = self.payload(variant)
            record = launch_case(payload, self.batch/variant, deadline=self.deadline, interpreter=sys.executable)
            self.assertTrue(record['cleanup_complete'], record)
            self.assertEqual(record['returncode'], 0,
                (record, (self.batch/variant/'process.log').read_text()))
            result = json.loads((Path(payload['settings']['project_root'])/'qualified-result.json').read_text())
            self.assertIsNone(result['global_stop'], result)
            self.assertEqual(result['view']['status'], 'completed')
            self.assertFalse(result['route_preselected'])
            self.assertEqual(result['actual_plan']['task_kind'], 'memory')
            self.assertEqual(result['strict_usage_summary']['total_tokens'], 45)
            self.assertEqual(result['real_model_calls'], 0)
            for item in result['source_imports'].values():
                self.assertTrue(Path(item['path']).is_relative_to(Path(self.sources[variant]).resolve()))
            bodies.append(result)
        self.assertEqual(bodies[-1]['budget']['physical_sent_requests'], 6)
        self.assertEqual(bodies[-1]['budget']['actual_known_tokens'], 90)
        self.assertNotEqual(bodies[0]['source_imports']['context']['sha256'],
                            bodies[1]['source_imports']['context']['sha256'])

    def test_real_without_fresh_grant_does_not_create_case_or_transport(self):
        payload = self.payload('B')
        payload.pop('substitute_responses')
        payload['execution_kind'] = 'real'
        payload['settings'].update(deepseek_base_url='https://api.deepseek.com', deepseek_api_key='synthetic-key')
        with self.assertRaises(PermissionError):
            execute_prepared_case(payload, inner_factory=lambda: self.fail('transport constructed'))
        self.assertFalse(Path(payload['settings']['project_root']).exists())
