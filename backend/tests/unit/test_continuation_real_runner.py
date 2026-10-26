"""逐轮真实入口的纯离线前置合同；不读取真实凭据，不创建业务或模型请求。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


RUNNER_PATH = Path(__file__).resolve().parents[2] / 'experiments' / 'memory_continuation' / 'run_real.py'
spec = importlib.util.spec_from_file_location('continuation_runner_preflight_test', RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
previous_path = list(sys.path)
try:
    spec.loader.exec_module(runner)
finally:
    sys.path[:] = previous_path


class ContinuationRealRunnerTests(unittest.TestCase):
    """审阅身份、摘要覆盖与端点解析均在网络之前确定。"""

    def setUp(self):
        """仅使用本例临时工件，所有真实连接均拒绝。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for target in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.getaddrinfo'):
            blocker = patch(target, side_effect=AssertionError('runner单测禁止真实网络'))
            blocker.start()
            self.addCleanup(blocker.stop)

    def artifacts(self, turn='T1', task='synthetic-task', corrections=()) -> dict:
        """生成模型结果的合成替身工件，只用于前置校验测试。"""
        directory = self.root / turn
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'answer.md').write_text('本例是合成回答，尚未经过真实模型。')
        memory = {'task_id': task, 'continuation_memory': {'status': 'applied',
            'conclusions': [{'key': 'metric', 'text': '保留当前定义及条件。'}],
            'corrections': list(corrections)}}
        (directory / 'memory.json').write_text(json.dumps(memory, ensure_ascii=False))
        (directory / 'trace.json').write_text(json.dumps({'root': {'task_id': task}}))
        return memory

    def review(self, turn='T1', task='synthetic-task', **changes) -> dict:
        """绑定本例实际字节，只有测试明确赋值才形成合成通过记录。"""
        value = {**runner.semantic_review_binding(self.root, turn, task), 'passed': True,
                 'checks': {'answer_semantics': True, 'summary_semantics': True}, **changes}
        self.write_review(turn, value)
        return value

    def write_review(self, turn: str, value: dict) -> None:
        """只写临时目录中的审阅替身，不改真实pilot。"""
        (self.root / turn / 'semantic-review.json').write_text(json.dumps(value, ensure_ascii=False))

    @staticmethod
    def relation() -> dict:
        """给出可逐项绑定的合成关系身份，不含真实key或hash。"""
        return {'previous_task_id': 'synthetic-T1', 'previous_key': 'old_metric',
                'previous_content_hash': 'a' * 64, 'replacement_key': 'new_metric', 'relation': 'updates'}

    def test_valid_review_is_bound_to_task_and_three_actual_files(self):
        """明确布尔通过和三个实际工件hash完全匹配时才返回审阅。"""
        self.artifacts()
        review = self.review()
        self.assertEqual(runner.validate_semantic_review(self.root, 'T1', 'synthetic-task'), review)
        self.assertEqual(set(review['artifact_sha256']), {'answer.md', 'memory.json', 'trace.json'})

    def test_truthy_nonboolean_review_does_not_pass(self):
        """字符串false、字符串true与整数1都不能代替布尔True。"""
        self.artifacts()
        for value in ('false', 'true', 1, [], None):
            with self.subTest(value=value):
                self.review(passed=value)
                with self.assertRaisesRegex(ValueError, '未明确通过'):
                    runner.validate_semantic_review(self.root, 'T1', 'synthetic-task')

    def test_review_for_another_task_or_turn_is_rejected(self):
        """即使工件hash相同，另一任务或轮次的通过记录也不能复用。"""
        self.artifacts()
        for changed in ({'task_id': 'different-task'}, {'turn_id': 'T2'}):
            with self.subTest(changed=changed):
                self.review(**changed)
                with self.assertRaisesRegex(ValueError, '未绑定'):
                    runner.validate_semantic_review(self.root, 'T1', 'synthetic-task')

    def test_modified_artifact_invalidates_previous_review(self):
        """回答、摘要或trace任一字节变化后，旧审阅不能继续放行。"""
        for name in ('answer.md', 'memory.json', 'trace.json'):
            with self.subTest(name=name):
                self.artifacts()
                self.review()
                path = self.root / 'T1' / name
                path.write_text(path.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, '未绑定'):
                    runner.validate_semantic_review(self.root, 'T1', 'synthetic-task')

    def test_artifacts_with_wrong_internal_task_identity_are_rejected(self):
        """外部review不能为错绑的实际memory或trace任务身份背书。"""
        self.artifacts(task='another-task')
        self.review(task='another-task')
        value = runner.semantic_review_binding(self.root, 'T1', 'synthetic-task')
        self.write_review('T1', {**value, 'passed': True,
            'checks': {'answer_semantics': True, 'summary_semantics': True}})
        with self.assertRaisesRegex(ValueError, '不属于该根任务'):
            runner.validate_semantic_review(self.root, 'T1', 'synthetic-task')

    def test_answer_and_summary_each_require_semantic_pass(self):
        """仅回答通过或仅摘要通过都不允许继续付费轮次。"""
        self.artifacts()
        for checks in ({}, {'answer_semantics': True},
                       {'answer_semantics': True, 'summary_semantics': 'true'},
                       {'answer_semantics': False, 'summary_semantics': True}):
            with self.subTest(checks=checks):
                self.review(checks=checks)
                with self.assertRaisesRegex(ValueError, '逐项审阅'):
                    runner.validate_semantic_review(self.root, 'T1', 'synthetic-task')

    def test_applied_empty_conclusions_do_not_establish_summary_coverage(self):
        """applied状态本身不能把空摘要判作机制完成。"""
        memory = self.artifacts()
        memory['continuation_memory']['conclusions'] = []
        (self.root / 'T1' / 'memory.json').write_text(json.dumps(memory))
        self.review()
        self.assertFalse(runner.summary_has_conclusions(memory['continuation_memory']))
        with self.assertRaisesRegex(ValueError, '非空接续结论'):
            runner.validate_semantic_review(self.root, 'T1', 'synthetic-task')
        for note in (None, {}, {'status': 'applied', 'conclusions': [{'text': ' '}]}):
            self.assertFalse(runner.summary_has_conclusions(note))

    def test_t2_updates_type_alone_cannot_replace_target_semantic_review(self):
        """存在updates关系但没有具体目标语义审阅时不放行T3。"""
        self.artifacts('T2', corrections=[self.relation()])
        self.review('T2')
        with self.assertRaisesRegex(ValueError, '逐条审阅'):
            runner.validate_semantic_review(self.root, 'T2', 'synthetic-task')

    def test_t2_target_review_is_bound_to_actual_relationship(self):
        """逐关系判据有明确理由且身份匹配，才承认审阅覆盖了实际更新。"""
        relation = self.relation()
        self.artifacts('T2', corrections=[relation])
        assessment = {**relation, 'passed': True,
            'reason': '合成审核：旧固定分母定义更新为保留集合分母，覆盖率定义保持。'}
        review = self.review('T2', correction_target_reviews=[assessment])
        self.assertEqual(runner.validate_semantic_review(self.root, 'T2', 'synthetic-task'), review)

    def test_t2_wrong_target_failed_or_reasonless_review_is_rejected(self):
        """错目标、否决、空理由和不完整身份都不能作为关系审阅通过。"""
        relation = self.relation()
        self.artifacts('T2', corrections=[relation])
        good = {**relation, 'passed': True, 'reason': '合成关系审核说明。'}
        changes = [{'replacement_key': 'unrelated'}, {'passed': 'true'},
                   {'passed': False}, {'reason': ' '}, {'previous_key': None}]
        for changed in changes:
            with self.subTest(changed=changed):
                self.review('T2', correction_target_reviews=[{**good, **changed}])
                with self.assertRaises(ValueError):
                    runner.validate_semantic_review(self.root, 'T2', 'synthetic-task')

    def test_credentials_use_explicit_file_and_ignore_environment_override(self):
        """环境中的其他凭据、模型和URL不能改变已显式选择的文件配置。"""
        config = self.root / 'synthetic.env'
        config.write_text('DEEPSEEK_API_KEY=file-test-value\nDEEPSEEK_MODEL=file-model\n'
                          'DEEPSEEK_BASE_URL=https://api.deepseek.com/v1\n')
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'environment-other-value',
                'DEEPSEEK_MODEL': 'environment-model', 'DEEPSEEK_BASE_URL': 'https://localhost'}, clear=True):
            values = runner.credential_values(config)
        self.assertEqual(values['DEEPSEEK_API_KEY'], 'file-test-value')
        self.assertEqual(values['DEEPSEEK_MODEL'], 'file-model')
        self.assertEqual(values['DEEPSEEK_BASE_URL'], 'https://api.deepseek.com/v1')

    def test_environment_key_does_not_fill_missing_explicit_file_key(self):
        """显式配置文件没有key时，不能静默改用其他环境账户。"""
        config = self.root / 'synthetic.env'
        config.write_text('DEEPSEEK_MODEL=file-model\n')
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'environment-other-value'}, clear=True):
            with self.assertRaisesRegex(ValueError, '未配置模型凭据'):
                runner.credential_values(config)

    def test_nonpublic_ipv4_and_ipv6_dns_results_are_rejected(self):
        """官方域名也不能通过DNS解析进入本机、私网或多播地址。"""
        addresses = ['127.0.0.1', '10.0.0.1', '192.168.1.1', '169.254.1.1',
                     '0.0.0.0', '100.64.0.1', '224.0.0.1', '::1', '::',
                     'fc00::1', 'fe80::1', 'ff02::1', '::ffff:127.0.0.1']
        for address in addresses:
            with self.subTest(address=address):
                row = (socket.AF_INET6 if ':' in address else socket.AF_INET,
                       socket.SOCK_STREAM, 6, '', (address, 443))
                with self.assertRaises(PermissionError):
                    runner.public_model_ips([row])

    def test_public_ipv4_ipv6_dns_results_are_accepted_without_resolution(self):
        """纯函数接受给定公网单播结果，不自行DNS或联网。"""
        addresses = ['8.8.8.8', '2606:4700:4700::1111']
        rows = [(socket.AF_INET6 if ':' in value else socket.AF_INET,
                 socket.SOCK_STREAM, 6, '', (value, 443)) for value in addresses]
        self.assertEqual(runner.public_model_ips(rows), set(addresses))
        for rows in ([], [(None,)], [(None, None, None, None, ('not-an-ip', 443))]):
            with self.subTest(rows=rows):
                with self.assertRaises(PermissionError):
                    runner.public_model_ips(rows)


if __name__ == '__main__':
    unittest.main()
