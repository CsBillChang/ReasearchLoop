"""固定验收集的语义记分/技术停止合同；仅合成终稿及本机临时文件。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import harness_ab_pair_gate as gate


class AgentFinalGateTests(unittest.TestCase):
    """防止继续记分意外放宽旧首错、未知或权限问题。"""

    def setUp(self):
        """创建独立的两份合成终稿，绝不创建真实模型任务。"""
        self.temporary = tempfile.TemporaryDirectory(prefix='agent-final-gate-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cid = 'AFE-MEM-01-R1'
        self.case = {'case_id': self.cid, 'suite_kind': 'agent_final_v1',
            'prompt': 'Synthetic history continuation', 'materials': [],
            'preconditions': {'catalog': [], 'save_policy': 'deny',
                'history_messages': [{'role': 'user', 'content': 'Synthetic seed'}]}}
        self.cases = self.root / 'cases.json'
        self.cases.write_text(json.dumps({'cases': [self.case]}))
        self.manifest = {'experiment_kind': 'prompt_bundle_v1', 'cases_path': str(self.cases),
            'runtime_root': str(self.root / 'runtime'),
            'semantic_gate_root': str(self.root / 'control'),
            'final_suite_policy': deepcopy(gate.FINAL_SUITE_POLICY),
            'pair_review_policy': {'poll_seconds': 0.5},
            'execution_order': [[self.cid, 'A'], [self.cid, 'B'], ['AFE-NEXT-R1', 'A']]}
        self.rows = []
        for variant in ('A', 'B'):
            root_id = 'synthetic-root-' + variant
            cell = self.root / 'runtime/cells' / (self.cid + '-' + variant)
            cell.mkdir(parents=True)
            values = {'result.json': {'status': 'executed', 'root_id': root_id},
                'task-view.json': {'task_id': root_id, 'status': 'completed', 'prompt': self.case['prompt'],
                    'result': {'answer_markdown': 'Synthetic response ' + variant}},
                'model-calls.json': [], 'actions.json': {}}
            for name, value in values.items():
                (cell / name).write_text(json.dumps(value))
            self.rows.append({'case_id': self.cid, 'variant': variant, 'root_id': root_id, 'issues': []})

    def challenge(self):
        """生成一次性匿名挑战，实际计算终稿及执行工件摘要。"""
        return gate.prepare_pair_gate(self.manifest, self.cid, self.rows,
            gate.manifest_identity(self.manifest), clock=lambda: 100)

    def review(self, challenge, labels=('full', 'full')):
        """固定为合成审阅，避免将测试判词当成实际质量分。"""
        return {'version': 1, 'challenge_nonce': challenge['nonce'],
            'challenge_sha256': gate.sha256(gate.canonical(challenge)), 'reviewed_at': 101,
            'reviewer': {'kind': 'human', 'identity': 'synthetic contract reviewer',
                'blinded': True, 'full_authorized_source_checked': True, 'paid_model_calls': 0},
            'reviews': [{'anonymous_id': challenge['anonymous_ids'][variant],
                'answer_sha256': challenge['identity']['answers'][variant]['answer_sha256'],
                'source_sha256s': [], 'whole_answer_quality': label,
                'core_complete': True, 'critical_core_error': False, 'requested_scope_complete': True,
                'execution_complete': True, 'unauthorized_delivery_write': False, 'forged_success': False,
                'evidence_notes': 'Synthetic only; no real answer has been graded.'}
                for variant, label in zip(('A', 'B'), labels)]}

    def evaluate(self, challenge, review):
        """执行真实审阅门函数，固定时钟以检查身份而非耗时。"""
        return gate.evaluate_pair_review(self.manifest, self.cid, self.rows,
            gate.manifest_identity(self.manifest), challenge, review, clock=lambda: 102)

    def test_legacy_joint_partial_pair_still_stops_without_execution_flags(self):
        """新合同缺席时，原双方partial首错与旧判词格式不变。"""
        self.manifest.pop('final_suite_policy')
        challenge = self.challenge(); review = self.review(challenge, ('partial_usable', 'partial_usable'))
        for answer in review['reviews']:
            for key in ('execution_complete', 'unauthorized_delivery_write', 'forged_success'):
                del answer[key]
        decision = self.evaluate(challenge, review)
        self.assertEqual(decision['status'], 'stopped')
        self.assertIn('both_answers_below_full_quality', decision['stop_reasons'])
        self.assertNotIn('semantic_findings', decision)

    def test_fixed_partial_pair_continues_but_preserves_low_quality(self):
        """继续的是固定分母，质量仍记录partial且不算等价。"""
        challenge = self.challenge(); review = self.review(challenge, ('partial_usable', 'partial_usable'))
        decision = self.evaluate(challenge, review)
        self.assertEqual(decision['status'], 'passed')
        self.assertTrue(decision['continuation_authorized'])
        self.assertFalse(decision['quality_equivalent'])
        self.assertEqual(decision['quality'], {'A': 'partial_usable', 'B': 'partial_usable'})
        self.assertIn('both_answers_below_full_quality', decision['semantic_findings'])
        self.assertIn('B_quality_partial_usable', decision['semantic_findings'])

    def test_semantic_core_error_and_candidate_regression_are_recorded(self):
        """B的核心错漏仍保留证据，不再导致普通语义提前截断。"""
        challenge = self.challenge(); review = self.review(challenge, ('full', 'unusable'))
        review['reviews'][1].update(core_complete=False, critical_core_error=True,
            requested_scope_complete=False, execution_complete=False)
        decision = self.evaluate(challenge, review)
        self.assertEqual(decision['status'], 'passed')
        self.assertFalse(decision['quality_equivalent'])
        self.assertIn('candidate_quality_regression', decision['semantic_findings'])
        self.assertIn('B_core_error_or_missing', decision['semantic_findings'])
        self.assertIn('B_execution_incomplete', decision['semantic_findings'])

    def test_failed_label_is_a_recorded_semantic_failure(self):
        """新评分合同的failed标签可保留，不需要伪装成full放行。"""
        challenge = self.challenge(); review = self.review(challenge, ('failed', 'full'))
        decision = self.evaluate(challenge, review)
        self.assertEqual(decision['status'], 'passed')
        self.assertIn('A_quality_failed', decision['semantic_findings'])
        self.assertFalse(decision['quality_equivalent'])

    def test_unknown_or_not_assessable_halts(self):
        """证据不够不能按普通失败盲目继续，更不能填full。"""
        challenge = self.challenge()
        for label in ('unknown', 'not_assessable'):
            with self.subTest(label=label):
                decision = self.evaluate(challenge, self.review(challenge, ('full', label)))
                self.assertEqual(decision['status'], 'stopped')
                self.assertFalse(decision['continuation_authorized'])
                self.assertIn('B_review_unknown', decision['stop_reasons'])

    def test_unauthorized_write_and_forged_success_halt(self):
        """严重写入/伪造的判定不是可继续的普通语义错误。"""
        challenge = self.challenge()
        for flag in ('unauthorized_delivery_write', 'forged_success'):
            with self.subTest(flag=flag):
                review = self.review(challenge, ('partial_usable', 'full'))
                review['reviews'][0][flag] = True
                decision = self.evaluate(challenge, review)
                self.assertEqual(decision['status'], 'stopped')
                self.assertIn('A_' + flag, decision['stop_reasons'])

    def test_new_execution_flags_are_explicit_booleans(self):
        """缺检查结果与用0代替False均拒绝，避免静默推导权限安全。"""
        challenge = self.challenge()
        for key in ('execution_complete', 'unauthorized_delivery_write', 'forged_success'):
            for value in (None, 0, 'false'):
                with self.subTest(key=key, value=value):
                    review = self.review(challenge); review['reviews'][0][key] = value
                    with self.assertRaisesRegex(PermissionError, 'EXECUTION_FLAGS_REQUIRED'):
                        self.evaluate(challenge, review)

    def test_full_requires_execution_completion_and_no_severe_flag(self):
        """full不能同时承认未执行必要操作或存在越权/伪造。"""
        challenge = self.challenge()
        for key, value in [('execution_complete', False), ('unauthorized_delivery_write', True), ('forged_success', True)]:
            review = self.review(challenge); review['reviews'][0][key] = value
            with self.assertRaisesRegex(PermissionError, 'FULL_LABEL_CONTRADICTS'):
                self.evaluate(challenge, review)

    def test_backend_partial_cannot_receive_full_new_grade(self):
        """后端已标partial的产物不能被full判词掩盖。"""
        path = self.root / 'runtime/cells' / (self.cid + '-A') / 'task-view.json'
        value = gate.read_json(path); value['status'] = 'partial'; path.write_text(json.dumps(value))
        challenge = self.challenge()
        with self.assertRaisesRegex(PermissionError, 'FULL_LABEL_CONTRADICTS'):
            self.evaluate(challenge, self.review(challenge))

    def test_new_full_pair_keeps_quality_equivalent(self):
        """只有完整执行与语义full两侧均满足，才标记等价。"""
        challenge = self.challenge(); decision = self.evaluate(challenge, self.review(challenge))
        self.assertEqual(decision['status'], 'passed')
        self.assertTrue(decision['quality_equivalent'])
        self.assertEqual(decision['semantic_findings'], [])
        packet = gate.read_json(self.root / 'control' / self.cid / 'review-packet.json')
        self.assertEqual(packet['review_protocol']['required_execution_flags'],
            ['execution_complete', 'unauthorized_delivery_write', 'forged_success'])

    def test_review_or_artifact_tamper_rejected_even_when_continuation_allowed(self):
        """更换匿名ID、答案或用量工件均不能借新策略绕过身份。"""
        challenge = self.challenge(); review = self.review(challenge, ('partial_usable', 'partial_usable'))
        for key, value in [('anonymous_id', 'wrong'), ('answer_sha256', '0'*64), ('source_sha256s', ['1'*64])]:
            with self.subTest(key=key):
                altered = deepcopy(review); altered['reviews'][0][key] = value
                with self.assertRaises(PermissionError): self.evaluate(challenge, altered)
        receipt = self.root / 'runtime/cells' / (self.cid + '-B') / 'model-calls.json'
        receipt.write_text('[{"changed":true}]')
        with self.assertRaisesRegex(PermissionError, 'CHALLENGE_OR_SOURCE_MISMATCH'):
            self.evaluate(challenge, review)

    def test_policy_is_strictly_versioned_and_joint_only(self):
        """拼错版本、宽松策略或借用其他旧实验名均先拒绝。"""
        for policy in [None, {}, {'version': 'agent_final_v0', 'semantic_failures': 'record_and_continue'},
                {**gate.FINAL_SUITE_POLICY, 'unknown': 'continue'}]:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(PermissionError, 'FINAL_SUITE_POLICY_INVALID'):
                    gate.validate_final_suite_policy({**self.manifest, 'final_suite_policy': policy})
        with self.assertRaisesRegex(PermissionError, 'FINAL_SUITE_POLICY_INVALID'):
            gate.validate_final_suite_policy({**self.manifest, 'experiment_kind': 'supervisor_goal_v1'})

    def test_prior_partial_decision_allows_next_grant_but_tamper_does_not(self):
        """独立格重验历史判词；继续必须基于同一挑战，不只相信status字符串。"""
        gate.initialize_scope_session(self.manifest, 90, clock=lambda: 90)
        challenge = self.challenge(); review = self.review(challenge, ('partial_usable', 'partial_usable'))
        decision = self.evaluate(challenge, review)
        gate.write_new(self.root / 'control' / self.cid / 'review.json', review)
        gate.write_new(self.root / 'control' / self.cid / 'decision.json', decision)
        with patch.object(gate, 'paired_source_contract', return_value={'synthetic': 'only'}):
            gate.issue_cell_grant(self.manifest, 'AFE-NEXT-R1', 'A', 110)
            active = {**self.manifest, 'cell_absolute_deadline': 110}
            gate.validate_prior_pair_gates(active, 'AFE-NEXT-R1', 'A', clock=lambda: 102)
            path = self.root / 'control' / self.cid / 'decision.json'
            changed = deepcopy(decision); changed['quality']['B'] = 'full'; path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(PermissionError, 'PAIR_PRIOR_REVIEW_REQUIRED'):
                gate.validate_prior_pair_gates(active, 'AFE-NEXT-R1', 'A', clock=lambda: 102)


if __name__ == '__main__':
    unittest.main()
