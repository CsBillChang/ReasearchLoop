"""Reflection 新批次的零模型链路资格；所有材料及供应商行为均为合成。"""
from decimal import Decimal
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from integration.test_research_reflection import ResearchReflectionScenario, ScopedPDF
from research_loop.agents.reflection import ReflectionService
from research_loop.domain.reflection import ResearchReviewInput
from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent
from scripts.reflection_baseline_runtime import (BatchBudget, BudgetRefused, baseline_result,
    business_source_hashes, deny_socket_connections, run_substitute)


class BaselineScenario(ResearchReflectionScenario):
    """复用合成角色决策；拒绝修改分支只提交一次错误身份再如实结束。"""

    def __init__(self, mode='revision'):
        super().__init__('missing' if mode == 'missing' else 'answer')
        self.baseline_mode = mode
        self.summary = '当前范围无法确认，保留已有正文和未解决项。'
        if mode == 'missing':
            self.before = '保留开头。在已读范围无法确认方法设置。保留结尾。'
        if mode == 'correct':
            self.before = self.after

    def handler(self, request):
        body = json.loads(request.content)
        assert 'LABEL_FILE_SENTINEL' not in json.dumps(body, ensure_ascii=False)
        if self.baseline_mode == 'unknown':
            self.requests.append(body)
            raise TimeoutError('synthetic sent request with no response')
        if self.baseline_mode == 'not_sent':
            self.requests.append(body)
            raise RequestNotSent('synthetic local validation rejected before send')
        response = super().handler(request)
        call = response.json()['choices'][0]['message']['tool_calls'][0]['function']
        args = json.loads(call['arguments'])
        if self.baseline_mode == 'correct' and call['name'] == 'revise_research':
            observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
            current = next(o['data']['target'] for o in observations if o.get('data', {}).get('review_id'))
            return self.response('finish_task', {'reflection_target_id': current['target_id'],
                'reflection_content_hash': current['content_hash'],
                'goals': [{'goal_id': 'method', 'status': 'satisfied', 'reason': '当前稿已有原文支持'}]})
        if self.baseline_mode == 'rejected' and call['name'] == 'revise_research':
            args['content_hash'] = '0'*64
            return self.response('revise_research', args)
        if self.baseline_mode == 'rejected' and call['name'] == 'finish_task':
            observations = [json.loads(m['content']) for m in body['messages'] if m['role'] == 'tool']
            reviews = [o['data'] for o in observations if o.get('data', {}).get('review_id')]
            if reviews:
                current = reviews[-1]['target']
                return self.response('finish_task', {'reflection_target_id': current['target_id'],
                    'reflection_content_hash': current['content_hash'],
                    'processing_summary': '局部修订的版本身份被拒绝，正文保持原稿；问题未修复。',
                    'goals': [{'goal_id': 'method', 'status': 'unresolved', 'reason': '修改被拒绝'}]})
        if self.baseline_mode == 'missing' and call['name'] == 'finish_task':
            args['processing_summary'] = self.summary
            return self.response('finish_task', args)
        return response


class UnsupportedValueScenario(BaselineScenario):
    """原稿含无依据的显存数值，模拟模型将其撤回并限定查证范围。"""

    def __init__(self):
        super().__init__()
        self.before = '方法使用语言前缀。峰值显存为8 GB。方法使用嵌入表示。'
        self.disclosure = '所给物理第1—2页文字未提供显存用量依据，撤回原精确数值。'
        self.after = '方法使用语言前缀。' + self.disclosure + '方法使用嵌入表示。'

    def handler(self, request):
        body = json.loads(request.content)
        response = super().handler(request)
        call = response.json()['choices'][0]['message']['tool_calls'][0]['function']
        args = json.loads(call['arguments'])
        if call['name'] == 'review_research':
            args['verification_questions'] = ['所给页是否支持显存说法及查证范围限定？']
        elif call['name'] == 'finish_review':
            target = json.loads(body['messages'][1]['content'].split('\n', 1)[1])['target']
            item = target['items'][0]
            corrected = self.disclosure in item['text']
            args['findings'] = [{'item_id': item['item_id'],
                'status': 'supported' if corrected else 'insufficient',
                'basis': 'scope_disclosure' if corrected else 'not_found',
                'source_quote': None,
                'evidence_ids': args['findings'][0]['evidence_ids'],
                'reason': '披露仅限定实际所给两页文字，未推断全篇。' if corrected else
                          '所给两页未提供显存数值依据，不能编造替代数值。',
                'suggested_action': 'keep' if corrected else 'revise'}]
        elif call['name'] == 'revise_research':
            args['patches'][0].update(old_text='峰值显存为8 GB。', new_text=self.disclosure,
                reason='撤回未获给定证据支持的精确数值，保留其他正文。')
        elif call['name'] == 'finish_task':
            args['processing_summary'] = '已撤回无依据的显存数值；当前仅能披露所给页的证据范围。'
            args['goals'][0].update(status='unresolved', reason='给定页不足以确认实际显存值，已有内容已交付。')
        return self.response(call['name'], args)


class ReflectionBaselineRuntimeTests(unittest.TestCase):
    """每例使用全新目录、SQLite、无网络替身和正式持久结果。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = Path(__file__).resolve().parents[3]
        self.before_sources = business_source_hashes(self.repo)
        self.addCleanup(lambda: self.assertEqual(business_source_hashes(self.repo), self.before_sources))

    def export_evidence(self, run):
        """仅在显式资格输出目录留存合成轨迹，避免用测试摘要代替原件。"""
        selected = os.getenv('REFLECTION_BASELINE_QUALIFICATION_OUTPUT')
        if selected:
            directory = Path(selected).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            with (directory/(self._testMethodName + '.json')).open('x') as handle:
                json.dump(run.report(), handle, ensure_ascii=False, indent=2)

    def execute(self, mode='revision', scenario=None):
        scenario = scenario or BaselineScenario(mode)
        with patch('research_loop.config.Settings.from_env', side_effect=AssertionError('dotenv not allowed')):
            run = run_substitute(self.root/'runtime', prompt='核验给定当前稿，必要时局部修订，禁止保存。',
                provider_factory=scenario.factory, pdf_factory=lambda settings, files: ScopedPDF())
        self.addCleanup(self.export_evidence, run)
        self.assertEqual(business_source_hashes(self.repo), self.before_sources)
        self.assertEqual(run.resource_boundary['retrieval'], 'explicitly_disabled_for_substitute_only')
        self.assertFalse(run.db.fetchall('SELECT * FROM save_receipts'))
        self.assertFalse(list((run.root/'output').rglob('*.md')))
        self.assertFalse(run.db.fetchall('SELECT * FROM drafts'))
        return run, scenario

    def test_no_action_baseline_never_constructs_provider_or_directory(self):
        with patch.object(DeepSeekProvider, '__init__', side_effect=AssertionError('provider constructed')):
            result = baseline_result('固定当前稿。')
        self.assertEqual(result['answer_body_markdown'], '固定当前稿。')
        self.assertEqual(result['physical_requests'], 0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_native_review_model_patch_final_delivery_and_restore(self):
        run, scenario = self.execute()
        view = run.view()
        self.assertEqual(view['status'], 'completed', view)
        self.assertEqual(view['result']['answer_body_markdown'], scenario.after)
        snapshot = ReflectionService(run.db).snapshot(run.task_id)
        target = next(iter(snapshot['research_targets'].values()))
        self.assertEqual(len(target['patches']), 1)
        self.assertEqual([a['result']['status'] for a in target['attempts']], ['issues', 'supported'])
        actions = run.db.list_actions(run.task_id)
        modified = [a for a in actions if a['tool_name'] == 'revise_research']
        self.assertEqual(len(modified), 1)
        self.assertEqual(modified[0]['status'], 'succeeded')
        patch_text = modified[0]['arguments']['patches'][0]['new_text']
        # 补丁来自被持久化的模型工具调用，不读取标签文件或传入预填patch。
        raw_calls = run.report()['physical_receipts']
        model_patches = []
        for row in raw_calls:
            response = json.loads(row['response_json']) if row['response_json'] else {}
            for choice in response.get('choices', []):
                for call in choice['message'].get('tool_calls', []):
                    if call['function']['name'] == 'revise_research':
                        model_patches.extend(json.loads(call['function']['arguments'])['patches'])
        self.assertTrue(any(item['new_text'] == patch_text for item in model_patches))
        # 模拟合法重新入队，只改本例 SQLite；业务 checkpoint/成功收据保持原件。
        run.db.update_task(run.task_id, status='queued', result_json=None)
        with patch('pathlib.Path.read_text', side_effect=AssertionError('label read forbidden')):
            restored = run.resume()
        self.assertEqual(restored['result'], view['result'])
        self.assertEqual(len(scenario.requests), len(raw_calls))
        self.assertEqual(run.budget.snapshot()['pending_reservations'], 0)

    def test_already_correct_review_preserves_body_without_patch(self):
        run, scenario = self.execute('correct')
        view = run.view()
        self.assertEqual(view['status'], 'completed')
        self.assertEqual(view['result']['answer_body_markdown'], scenario.before)
        self.assertFalse([a for a in run.db.list_actions(run.task_id) if a['tool_name'] == 'revise_research'])

    def test_rejected_revision_keeps_original_body_and_discloses_failure(self):
        run, scenario = self.execute('rejected')
        view = run.view()
        self.assertEqual(view['status'], 'partial', view)
        self.assertEqual(view['result']['answer_body_markdown'], scenario.before)
        self.assertIn('被拒绝', view['result']['processing_summary'])
        patches = [a for a in run.db.list_actions(run.task_id) if a['tool_name'] == 'revise_research']
        self.assertEqual([a['status'] for a in patches], ['failed'])
        self.assertFalse(next(iter(ReflectionService(run.db).snapshot(run.task_id)['research_targets'].values()))['patches'])

    def test_missing_evidence_is_partial_with_scope_and_no_fake_patch(self):
        run, scenario = self.execute('missing')
        view = run.view()
        self.assertEqual(view['status'], 'partial')
        self.assertEqual(view['result']['answer_body_markdown'], scenario.before)
        self.assertEqual(view['result']['processing_summary'], scenario.summary)
        self.assertFalse([a for a in run.db.list_actions(run.task_id) if a['tool_name'] == 'revise_research'])
        self.assertEqual(view['result']['research_reviews'][0]['quality_status'], 'issues')

    def test_unknown_receipt_blocks_resend_and_keeps_budget_reservation(self):
        run, scenario = self.execute('unknown')
        self.assertEqual(len(scenario.requests), 1)
        rows = run.report()['physical_receipts']
        self.assertEqual([r['status'] for r in rows], ['unknown'])
        run.resume()
        self.assertEqual(len(scenario.requests), 1)
        self.assertEqual(run.budget.snapshot()['pending_reservations'], 1)
        self.assertEqual(run.budget.snapshot()['stop_reason'], 'unknown_call')
        with self.assertRaises(BudgetRefused):
            run.budget.reserve(1, 1)

    def test_explicit_not_sent_receipt_releases_reservation_and_stops_batch(self):
        run, scenario = self.execute('not_sent')
        rows = run.report()['physical_receipts']
        self.assertEqual(len(scenario.requests), 1)
        self.assertEqual([item['status'] for item in rows], ['not_sent'])
        self.assertEqual([item['sent_state'] for item in rows], ['not_sent'])
        snapshot = run.budget.snapshot()
        self.assertEqual(snapshot['not_sent_calls'], 1)
        self.assertEqual(snapshot['pending_reservations'], 0)
        self.assertEqual(snapshot['requests_reserved'], 0)
        self.assertEqual(snapshot['stop_reason'], 'provider_not_sent')
        self.assertIn('before send', snapshot['not_sent_reasons'][0])
        with self.assertRaises(BudgetRefused):
            run.budget.reserve(1, 1)

    def test_unsupported_value_is_withdrawn_by_model_patch_and_persisted(self):
        run, scenario = self.execute(scenario=UnsupportedValueScenario())
        view = run.view()
        self.assertEqual(view['status'], 'partial', view)
        self.assertEqual(view['result']['answer_body_markdown'], scenario.after)
        self.assertNotIn('峰值显存为8 GB', view['result']['answer_body_markdown'])
        self.assertTrue(view['result']['answer_body_markdown'].startswith('方法使用语言前缀。'))
        self.assertTrue(view['result']['answer_body_markdown'].endswith('方法使用嵌入表示。'))
        target = next(iter(ReflectionService(run.db).snapshot(run.task_id)['research_targets'].values()))
        self.assertEqual(len(target['patches']), 1)
        self.assertEqual([a['result']['status'] for a in target['attempts']], ['issues', 'supported'])
        self.assertEqual(target['attempts'][0]['result']['findings'][0]['status'], 'insufficient')
        self.assertEqual(target['attempts'][-1]['result']['findings'][0]['basis'], 'scope_disclosure')
        patches = [a for a in run.db.list_actions(run.task_id) if a['tool_name'] == 'revise_research']
        self.assertEqual([a['status'] for a in patches], ['succeeded'])
        self.assertEqual(patches[0]['arguments']['patches'][0]['new_text'], scenario.disclosure)

    def test_wrong_hash_and_foreign_source_are_rejected_before_callback(self):
        run, _ = self.execute()
        record = next(iter(ReflectionService(run.db).snapshot(run.task_id)['research_targets'].values()))
        current = record['current']
        service = ReflectionService(run.db)
        with self.assertRaisesRegex(ValueError, 'stale'):
            service.research_target(run.task_id, current['target_id'], '0'*64)
        with self.assertRaises((PermissionError, ValueError)):
            service.review_research(run.task_id, ResearchReviewInput(target_type='answer',
                answer_markdown='另一待核验稿。', reason='检查来源', verification_questions=['是否有据？'],
                evidence_ids=['foreign-evidence']), lambda *args: self.fail('callback must not execute'))

    def test_true_execution_is_rejected_before_creating_resources(self):
        with self.assertRaisesRegex(PermissionError, 'REAL_EXECUTION_NOT_AUTHORIZED'):
            run_substitute(self.root/'forbidden', prompt='unused', provider_factory=lambda settings: None,
                           pdf_factory=lambda settings, files: None, execution_kind='real_model')
        self.assertFalse((self.root/'forbidden').exists())

    def test_socket_guard_denies_even_an_accidental_real_factory(self):
        with deny_socket_connections(), self.assertRaisesRegex(PermissionError, 'NETWORK_DENIED'):
            socket.create_connection(('127.0.0.1', 33316))

    def test_runtime_rejects_any_checkout_marker_before_creating_directories(self):
        for kind in ('directory', 'worktree_file'):
            with self.subTest(kind=kind):
                checkout = self.root/kind
                checkout.mkdir()
                if kind == 'directory':
                    (checkout/'.git').mkdir()
                else:
                    (checkout/'.git').write_text('gitdir: ../metadata\n')
                before = {str(path.relative_to(checkout)) for path in checkout.rglob('*')}
                target = checkout/'missing-parent'/'runtime'
                with self.assertRaisesRegex(ValueError, 'outside every Git checkout'):
                    run_substitute(target, prompt='unused',
                        provider_factory=lambda settings: self.fail('provider constructed'),
                        pdf_factory=lambda settings, files: self.fail('PDF constructed'))
                self.assertEqual({str(path.relative_to(checkout)) for path in checkout.rglob('*')}, before)

    def test_runtime_rejects_actual_worktree_and_symlink_alias_before_writing(self):
        name = '_reflection_forbidden_' + uuid4().hex
        alias = self.root/'source-alias'
        alias.symlink_to(self.repo, target_is_directory=True)
        for target in (self.repo/name, alias/name):
            with self.subTest(target=str(target)):
                with self.assertRaisesRegex(ValueError, 'outside every Git checkout'):
                    run_substitute(target, prompt='unused', provider_factory=lambda settings: None,
                                   pdf_factory=lambda settings, files: None)
                self.assertFalse(target.exists())


class ReflectionBatchBudgetTests(unittest.TestCase):
    """验证批次发送前预留、结算、拒绝和未结算停止，不改产品额度。"""

    @staticmethod
    def usage(prompt=10, completion=5):
        return {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion}

    def test_default_trial_limits_and_reservation_release(self):
        budget = BatchBudget()
        identity = budget.reserve(1000, 65536)
        budget.settle(identity, self.usage())
        budget.settle(identity, self.usage())
        snapshot = budget.snapshot()
        self.assertEqual(snapshot['limits'], {'requests': 48, 'tokens': 2_000_000, 'cost_cny': '16'})
        self.assertEqual(snapshot['known_tokens'], 15)
        self.assertEqual(snapshot['requests_reserved'], 1)
        self.assertEqual(snapshot['pending_reservations'], 0)

    def test_request_token_and_money_refuse_before_send(self):
        for budget, reservation, expected in [
                (BatchBudget(max_requests=1), (1, 1), 'request_limit'),
                (BatchBudget(max_tokens=10), (6, 5), 'token_reservation_limit'),
                (BatchBudget(max_cost_cny=Decimal('0.000001')), (1, 1), 'money_reservation_limit')]:
            with self.subTest(expected=expected):
                if expected == 'request_limit':
                    identity = budget.reserve(1, 1)
                    budget.settle(identity, self.usage(1, 1))
                with self.assertRaisesRegex(BudgetRefused, expected):
                    budget.reserve(*reservation)

    def test_unknown_missing_usage_and_known_failure_preserve_stop(self):
        for status, usage, expected in [('unknown', None, 'unknown_call'),
                ('rejected', None, 'usage_missing'), ('failed', self.usage(), 'provider_failed')]:
            with self.subTest(status=status):
                budget = BatchBudget()
                identity = budget.reserve(100, 100)
                budget.settle(identity, usage, status=status)
                snapshot = budget.snapshot()
                self.assertEqual(snapshot['stop_reason'], expected)
                self.assertEqual(snapshot['pending_reservations'], 0 if usage else 1)
                self.assertEqual(snapshot['known_tokens'], 15 if usage else 0)
                with self.assertRaises(BudgetRefused):
                    budget.reserve(1, 1)

    def test_invalid_usage_and_overrun_stop_without_hiding_actual_usage(self):
        budget = BatchBudget()
        identity = budget.reserve(1, 1)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            budget.settle(identity, {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 3})
        self.assertEqual(budget.snapshot()['pending_reservations'], 1)
        self.assertEqual(budget.snapshot()['stop_reason'], 'usage_invalid')
        budget = BatchBudget()
        identity = budget.reserve(1, 1)
        with self.assertRaisesRegex(BudgetRefused, 'upper_bound_breached'):
            budget.settle(identity, self.usage(2, 1))
        self.assertEqual(budget.snapshot()['known_tokens'], 3)

    def test_explicit_not_sent_is_idempotent_without_zero_usage_settlement(self):
        budget = BatchBudget()
        identity = budget.reserve(1000, 65536)
        reason = 'local validation refused before network send'
        budget.settle_not_sent(identity, reason)
        budget.settle_not_sent(identity, reason)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot['requests_reserved'], 0)
        self.assertEqual(snapshot['not_sent_calls'], 1)
        self.assertEqual(snapshot['pending_reservations'], 0)
        self.assertEqual(snapshot['not_sent_reasons'], [reason])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            budget.settle(identity, self.usage(0, 0))
        with self.assertRaisesRegex(BudgetRefused, 'provider_not_sent'):
            budget.reserve(1, 1)


if __name__ == '__main__':
    unittest.main()
