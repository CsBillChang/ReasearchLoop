"""接续实验的默认合同与容量边界，不访问模型或数据库服务。"""
from __future__ import annotations

import hashlib
import json
import unittest

from pydantic import ValidationError

from research_loop.domain.continuation_memory import ContinuationCandidate, ContinuationFinishTaskInput
from research_loop.services.continuation_memory import record_identity
from research_loop.tools.delegation import specs


class ContinuationContractTests(unittest.TestCase):
    """保护默认声明及来源身份，避免实验改变原有交付接口。"""

    def test_disabled_delegation_matches_baseline_except_explicit_guidance(self):
        """仅归一已审查的引句/零论文说明，其余描述和参数保持866c193基线。"""
        schemas = [item.schema() for item in specs()]
        plan = next(item['function'] for item in schemas if item['function']['name'] == 'set_research_plan')
        routing = ('For self-contained discussion, calculation or comparison using only current user-provided data '
                   'or conversation history, with no paper material, deep-reading or file-save deliverable, '
                   'use task_kind=memory and min_papers=0. Paper research uses search/compare/survey and needs '
                   'a user/legacy material goal; existing papers count, so no new search does not imply memory. ')
        self.assertTrue(plan['description'].startswith(routing))
        self.assertEqual(plan['description'].count(routing), 1)
        plan['description'] = plan['description'].removeprefix(routing)
        old = ('Put actual user hard sources/date constraints on the corresponding origin=user goals '
               'with verbatim source_quote; top-level sources/date')
        new = ('Put actual user hard sources/date constraints on the corresponding origin=user goals '
               'with verbatim source_quote. Copy one short contiguous span from the actual user request '
               'or resumed answer; do not paraphrase, add punctuation, or join separated spans with ellipses. '
               'Split separated requirements into goals with their own exact quotes. Top-level sources/date')
        self.assertEqual(plan['description'].count(new), 1)
        plan['description'] = plan['description'].replace(new, old, 1)
        finish = next(item['function'] for item in schemas if item['function']['name'] == 'finish_task')
        operation = (' A memory plan requires a successful memory read or operation before finishing; '
                     'use available memory tools as needed and never claim nonexistent history.')
        self.assertTrue(finish['description'].endswith(operation))
        self.assertEqual(finish['description'].count(operation), 1)
        finish['description'] = finish['description'].removesuffix(operation)
        payload = json.dumps(schemas, sort_keys=True, ensure_ascii=False)
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(),
                         'fce1914bad533e5719a526b56b23d43094248f6edeeee8e4fbd92a295bcea3c4')

    def test_experiment_changes_only_finish_schema(self):
        """实验参数与实际执行共用ToolSpec，其他角色工具声明保持一致。"""
        before = {item.name: item for item in specs()}
        after = {item.name: item for item in specs(continuation_memory=True)}
        self.assertEqual(before.keys(), after.keys())
        for name in before.keys() - {'finish_task'}:
            self.assertEqual(before[name].schema(), after[name].schema())
        self.assertIs(before['finish_task'].handler, after['finish_task'].handler)
        self.assertIn('continuation_memory', after['finish_task'].schema()['function']['parameters']['properties'])

    def test_bad_optional_memory_does_not_break_finish_validation(self):
        """旁路原件留到终态独立拒绝，不能破坏有效研究正文。"""
        value = ContinuationFinishTaskInput.model_validate({'answer_markdown': '合成回答',
            'goals': [{'goal_id': 'one', 'status': 'satisfied', 'reason': '已交付'}],
            'continuation_memory': '坏候选'})
        self.assertEqual(value.answer_markdown, '合成回答')
        self.assertEqual(value.continuation_memory, '坏候选')

    def test_derived_note_does_not_rewrite_original_record_identity(self):
        """投影恢复可追加摘要；原回答被改写则必须使既有来源身份失效。"""
        record = {'task_id': 'synthetic', 'prompt': '问题', 'answer': '回答', 'research_progress': []}
        identity = record_identity(record)
        self.assertEqual(identity, record_identity({**record, 'continuation_memory': {'status': 'applied'}}))
        self.assertNotEqual(identity, record_identity({**record, 'answer': '被改写的回答'}))

    def test_candidate_cannot_claim_verified_or_saved_status(self):
        """模型输入不能添加核验通过或保存授权字段。"""
        for key in ('verification_status', 'save_policy'):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                ContinuationCandidate.model_validate({'research_goal': '合成目标', key: 'approved'})

    def test_correction_requires_hash_and_known_relation_type(self):
        """拒绝不带身份或伪造自动批准语义的更正。"""
        base = {'previous_task_id': 'old', 'previous_key': 'old',
                'previous_content_hash': 'a' * 64, 'replacement_key': 'new',
                'relation': 'corrects', 'reason': '合成更正'}
        for override in ({'previous_content_hash': ''}, {'relation': 'verified'}, {'replacement_key': ''}):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                ContinuationCandidate.model_validate({'research_goal': '合成目标',
                                                       'corrections': [{**base, **override}]})
