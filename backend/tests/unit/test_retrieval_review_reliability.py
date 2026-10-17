"""材料覆盖、批次交接与当前核验对象的确定性回归，不调用模型。"""
from copy import deepcopy
import json
import unittest

from research_loop.domain.context import project_review_answer
from research_loop.services.research_progress import evidence_progress, readback_handoff, visible_evidence
from research_loop.storage.memory import digest
from research_loop.workers.executor import research_review_failure


class EvidenceCoverageTests(unittest.TestCase):
    """区分实际新增、当前可见与恢复窗口，覆盖版本/位置/原字词。"""

    def row(self, text='0123456789', start=0, version='v1'):
        """生成同页的精确连续窗口。"""
        return {'document_id': 'doc', 'source_sha256': version, 'page_number': 2,
                'char_start': start, 'text': text}

    def observation(self, *rows):
        """构造纯来源观察，不生成核验判断。"""
        return {'ok': True, 'data': {'pages': list(rows)}}

    def action(self, row, sequence=1, kind='new_evidence'):
        """模拟持久化成功行动，批次身份来自真实请求序号。"""
        value = self.observation(row)
        value['data']['research_increment'] = {'kind': kind, 'request_sequence': sequence}
        return {'action_id': str(sequence), 'status': 'succeeded', 'tool_name': 'read_paper',
                'arguments': {'target_ref': 'input', 'start_page': 2, 'page_count': 1}, 'observation': value}

    def messages(self, observation):
        """使用正式配对消息表示实际可见观察。"""
        return [{'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {'name': 'read_paper'}}]},
                {'role': 'tool', 'tool_call_id': 'c', 'content': json.dumps(observation)}]

    def test_full_page_covers_slice_and_visible_slice_is_repeat(self):
        """整页切片不算新增，当前已可见则准确记为重复。"""
        page = self.row(); actions = [self.action(page)]
        report = evidence_progress(actions, 'retrieve_evidence', {'query': 'new wording'},
                                   self.observation(self.row('3456', 3)), self.messages(self.observation(page)))
        self.assertEqual(report['kind'], 'same_evidence')
        self.assertEqual(report['new_source_count'], 0)
        self.assertEqual(report['already_visible_count'], 1)
        self.assertFalse(report['semantic_verdict'])

    def test_missing_window_and_unknown_visibility_are_distinct(self):
        """未呈现窗口可恢复；没有实际请求就不冒称上下文读回。"""
        actions = [self.action(self.row())]
        obs = self.observation(self.row('3456', 3))
        self.assertEqual(evidence_progress(actions, 'retrieve_evidence', {}, obs, [])['kind'], 'context_readback')
        self.assertEqual(evidence_progress(actions, 'read_paper', {}, obs)['kind'], 'existing_evidence')

    def test_union_overlaps_new_text_version_and_unknown_version(self):
        """精确互补片段可以覆盖；冲突字词和新版本不能借旧内容通过。"""
        actions = [self.action(self.row('01234')), self.action(self.row('456789', 4), 2)]
        self.assertEqual(evidence_progress(actions, 'read_paper', {}, self.observation(self.row()))['new_source_count'], 0)
        for row in [self.row('new text'), self.row(version='v2'), self.row(version=None), self.row('3456', 30)]:
            with self.subTest(row=row):
                self.assertEqual(evidence_progress(actions, 'read_paper', {}, self.observation(row))['kind'], 'new_evidence')
        unknown = self.action(self.row(version=None))
        self.assertEqual(evidence_progress([unknown], 'read_paper', {}, self.observation(self.row('3456', 3, None)))['kind'], 'new_evidence')

    def test_visible_defaults_and_excerpt_spans_keep_exact_coordinates(self):
        """不把带省略标记的拼接文字当连续原文，也不漏公共版本字段。"""
        obs = {'ok': True, 'data': {'hits_defaults': {'document_id': 'doc', 'source_sha256': 'v1'},
            'hits': [{'page_number': 2, 'char_start': 100, 'text': 'ab [省略] yz',
                      'context_spans': [{'start': 0, 'text': 'ab'}, {'start': 8, 'text': 'yz'}]}]}}
        messages = self.messages(obs); messages[0]['tool_calls'][0]['function']['name'] = 'retrieve_evidence'
        spans = visible_evidence(messages)
        self.assertEqual([(s[3], s[4]) for s in spans], [(100, 'ab'), (108, 'yz')])
        self.assertEqual(visible_evidence([{'role': 'user', 'content': json.dumps(obs)}]), [])

    def test_batch_order_cannot_hide_repeat_or_block_new_material(self):
        """重复加既有片段仍交接；同批真实新增页则允许继续，顺序一致。"""
        old = self.action(self.row())
        first = self.action(self.row('234', 2), 2, 'context_readback')
        first['tool_name'] = 'retrieve_evidence'; first['arguments'] = {'query': 'restored'}
        repeated = self.action(self.row('234', 2), 3, 'same_evidence')
        repeated['tool_name'] = 'retrieve_evidence'; repeated['arguments'] = {'query': 'reworded'}
        other = self.action(self.row('789', 7), 3, 'same_evidence'); other['action_id'] = 'other'
        for batch in [[repeated, other], [other, repeated]]:
            self.assertIsNotNone(readback_handoff([old, first] + batch))
        new = self.action({**self.row('new source'), 'page_number': 3}, 3, 'new_evidence')
        for batch in [[repeated, new], [new, repeated]]:
            self.assertIsNone(readback_handoff([old, first] + batch))
        self.assertIsNone(readback_handoff([old, first]))

    def test_regrouped_old_hits_and_versions_use_accumulated_coverage(self):
        """A/B分开读回后组合查询不算进展，相同参数的旧版本也不能替代新版本。"""
        left = self.action(self.row('01234'), 1, 'context_readback')
        right = self.action(self.row('56789', 5), 2, 'context_readback')
        whole = self.action(self.row(), 3, 'context_readback')
        whole.update(tool_name='retrieve_evidence', arguments={'query': 'combined'})
        self.assertIsNotNone(readback_handoff([left, right, whole]))
        changed = self.action(self.row(version='v2'), 2, 'context_readback')
        self.assertIsNone(readback_handoff([left, changed]))


class ReviewAnswerProjectionTests(unittest.TestCase):
    """核验对象是待查回答；折叠只做精确引用，不生成语义摘要。"""

    def sample(self):
        """使用1,929字符合成回答和原有消息读回合同。"""
        text = ('候选回答与条件。'*300)[:1929]
        context = {'messages': [{'message_id': 'm', 'content': text}], 'current_task': {
            'save_policy': 'deny', 'current_review_answer': {'text': text, 'source_id': 'm', 'target_id': 't', 'content_hash': 'v'},
            'research_reviews': [{'target': {'target_id': 't', 'content_hash': 'v', 'items': [{'text': text}]}}]}}
        data = {'source_type': 'message', 'source_id': 'm', 'content': text, 'content_hash': digest(text),
                'offset': 0, 'total_chars': len(text), 'next_offset': None}
        messages = [{'role': 'assistant', 'reasoning_content': '原推理', 'tool_calls': [{'id': 'c', 'function': {'name': 'read_memory', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'c', 'content': json.dumps({'ok': True, 'data': data})}]
        return text, messages, context

    def test_exact_bound_answer_retained_once_and_originals_unchanged(self):
        """正文完整保留且可定位；观察/history/target中的副本引用同一处。"""
        text, messages, context = self.sample(); original = deepcopy((messages, context))
        output, current, metrics = project_review_answer(messages, context)
        self.assertEqual((messages, context), original)
        self.assertEqual(current['current_task']['current_review_answer']['text'], text)
        self.assertEqual(metrics['current_review_answer_chars'], 1929)
        data = json.loads(output[-1]['content'])['data']
        self.assertNotIn('content', data); self.assertEqual(data['next_offset'], None)
        self.assertEqual(data['content_in'], 'current_task.current_review_answer.text')
        self.assertEqual(output[0], messages[0])
        self.assertEqual(current['current_task']['save_policy'], 'deny')

    def test_changed_source_old_version_and_wrong_offset_are_not_linked(self):
        """同内容外部消息、旧版本或游标冲突都不能借当前正文替代。"""
        for change in [{'source_id': 'other'}, {'content_hash': 'wrong'}, {'offset': 1}]:
            text, messages, context = self.sample()
            value = json.loads(messages[-1]['content']); value['data'].update(change)
            messages[-1]['content'] = json.dumps(value)
            output, _, _ = project_review_answer(messages, context)
            self.assertIn('content', json.loads(output[-1]['content'])['data'])

    def test_no_bound_answer_does_not_pin_arbitrary_history(self):
        """没有明确引用时不默认最近回复，也不为相似文字创建身份。"""
        _, messages, context = self.sample(); context['current_task'].pop('current_review_answer')
        output, current, metrics = project_review_answer(messages, context)
        self.assertEqual((output, current), (messages, context))
        self.assertFalse(metrics['current_review_answer_retained'])


class ReviewFailureDiagnosticsTests(unittest.TestCase):
    """历史参数错误、合法恢复和最后提交失败分别保留。"""

    def test_last_blocking_submission_has_priority_without_erasing_first_error(self):
        """与原轨迹相同的20→8→提交失败，不把已纠正首错当唯一根因。"""
        actions = [{'action_id': 'a1', 'sequence_no': 1, 'status': 'failed', 'tool_name': 'retrieve_evidence',
                    'observation': {'kind': 'validation_error', 'message': 'limit20 exceeds8'}},
                   {'action_id': 'a2', 'sequence_no': 2, 'status': 'succeeded', 'tool_name': 'retrieve_evidence', 'observation': {'ok': True}},
                   {'action_id': 'a3', 'sequence_no': 3, 'status': 'failed', 'tool_name': 'finish_review',
                    'observation': {'kind': 'validation_error', 'message': 'findings[7]: invalid execution disclosure'}}]
        before = deepcopy(actions); result = research_review_failure(actions, 'budget exhausted', 'budget exhausted')
        self.assertEqual(result['first_error']['action_id'], 'a1')
        self.assertEqual(result['blocking_error']['action_id'], 'a3')
        self.assertEqual(result['blocking_error']['sequence'], 3)
        self.assertEqual(len(result['tool_errors']), 2)
        self.assertFalse(result['semantic_verdict_available']); self.assertEqual(actions, before)
        actions.append({'action_id': 'a4', 'status': 'succeeded', 'tool_name': 'finish_review'})
        self.assertIsNone(research_review_failure(actions, 'storage interrupted', None)['blocking_error'])
