"""修订观察的正文去重合同；只做精确同版本引用，不生成摘要或核验判断。"""
from copy import deepcopy
import json
import unittest

from research_loop.domain.context import project_review_answer
from research_loop.storage.memory import dump


class RevisionAnswerProjectionTests(unittest.TestCase):
    """当前正文、历史版本和真实调用结构分别保持。"""

    def sample(self):
        """构造有版本和来源的单一answer目标与配对修订观察。"""
        text = '保留开头。已根据原文修订。保留结尾。'
        answer = {'text': text, 'target_id': 'target', 'content_hash': 'v2', 'source_id': 'message'}
        context = {'current_task': {'task_id': 'task', 'save_policy': 'deny',
            'current_review_answer': answer, 'research_reviews': [{'content_hash': 'v1',
                'current_version_reviewed': False, 'findings': [{'status': 'contradicted', 'reason': '旧版意见'}]}]}}
        target = {'target_type': 'answer', 'source_task_id': 'task', 'target_id': 'target',
            'content_hash': 'v2', 'items': [{'item_id': 'answer', 'text': text,
                'sources': [{'evidence_id': 'page', 'source_sha256': 'pdf-v1'}]}]}
        args = dump({'target_id': 'target', 'content_hash': 'v1', 'patches': [{'new_text': text}]})
        messages = [{'role': 'assistant', 'content': '', 'reasoning_content': '真实推理保持',
            'tool_calls': [{'id': 'revision', 'type': 'function',
                'function': {'name': 'revise_research', 'arguments': args}}]},
            {'role': 'tool', 'tool_call_id': 'revision', 'content': dump({'ok': True, 'kind': 'success',
                'data': {'target': target, 'quality_status': 'unreviewed',
                         'revision': {'before_hash': 'v1', 'after_hash': 'v2', 'patches': [{'new_text': text}]}}})}]
        return text, messages, context

    def test_exact_revision_links_only_body_and_keeps_all_other_fields(self):
        """当前正文只保留一份，来源/patch/旧意见/权限和调用参数不改变。"""
        text, messages, context = self.sample()
        before = deepcopy((messages, context))
        output, current, _ = project_review_answer(messages, context)
        value = json.loads(output[1]['content']); original = json.loads(messages[1]['content'])
        item = value['data']['target']['items'][0]
        self.assertNotIn('text', item)
        self.assertEqual(item['text_in'], 'current_task.current_review_answer.text')
        item['text'] = current['current_task']['current_review_answer']['text']; item.pop('text_in')
        self.assertEqual(value, original)
        self.assertEqual(current, context)
        self.assertEqual(output[0], messages[0])
        self.assertEqual((messages, context), before)
        self.assertEqual(current['current_task']['current_review_answer']['text'], text)

    def test_old_foreign_missing_identity_and_different_body_are_not_linked(self):
        """不能仅凭标题、相同正文或一个hash跨对象/版本合并。"""
        changes = [({'content_hash': 'v1'}, None), ({'target_id': 'other'}, None),
                   ({'source_task_id': 'foreign'}, None), ({'target_type': 'findings'}, None),
                   ({'content_hash': None}, None), ({'target_id': None}, None),
                   ({}, '同版标记下却不同的正文。')]
        for change, text in changes:
            with self.subTest(change=change, text=text):
                _, messages, context = self.sample()
                value = json.loads(messages[1]['content']); value['data']['target'].update(change)
                if text is not None: value['data']['target']['items'][0]['text'] = text
                messages[1]['content'] = dump(value)
                output, current, _ = project_review_answer(messages, context)
                self.assertEqual(output, messages); self.assertEqual(current, context)

    def test_mixed_batch_links_current_and_keeps_old_and_other_tools(self):
        """同一批次的新旧修订与原文读取分别处理，不改真实assistant批次。"""
        _, messages, context = self.sample()
        original_call = messages[0]['tool_calls'][0]
        old = deepcopy(messages[1]); old['tool_call_id'] = 'old'
        value = json.loads(old['content']); value['data']['target']['content_hash'] = 'v1'
        old['content'] = dump(value)
        page = {'role': 'tool', 'tool_call_id': 'page', 'content': dump({'ok': True,
            'data': {'pages': [{'text': context['current_task']['current_review_answer']['text']}]}})}
        messages[0]['tool_calls'] += [{**deepcopy(original_call), 'id': 'old'},
            {'id': 'page', 'function': {'name': 'read_paper', 'arguments': '{}'}}]
        messages += [old, page]
        output, _, _ = project_review_answer(messages, context)
        self.assertNotIn('text', json.loads(output[1]['content'])['data']['target']['items'][0])
        self.assertEqual(output[2:], [old, page]); self.assertEqual(output[0], messages[0])

    def test_failed_nonobject_and_unpaired_observations_are_unchanged(self):
        """失败、损坏封装或无匹配工具调用时不把正文当作成功修订。"""
        _, template, context = self.sample()
        values = ['not json', '[]', dump({'ok': False, 'data': json.loads(template[1]['content'])['data']}),
                  dump({'ok': True, 'data': {'target': []}})]
        for value in values:
            with self.subTest(value=value[:50]):
                messages = deepcopy(template); messages[1]['content'] = value
                self.assertEqual(project_review_answer(messages, context)[0], messages)
        messages = deepcopy(template); messages[1]['tool_call_id'] = 'unpaired'
        self.assertEqual(project_review_answer(messages, context)[0], messages)

    def test_missing_current_identity_never_borrows_an_arbitrary_result(self):
        """没有绑定对象或当前版本身份不全时，修订工具仍携带自己的正文。"""
        for field in ['current_review_answer', 'target_id', 'content_hash', 'task_id']:
            with self.subTest(field=field):
                _, messages, context = self.sample()
                if field in {'current_review_answer', 'task_id'}: context['current_task'].pop(field)
                else: context['current_task']['current_review_answer'].pop(field)
                self.assertEqual(project_review_answer(messages, context)[0], messages)

    def test_repeated_projection_and_absent_memory_tool_keep_local_link(self):
        """再次装配不扩回正文，引用只依赖本请求，不隐含read_memory权限。"""
        _, messages, context = self.sample()
        first, current, _ = project_review_answer(messages, context)
        second, again, _ = project_review_answer(first, current)
        self.assertEqual((second, again), (first, current))
        item = json.loads(second[-1]['content'])['data']['target']['items'][0]
        self.assertEqual(item['text_in'], 'current_task.current_review_answer.text')

    def test_non_answer_shape_is_not_partially_folded(self):
        """多项目或错误item身份不是单一回答，不把部分文本指向整份回答。"""
        for multiple in [False, True]:
            _, messages, context = self.sample(); value = json.loads(messages[1]['content'])
            if multiple: value['data']['target']['items'] *= 2
            else: value['data']['target']['items'][0]['item_id'] = 'finding'
            messages[1]['content'] = dump(value)
            self.assertEqual(project_review_answer(messages, context)[0], messages)
