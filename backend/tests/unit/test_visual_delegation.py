"""视觉日常委派的参数、角色权限与紧凑观察测试，不发送模型请求。"""
import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from research_loop.domain.tasks import SavePolicy
from research_loop.domain.visual_reading import VisualDelegationInput, compact_observation
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry


def arguments(**updates):
    """提供不含模型自报hash/路径的委派参数。"""
    return {'evidence_id': 'evidence:fixture', 'question': '式(2)使用什么符号？',
            'reason': '文字中的运算符损坏', 'trigger': 'damaged_formula', **updates}


class VisualDelegationUnitTests(unittest.TestCase):
    """防止 schema 与执行权限脱节，完整原件不因观察分页丢失。"""

    def test_schema_and_execution_only_allow_supervisor(self):
        """Researcher/Reviewer即使构造同名调用也不能绕过Supervisor选择。"""
        registry = ToolRegistry(specs())
        dispatch = Mock(return_value={'status': 'complete'})
        for role in ('researcher', 'reviewer'):
            self.assertNotIn('delegate_visual_reading', {s['function']['name'] for s in registry.schemas(role)})
            result = registry.execute('delegate_visual_reading', arguments(), ToolContext(
                task_id='task', save_policy=SavePolicy.deny, role=role, services={'visual_reading': dispatch}))
            self.assertFalse(result.ok)
        dispatch.assert_not_called()
        schema = registry.get('delegate_visual_reading').schema()['function']['parameters']
        self.assertNotIn('pdf_sha256', schema['properties'])
        self.assertNotIn('path', schema['properties'])
        self.assertIn('delegate_visual_reading', {s['function']['name'] for s in registry.schemas('supervisor')})

    def test_trigger_bbox_and_injected_paths_reject(self):
        """只接受两类声明的缺口，拒绝越页区域及任意文件/配置输入。"""
        for changed in ({'trigger': 'read_every_page'}, {'bbox': [0, 0, 2, 1]},
                        {'bbox': [0, 0, float('nan'), 1]}, {'offset': -1}, {'offset': True},
                        {'path': '/private/file'}, {'pdf_sha256': 'a' * 64}):
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                VisualDelegationInput.model_validate(arguments(**changed))

    def test_dispatch_handler_does_not_call_provider(self):
        """处理函数只委派节点并接收观察；未配置节点时返回明确能力缺口。"""
        registry = ToolRegistry(specs())
        ctx = ToolContext(task_id='task', save_policy=SavePolicy.deny)
        missing = registry.execute('delegate_visual_reading', arguments(), ctx)
        self.assertFalse(missing.ok)
        self.assertFalse(missing.retryable)
        dispatch = Mock(return_value={'status': 'partial', 'kind': 'model_observation', 'answer_excerpt': '仍有不确定性'})
        ctx.services['visual_reading'] = dispatch
        result = registry.execute('delegate_visual_reading', arguments(), ctx)
        self.assertTrue(result.ok)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(dispatch.call_args.args[0], 'task')

    def test_compact_observation_is_losslessly_pageable_without_images(self):
        """父角色按原参数读回完整答案，不复制图片、页全文或本机路径。"""
        text = '甲' * 7000
        result = {'status': 'complete', 'answer': text, 'request_id': 'fixture', 'usage': {'total_tokens': 3},
                  'source': {'physical_page': 2, 'text': '整页正文不得回传',
                             'assets': [{'path': '/private/cache.png', 'sha256': 'a' * 64,
                                         'kind': 'page', 'bytes': 'data:image/png;base64,fixture'}]}}
        args = VisualDelegationInput.model_validate(arguments())
        first = compact_observation(result, args, 'child', False)
        second = compact_observation(result, VisualDelegationInput.model_validate(first['readback']['arguments']), 'child', True)
        self.assertEqual(first['answer_excerpt'] + second['answer_excerpt'], text)
        self.assertEqual(first['next_offset'], 6000)
        self.assertIsNone(second['next_offset'])
        self.assertEqual(first['request_id'], second['request_id'])
        self.assertFalse(first['semantically_verified'])
        self.assertNotIn('/private', json.dumps(first))
        self.assertNotIn('base64', json.dumps(first))
        self.assertNotIn('整页正文', json.dumps(first))
        with self.assertRaises(ValueError):
            compact_observation(result, VisualDelegationInput.model_validate(arguments(offset=9000)), 'child', True)


if __name__ == '__main__':
    unittest.main()
