"""整份上下文投影的零模型合同：原件可精确读回，容量不以篡改成果换取。"""
import copy
import json
import unittest

from research_loop.agents.memory import estimate
from research_loop.domain.context import project_research_handoffs
from research_loop.domain.context_workspace import archive_location, project_workspace
from research_loop.providers.deepseek import validate_messages
from research_loop.storage.memory import dump
from unit.test_parent_handoff_projection import handoff, progress


def fixture():
    """生成与论文名称无关的候选、历史和诊断；不是供应商的真实输出。"""
    p = progress()
    messages = handoff(p)
    value = json.loads(messages[-1]['content'])
    value['data']['subtasks'][0]['assessments'] = [{'claim': '未独立核实的候选'*5000, 'verdict': 'insufficient'}]
    messages[-1]['content'] = dump(value)
    context = {'messages': [{'role': 'assistant', 'content': '历史研究内容'*8000, 'excerpted': False}],
               'preferences': [{'content': '用中文', 'kind': 'explicit'}],
               'current_task': {'user_request': '比较不同材料，不保存', 'save_policy': 'deny',
                   'latest_clarification': '只比较方法', 'subtasks': [{'subtask_id': 'child', 'status': 'partial',
                       'stop_reason': '尚缺图表依据', 'research_progress': p}]},
               'recent_research': [{'stage_findings': ['旧结论不能替代原文'*4000]}]}
    archive = {'workspace_context': copy.deepcopy(context), 'tool_results': {'delegate': value}}
    messages, context['current_task'], _ = project_research_handoffs(messages, context['current_task'])
    return messages, context, archive


def header(value):
    """模拟系统头包装，计量和生产使用相同 JSON 序列化。"""
    return {'role': 'system', 'content': dump(value)}


class WorkspaceTests(unittest.TestCase):
    """覆盖任意 Unicode、转义、工具权限、不可压缩输入与不同修订身份。"""

    def test_offsets_are_exact_for_duplicate_and_escaped_values(self):
        """相同值出现多次也定位指定路径，分页读取单位严格为字符。"""
        archive = {'甲': ['same', {'引号"': '同句\n\\尾'}, 'same'], '乙': 'same'}
        raw = dump(archive)
        for path, expected in [(['甲', 0], 'same'), (['甲', 1, '引号"'], '同句\n\\尾'), (['甲', 2], 'same'), (['乙'], 'same')]:
            start, length = archive_location(archive, path)
            self.assertEqual(json.loads(raw[start:start+length]), expected)
        self.assertNotEqual(archive_location(archive, ['甲', 0]), archive_location(archive, ['甲', 2]))
        with self.assertRaises(KeyError): archive_location(archive, ['不存在'])

    def test_whole_request_keeps_current_findings_and_exact_readbacks(self):
        """大历史与大工具叠加时仍保留当前全部发现、来源版本及停止状态。"""
        messages, context, archive = fixture()
        before = copy.deepcopy([messages, context, archive])
        tools = [{'type': 'function', 'function': {'name': 'read_memory'}}]
        out, view, metrics = project_workspace(messages, context, tools, archive, 'snap', '比较方法', 18000, header)
        self.assertLessEqual(estimate({'messages': out[:1]+[header(view)]+out[1:], 'tools': tools}), 18000)
        self.assertEqual(view['current_task'], context['current_task'])
        self.assertEqual(view['preferences'], context['preferences'])
        self.assertEqual([messages, context, archive], before)
        self.assertEqual(out[2], messages[2])
        validate_messages(out)
        child = json.loads(out[-1]['content'])['data']['subtasks'][0]
        self.assertEqual(child['gaps_in'], 'current_task.subtasks[0].research_progress.gaps')
        self.assertEqual(child['status'], 'partial')
        raw = dump(archive)
        def check(value):
            """重建每个省略字段，不只检查存在一个读回提示。"""
            if isinstance(value, dict):
                for key, detail in value.items():
                    if key.endswith('_context_archive'):
                        origin = archive
                        for part in detail['original_path']: origin = origin[part]
                        start = detail['readback']['offset']
                        chunks = [raw[i:min(i+6000, start+detail['value_chars'])] for i in range(start, start+detail['value_chars'], 6000)]
                        self.assertEqual(json.loads(''.join(chunks)), origin)
                    else: check(detail)
            elif isinstance(value, list):
                for item in value: check(item)
        check(view); check(json.loads(out[-1]['content']))
        self.assertGreaterEqual(len(metrics['workspace_externalized']), 2)

    def test_no_readback_tool_no_externalization(self):
        """收尾工具范围受限时不能给出不存在的工具或丢掉大内容。"""
        messages, context, archive = fixture()
        out, view, metrics = project_workspace(messages, context, [], archive, 'snap', '', 1000, header)
        self.assertEqual(view, context)
        self.assertEqual(metrics['workspace_externalized'], [])
        self.assertNotIn('context_archive', dump(json.loads(out[-1]['content'])))
        self.assertGreater(estimate(out), 1000)

    def test_same_wording_different_version_not_linked(self):
        """旧版本的相同缺口不能指向当前版本。"""
        messages, context, archive = fixture()
        data = json.loads(messages[-1]['content'])
        child = data['data']['subtasks'][0]
        child['research_progress'].pop('context_path')
        child['research_progress']['revision'] = 1
        messages[-1]['content'] = dump(data)
        out, _, _ = project_workspace(messages, context, [], archive, 'snap', '', 1000, header)
        self.assertEqual(json.loads(out[-1]['content'])['data']['subtasks'][0]['gaps'], progress()['gaps'])

    def test_mandatory_current_user_constraints_never_shrunk(self):
        """无法容纳必要输入时保留超限事实，不删除权限/参数来伪造通过。"""
        messages, context, archive = fixture()
        context['current_task']['user_request'] = '必须遵守的用户条件'*20000
        context['messages'].append({'role': 'user', 'content': '保留历史用户澄清'*5000})
        out, view, _ = project_workspace(messages, context, [{'function': {'name': 'read_memory'}}], archive, 'snap', '', 1000, header)
        self.assertEqual(view['current_task'], context['current_task'])
        self.assertEqual(view['messages'][-1], context['messages'][-1])
        self.assertGreater(estimate(view), 1000)

    def test_long_finding_only_excerpted_with_raw_readback(self):
        """最后才摘录超长候选正文；身份/支持状态/所有来源保持，全文仍可还原。"""
        p = progress(); p['findings'][0]['text'] = '候选机制与尚未证实的限制。'*6000
        full = {'current_task': {'save_policy': 'deny', 'research_progress': p}}
        archive = {'workspace_context': copy.deepcopy(full), 'tool_results': {}}
        messages = [{'role': 'system', 'content': '规则'}, {'role': 'user', 'content': '比较'}]
        _, full['current_task'], _ = project_research_handoffs(messages, full['current_task'])
        _, view, _ = project_workspace(messages, full, [{'function': {'name': 'read_memory'}}], archive, 'snap', '限制', 12000, header)
        finding = view['current_task']['research_progress']['findings'][0]
        detail = finding.pop('text_context_archive')
        start = detail['readback']['offset']; finding['text'] = json.loads(dump(archive)[start:start+detail['value_chars']])
        self.assertEqual(finding, full['current_task']['research_progress']['findings'][0])
