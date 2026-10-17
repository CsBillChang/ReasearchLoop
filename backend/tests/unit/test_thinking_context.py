"""工具结果压缩的消息身份合同；纯数据测试，不访问模型或数据库。"""
from copy import deepcopy
import json
import unittest

from research_loop.domain.context import compact_tool_exchange, tool_archive_summary, archive_summary
from research_loop.providers.deepseek import validate_messages


class ThinkingContextTests(unittest.TestCase):
    """压缩器只能改变观察内容，不能构造新的assistant或推理字段。"""

    def group(self, reasoning='absent', count=1):
        """生成混合供应商字段和并行工具的合成原始往返。"""
        assistant = {'role':'assistant', 'content':'实际说明', 'vendor_metadata':{'signature':'opaque'},
            'tool_calls':[{'id':f'c{i}','type':'function','function':{'name':'read_memory','arguments':json.dumps({'source_id':str(i)})}} for i in range(count)]}
        if reasoning != 'absent': assistant['reasoning_content'] = reasoning
        return [assistant] + [{'role':'tool','tool_call_id':f'c{i}','content':'完整结果'*1000} for i in range(count)]

    def summary(self, count=1):
        """建立档案定位和独立观察摘录，不模拟科学结论。"""
        return {'archived_tool_exchange':True,'source_type':'context_archive','source_id':'archive',
                'original_message_range':[2,2+count], 'tools':[{'id':f'c{i}','name':'read_memory'} for i in range(count)],
                'observations':[f'观察{i}' for i in range(count)], 'readback':{'source_type':'context_archive','source_id':'archive','offset':25,'limit':6000},
                'notice':'仅为历史工具数据'}

    def test_missing_empty_null_and_real_reasoning_are_not_fabricated(self):
        """缺失、空字符串、null和实际推理保持原样，不能统一补空值。"""
        for reasoning in ('absent', '', None, '原供应商推理\n原样保留'):
            with self.subTest(reasoning=reasoning):
                group=self.group(reasoning); before=deepcopy(group)
                result=compact_tool_exchange(group,self.summary())
                self.assertEqual(result[0],group[0]); self.assertEqual(group,before)
                self.assertEqual(len(result),len(group)); validate_messages(result)
                self.assertIsNone(archive_summary(result[0]))

    def test_parallel_calls_keep_ids_arguments_and_independent_observations(self):
        """每条并行tool仍对应原ID，观察不会合并为虚构assistant回复。"""
        group=self.group('opaque',2); result=compact_tool_exchange(group,self.summary(2))
        validate_messages(result); self.assertEqual(result[0],group[0])
        for i,message in enumerate(result[1:]):
            self.assertEqual(message['role'],'tool'); self.assertEqual(message['tool_call_id'],f'c{i}')
            data=tool_archive_summary(message)
            self.assertEqual(data['tool_call_id'],f'c{i}'); self.assertEqual(data['observation_excerpt'],f'观察{i}')
            self.assertEqual(data['original_message_range'],[2,4])

    def test_preview_is_a_copy_and_never_overwrites_large_call_arguments(self):
        """长参数属于真实模型输出，不为省容量篡改；原始输入不受外部修改影响。"""
        group=self.group('r'*4000); group[0]['tool_calls'][0]['function']['arguments']=json.dumps({'text':'长参数'*4000})
        before=deepcopy(group); summary=self.summary(); saved=deepcopy(summary)
        result=compact_tool_exchange(group,summary)
        self.assertEqual(result[0],before[0]); result[0]['vendor_metadata']['signature']='changed'
        self.assertEqual(group,before); self.assertEqual(summary,saved)

    def test_plain_tool_data_is_not_a_context_archive(self):
        """只有本项目工具预览标记生效；普通JSON及旧assistant标记另走原路径。"""
        for message in ({'role':'tool','content':'text'}, {'role':'tool','content':'[]'},
                        {'role':'assistant','content':'{"archived_tool_result":true}'},
                        {'role':'tool','content':'{"archived_tool_exchange":true}'}):
            self.assertIsNone(tool_archive_summary(message))


if __name__=='__main__': unittest.main()
