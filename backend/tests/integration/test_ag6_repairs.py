"""AG6暴露的跨层缺陷回归；真实数据库服务、合成论文/模型材料，不冒充语义验收。"""
import json
import unittest
from integration import test_memory as memory_tests
from integration.test_collaboration_contract import FixturePDF
from research_loop.agents.memory import ContextService
from research_loop.domain.reading import user_page_range
from research_loop.domain.memory import HistoryRead
from research_loop.providers.deepseek import validate_messages
from research_loop.services.papers import PaperService


class AG6RepairTests(unittest.TestCase):
    """覆盖范围隔离、召回容量和完整往返压缩，不改变原始记录。"""
    setUp = memory_tests.MemoryTests.setUp
    task = memory_tests.MemoryTests.task
    done = memory_tests.MemoryTests.done

    def test_explicit_attachment_range_and_ambiguity(self):
        """自然语言附件页范围可读，非连续与冲突仍拒绝，不扩大授权。"""
        for text in ['只根据上传论文的物理第1至2页，报告驱动版本', '仅阅读已上传附件的第 1 — 2 页']:
            scope=user_page_range([text]);self.assertEqual((scope.start,scope.end),(1,2))
        self.assertIsNone(user_page_range(['不要只读前8页，请读全文']))
        with self.assertRaises(ValueError):user_page_range(['只读第1页和第3页'])
        with self.assertRaises(ValueError):user_page_range(['只读前2页，另一篇只读前3页'])

    def test_review_selection_does_not_erase_acquired_scope(self):
        """全文证据已取得，只选一页审校不能改成任务只读一页。"""
        tid=self.task();papers=PaperService(self.db,None,FixturePDF())
        value=papers.read(tid,'input-A',attachments={'input-A':{'local_path':'unused.pdf','paper_id':'paper-A'}})
        refs=[p['evidence_id'] for p in value['pages']]
        selected=papers.evidence_bundle(tid,refs[:1])[0]
        self.assertEqual(selected['read_scope']['included_physical_pages'],[1])
        self.assertFalse(selected['read_scope']['all_physical_pages_included'])
        self.assertEqual(selected['task_evidence_scope']['authorized_physical_pages'],[1,2])
        self.assertTrue(selected['task_evidence_scope']['all_physical_pages_available'])
        self.assertFalse(selected['read_scope']['visual_content_interpreted'])
        with self.assertRaises(PermissionError):papers.evidence_bundle(tid,['foreign-evidence'])

    def test_search_card_bounded_but_original_readable(self):
        """大量候选元数据不进入召回工具回复，原件仍按身份分页取得。"""
        tid=self.task('研究图像中文字');self.done(tid,'已读图像中文字研究')
        self.conv.view(self.cid)
        row=self.db.fetchone('SELECT record_json FROM research_memories WHERE task_id=?',(tid,))
        record=json.loads(row['record_json'])
        record['papers']=[{'paper_id':str(i),'title':'图像中文字','abstract':'原始摘要'*2000} for i in range(100)]
        self.db.execute('UPDATE research_memories SET record_json=? WHERE task_id=?',(json.dumps(record,ensure_ascii=False),tid))
        current=self.task('回顾图像中文字研究')
        hit=next(h for h in self.memory.search('图像中文字','research_history',10,current) if h['source_id']==tid)
        self.assertLess(len(json.dumps(hit,ensure_ascii=False).encode()),12000)
        self.assertTrue(hit['excerpted']);self.assertNotIn('abstract',json.dumps(hit))
        original=self.memory.read(current,HistoryRead(source_type='research_history',source_id=tid,limit=10000))
        self.assertIn('原始摘要',original['content']);self.assertIsNotNone(original['next_offset'])
        self.assertFalse(original['is_original_paper_evidence'])

    def test_large_old_arguments_compact_as_complete_exchange(self):
        """压缩旧长参数及观察；推理字段、最新组/用户指令不改，原件可回读。"""
        tid=self.task('保留成果，不保存文件')
        messages=[{'role':'system','content':'system'},{'role':'user','content':'保留成果，不保存文件'}]
        for i in range(3):
            args={'markdown':'旧草稿'*4000 if i<2 else '最新'}
            messages += [{'role':'assistant','content':'','reasoning_content':'供应商要求原样回传的推理'+str(i),
                'tool_calls':[{'id':str(i),'type':'function','function':{'name':'create_note','arguments':json.dumps(args,ensure_ascii=False)}}]},
                {'role':'tool','tool_call_id':str(i),'content':'已保存数据库草稿'+str(i)}]
        before=json.dumps(messages,ensure_ascii=False)
        settings=self.settings.model_copy(update={'context_window_tokens':18000,'context_output_reserve':1000,'context_safety_tokens':500})
        projected,metrics=ContextService(self.db,self.memory,settings).prepare({'task_id':tid,'prompt':'保留成果，不保存文件'},messages,[])
        validate_messages(projected)
        self.assertLess(metrics['after'],metrics['input_capacity'])
        self.assertEqual(projected[-2:],messages[-2:]);self.assertEqual(projected[2],messages[1])
        self.assertEqual([m['reasoning_content'] for m in projected if m['role']=='assistant'],
                         [m['reasoning_content'] for m in messages if m['role']=='assistant'])
        self.assertEqual(json.dumps(messages,ensure_ascii=False),before)
        archived=self.db.fetchone('SELECT original_json FROM context_archives WHERE snapshot_id=?',(metrics['snapshot_id'],))
        self.assertEqual(json.loads(archived['original_json'])['messages'],messages)
