"""长工具历史工作集：压缩封装同时保留来源、限定与完整消息配对。"""
import copy
import json
import unittest
from research_loop.domain.context import compact_research_exchanges
from research_loop.providers.deepseek import validate_messages


class ContextWorkingSetTests(unittest.TestCase):
    """纯数据测试，不调用模型、不依赖论文ID或固定页码。"""

    def test_old_envelopes_removed_and_current_user_instruction_kept(self):
        """归档完整历史组，当前要求和最后一组调用仍在。"""
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': '对照条件，不保存'}]
        for i in range(20):
            messages += [{'role': 'assistant', 'reasoning_content': 'old long reasoning '*1000,
                'tool_calls': [{'id': str(i), 'function': {'name': 'read_paper', 'arguments': json.dumps({'target_ref': 'input'})}}]},
                {'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps({'ok': True, 'data': {'content_sha256':'a'*64, 'pages': [{
                    'document_id': 'document', 'evidence_id': 'page', 'page_number': 7,
                    'text': 'baseline\n仅统计骨干网络，均值±标准差，不代表所有组件。\n' + 'tail '*1000}]}})}]
        original = copy.deepcopy(messages)
        result, metrics = compact_research_exchanges(messages, '骨干网络 标准差', 32000)
        validate_messages(result)
        self.assertEqual(messages, original)
        self.assertEqual(result[1]['content'], '对照条件，不保存')
        self.assertEqual(result[-2:], original[-2:])
        self.assertEqual(metrics['archived_groups'], 19)
        self.assertLess(len(json.dumps(result)), len(json.dumps(original)) // 5)
        source = json.loads(result[-3]['content'])['sources'][0]
        self.assertEqual(source['text'], original[-1] and json.loads(original[-1]['content'])['data']['pages'][0]['text'][source['char_start']:source['char_end']])
        self.assertIn('均值±标准差', source['text'])
        self.assertEqual(source['source_sha256'],'a'*64)

    def test_each_document_keeps_a_source_and_version_not_merged(self):
        """两篇材料的原文依据不能因为相同页码合并。"""
        messages = [{'role': 'user', 'content': 'compare'}]
        for i in range(3):
            messages += [{'role':'assistant','tool_calls':[{'id':str(i),'function':{'name':'retrieve_evidence','arguments':'{"query":"conditions"}'}}]},
                {'role':'tool','tool_call_id':str(i),'content':json.dumps({'ok':True,'data':{'hits':[{
                    'document_id':'d'+str(i%2),'evidence_id':'p'+str(i%2),'page_number':1,
                    'char_start':100,'text':'conditions and exact source'}]}})}]
        result, _ = compact_research_exchanges(messages, 'conditions', 16000)
        source = next(json.loads(m['content'])['sources'] for m in result if m.get('role')=='user' and m.get('content','').startswith('{'))
        self.assertEqual({r['document_id'] for r in source}, {'d0','d1'})
        self.assertTrue(all(r['char_start']==100 for r in source))


class ComplementaryWindowTests(unittest.TestCase):
    """复现标题页覆盖互补方法页；材料合成，不依赖固定论文身份。"""

    @staticmethod
    def exchange(call, pages, *, start=None):
        """生成有完整配对和来源版本的实际读页观察。"""
        args = {'target_ref': 'input'}
        if start is not None:
            args.update(start_page=start, page_count=len(pages))
        return [{'role':'assistant','tool_calls':[{'id':str(call),'function':{'name':'read_paper','arguments':json.dumps(args)}}]},
                {'role':'tool','tool_call_id':str(call),'content':json.dumps({'ok':True,'data':{'content_sha256':'v1','pages':pages}})}]

    def test_alternating_method_pages_survive_across_rounds(self):
        """后读训练页不能挤掉之前的机制和损失页，标题词重合不应独占窗口。"""
        pages = [{'document_id':'doc','evidence_id':f'e{n}','page_number':n,'text':t} for n,t in [
            (11,'Alpha Beta Gamma Delta title abstract'),(17,'Pruning removes redundant parameters.'),
            (23,'Regularizer protects pretrained knowledge.'),(29,'Connector uses soft sparsity.')]]
        messages=[{'role':'system','content':'system'},{'role':'user','content':'Alpha Beta Gamma Delta mechanism'}]
        messages+=self.exchange(1,pages)
        messages+=self.exchange(2,[pages[1],pages[2]],start=17)
        messages+=self.exchange(3,[pages[3]],start=29)
        result, _ = compact_research_exchanges(messages,'Alpha Beta Gamma Delta mechanism',60000)
        sources=next(json.loads(m['content'])['sources'] for m in result if m.get('content','').startswith('{"archived_tool_groups"'))
        text=' '.join(s['text'] for s in sources)
        self.assertIn('Pruning removes',text)
        self.assertIn('Regularizer protects',text)
        self.assertIn('Connector uses',text)
        self.assertEqual(len({(s['evidence_id'],s['char_start'],s['char_end']) for s in sources}),len(sources))
        validate_messages(result)

    def test_versions_and_current_group_stay_separate(self):
        """同文档不同版本各自保留；最新组不得被拆开。"""
        a={'document_id':'doc','evidence_id':'old','page_number':2,'text':'old method','source_sha256':'old'}
        b={**a,'evidence_id':'new','text':'new method','source_sha256':'new'}
        m=[{'role':'user','content':'compare'}]+self.exchange(1,[a])+self.exchange(2,[b])
        result,_=compact_research_exchanges(m,'method',40000)
        sources=next(json.loads(x['content'])['sources'] for x in result if x.get('content','').startswith('{"archived_tool_groups"'))
        self.assertEqual({x['source_sha256'] for x in sources},{'old','new'})
        self.assertEqual(result[-2:],m[-2:])

    def test_complementary_pages_use_actual_remaining_space(self):
        """剩余空间足够时不被固定三分之一份额截断互补方法页。"""
        pages = [{'document_id': 'doc', 'evidence_id': f'e{i}', 'page_number': i,
                  'text': f'method {i} '+('condition '*500)} for i in range(1, 7)]
        messages = [{'role': 'system', 'content': '规则'}, {'role': 'user', 'content': '比较条件'}]
        messages += self.exchange('one', pages[:3], start=1)
        messages += self.exchange('two', pages[3:], start=4)
        messages += [{'role': 'assistant', 'tool_calls': [{'id': 'last', 'function': {'name': 'other', 'arguments': '{}'}}]},
                     {'role': 'tool', 'tool_call_id': 'last', 'content': json.dumps({'ok': True, 'data': '末次观察'*3000}, ensure_ascii=False)}]
        result, metrics = compact_research_exchanges(messages, '比较条件', 80000)
        self.assertGreater(metrics['working_set_budget'], 80000//3)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False).encode()), 80000)
        sources = next(json.loads(m['content'])['sources'] for m in result if m.get('content','').startswith('{"archived_tool_groups"'))
        self.assertEqual({s['evidence_id']:s['text'] for s in sources}, {p['evidence_id']:p['text'] for p in pages})

    def test_latest_retrieval_metadata_is_compacted_before_source_allocation(self):
        """先精简重复索引配置，不能先裁原文再空出本可利用的容量。"""
        pages = [{'document_id':'doc','evidence_id':f'e{i}','page_number':i,
                  'text': f'condition {i} '+('original '*700)} for i in (1, 2)]
        messages = [{'role':'system','content':'规则'}, {'role':'user','content':'比较'}]
        messages += self.exchange('one', [pages[0]], start=1)
        messages += self.exchange('two', [pages[1]], start=2)
        data = {'hits':[{**pages[0], 'char_start':0, 'char_end':len(pages[0]['text'])}],
                'scope':[{'document_id':'doc', 'pages':[{'id':str(i),'hash':'x'*64} for i in range(600)]}],
                'embedding': {'runtime':'details'*4000}}
        messages += [{'role':'assistant','tool_calls':[{'id':'retrieval','function':{'name':'retrieve_evidence','arguments':'{}'}}]},
                     {'role':'tool','tool_call_id':'retrieval','content':json.dumps({'ok':True,'data':data})}]
        before = copy.deepcopy(messages)
        result, _ = compact_research_exchanges(messages, 'condition', 36000)
        self.assertEqual(messages, before)
        latest = json.loads(result[-1]['content'])['data']
        self.assertEqual(latest['hits'], data['hits'])
        self.assertEqual(latest['scope'][0]['indexed_page_count'], 600)
        self.assertNotIn('embedding', latest)
        self.assertLessEqual(len(json.dumps(result,ensure_ascii=False).encode()),36000)
        sources=next(json.loads(m['content'])['sources'] for m in result if m.get('content','').startswith('{"archived_tool_groups"'))
        self.assertEqual({s['evidence_id']:s['text']for s in sources},{p['evidence_id']:p['text']for p in pages})

    def test_two_documents_prioritize_complementary_batches(self):
        """两篇材料共享容量时，优先完整保留两轮定向方法内容，再放标题页。"""
        documents={}
        for label in ('A','B'):
            documents[label]=[{'document_id':'d'+label,'evidence_id':f'{label}-{n}',
                'page_number':n,'text':('title keyword '*230 if n==11 else f'{label} method {n} '+('condition '*300))}
                for n in (11,17,23,29)]
        messages=[{'role':'user','content':'title keyword comparison'}]
        messages+=self.exchange('all',[p for d in documents.values() for p in d])
        for i,(label,indices) in enumerate([('A',[1,2]),('B',[1,2]),('A',[3]),('B',[3])]):
            messages+=self.exchange(i,[documents[label][n] for n in indices],start=17)
        result,_=compact_research_exchanges(messages,'title keyword comparison',80000)
        sources=next(json.loads(m['content'])['sources'] for m in result if m.get('content','').startswith('{"archived_tool_groups"'))
        by_id={s['evidence_id']:s['text'] for s in sources}
        for pages in documents.values():
            for p in pages[1:]:self.assertEqual(by_id[p['evidence_id']],p['text'])
