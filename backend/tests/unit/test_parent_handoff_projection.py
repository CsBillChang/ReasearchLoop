"""父任务成果投影合同：正文、版本、未核验状态与消息配对均不丢。"""
import copy
import json
import unittest

from research_loop.domain.context import project_research_handoffs
from research_loop.providers.deepseek import validate_messages


def progress(task='child', revision=2):
    """生成重复来源但不同版本/核验状态的公开合成成果。"""
    a = {'evidence_id':'e1','document_id':'doc1','paper_id':'p1','source_sha256':'a'*64,
         'text_sha256':'b'*64,'page_number':3}
    b = {**a, 'source_sha256':'c'*64}
    return {'task_id':task,'revision':revision,'action_id':'action','goal_ids':['g'],
            'objective':'比较方法','findings':[
                {'finding_id':'f1','text':'保留方法与限定。','sources':[a],
                 'verification_status':'unreviewed','source_integrity':'intact'},
                {'finding_id':'f2','text':'不同版本的候选。','sources':[a,b],
                 'verification_status':'unreviewed','source_integrity':'stale'}],
            'gaps':['未读取图片内容'],'next_question':'是否需要视觉信息？','read_ranges':[]}


def handoff(p, name='delegate_research'):
    """返回一组同时规划和委派的完整tool往返，含供应商推理字段。"""
    child = {'subtask_id':p['task_id'],'status':'partial','stop_reason':'具体缺口',
             'research_progress':p,'findings':[f['text'] for f in p['findings']],
             'gaps':list(p['gaps']),'evidence_ids':['e1'],
             'assessments':[{'claim':'保留判断','verdict':'insufficient','basis':'原文尚不支持'}]}
    return [{'role':'system','content':'原指令'}, {'role':'user','content':'比较、不保存'},
            {'role':'assistant','content':'','reasoning_content':'保持原推理字段', 'tool_calls':[
                {'id':'plan','type':'function','function':{'name':'set_research_plan','arguments':'{}'}},
                {'id':'delegate','type':'function','function':{'name':name,'arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'plan','content':'{"ok":true,"data":{"plan":"原计划"}}'},
            {'role':'tool','tool_call_id':'delegate','content':json.dumps({'ok':True,'data':{'subtasks':[child]}})}]


class ParentProjectionTests(unittest.TestCase):
    """不调用模型，仅验证可逆来源目录和同版本引用。"""

    def test_single_group_retains_exact_findings_and_sources(self):
        """单个最新批次也去掉重复展开；源身份可完整重建，不同版本不合并。"""
        p=progress(); msgs=handoff(p)
        context={'save_policy':'deny','subtasks':[{'subtask_id':'child','status':'partial','research_progress':p}]}
        before=copy.deepcopy([msgs,context])
        projected,current,metrics=project_research_handoffs(msgs,context)
        validate_messages(projected)
        self.assertEqual([msgs,context],before)
        self.assertEqual(projected[2],msgs[2])
        compact=current['subtasks'][0]['research_progress']
        self.assertEqual(len(compact['source_catalog']),2)
        for old,new in zip(p['findings'],compact['findings']):
            restored={k:v for k,v in new.items() if k!='source_refs'}
            restored['sources']=[compact['source_catalog'][key] for key in new['source_refs']]
            self.assertEqual(restored,old)
        result=json.loads(projected[-1]['content'])['data']['subtasks'][0]
        self.assertNotIn('findings',result)
        self.assertEqual(result['findings_in'],'current_task.subtasks[0].research_progress.findings')
        self.assertEqual(result['research_progress']['revision'],2)
        self.assertEqual(result['assessments'],json.loads(msgs[-1]['content'])['data']['subtasks'][0]['assessments'])
        self.assertEqual(current['save_policy'],'deny')
        self.assertEqual(metrics['handoff_projected_calls'],['delegate'])

    def test_old_revision_and_unrepresented_fields_stay_visible(self):
        """不能把旧结果指向新版本或扔掉仅在结果中的额外发现。"""
        old=progress(revision=1); current=progress(revision=2)
        msgs=handoff(old)
        value=json.loads(msgs[-1]['content']);value['data']['subtasks'][0]['findings'].append('额外结果')
        msgs[-1]['content']=json.dumps(value)
        out,_,_=project_research_handoffs(msgs,{'subtasks':[{'subtask_id':'child','research_progress':current}]})
        child=json.loads(out[-1]['content'])['data']['subtasks'][0]
        self.assertEqual(child['research_progress']['revision'],1)
        self.assertIn('findings',child['research_progress'])
        self.assertIn('额外结果',child['findings'])
        self.assertNotIn('context_path',child['research_progress'])

    def test_matching_revision_keeps_extra_findings(self):
        """精确重复以外的文本逐字保留，不做语义去重。"""
        p=progress();msgs=handoff(p)
        value=json.loads(msgs[-1]['content']);value['data']['subtasks'][0]['findings'].append('保留方法与限定！')
        msgs[-1]['content']=json.dumps(value)
        out,_,_=project_research_handoffs(msgs,{'subtasks':[{'subtask_id':'child','research_progress':p}]})
        self.assertEqual(json.loads(out[-1]['content'])['data']['subtasks'][0]['findings'],['保留方法与限定！'])

    def test_no_handoff_and_wrong_tool_unchanged(self):
        """不改原文、普通JSON和未成功工具；不据未知响应生成指针。"""
        p=progress();msgs=handoff(p,name='read_paper')
        out,_,metrics=project_research_handoffs(msgs,{})
        self.assertEqual(out,msgs);self.assertEqual(metrics['handoff_projected_calls'],[])
        msgs=handoff(p);v=json.loads(msgs[-1]['content']);v['ok']=False;msgs[-1]['content']=json.dumps(v)
        out,_,_=project_research_handoffs(msgs,{})
        self.assertEqual(out,msgs)

    def test_projected_input_is_idempotent(self):
        """再次投影不破坏来源引用或产生另一份目录。"""
        p=progress();a,b,_=project_research_handoffs(handoff(p),{'subtasks':[{'subtask_id':'child','research_progress':p}]})
        c,d,_=project_research_handoffs(a,b)
        self.assertEqual(a,c);self.assertEqual(b,d)
