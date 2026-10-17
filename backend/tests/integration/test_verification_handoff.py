"""实际 API/Worker 交接缺证判断、解析范围和独立核验；模型为显式供应商替身。"""
import json
from unittest.mock import patch

from research_loop.domain.evidence import PageEvidence
from research_loop.workers.executor import TaskExecutor
from integration import test_reflection_loop as loop
from integration.test_collaboration_contract import FixturePDF
from integration.test_adaptive_research import AdaptiveScenario


class LimitedPDF(FixturePDF):
    """全文可提取文字不等于图片内容已被理解。"""

    def pages(self, document, start, count):
        """返回两页有已知事实但不含被询问版本号的合成文字。"""
        return [PageEvidence(evidence_id=f'evidence:{document.paper_id}:{n}', document_id=document.document_id,
            page_number=n, text='The encoder uses four layers.', gaps=['公式符号缺失'],
            assets=[{'asset_id':f'image-{n}', 'page_number':n}]) for n in range(start,min(start+count,3))]


class MissingEvidenceScenario(loop.ReflectionScenario):
    """核验问题逐层传递，不依赖论文 ID 或测试专用生产分支。"""

    def handler(self, request):
        """观察实际模型输入并返回结构化工具调用，全部请求走生产收据实现。"""
        body=json.loads(request.content); self.requests.append(body)
        messages=body['messages']; names={t['function']['name'] for t in body.get('tools',[])}
        obs=[json.loads(m['content']) for m in messages if m['role']=='tool']
        done=[c['function']['name'] for m in messages for c in m.get('tool_calls',[])]
        reply=AdaptiveScenario.response
        if 'finish_research' in names:
            if not obs: return reply('read_paper',{'target_ref':'input-A'})
            refs=[p['evidence_id'] for p in obs[-1]['data']['pages']]
            return reply('finish_research',{'status':'partial','findings':['可提取文字未报告版本；未解释图片。'],
                'evidence_ids':refs,'paper_ids':['paper-A'],'gaps':['未找到版本依据'],
                'assessments':[{'claim':'实验环境的版本是99','status':'insufficient','basis':'not_found',
                    'reason':'可提取文字未报告版本；不能排除未读图像内容','evidence_ids':refs}]})
        if 'finish_review' in names:
            source=json.loads(messages[-1]['content'].split('\n',1)[1])
            assert all(p['gaps'] and p['assets'] and not p['read_scope']['visual_content_interpreted'] for p in source)
            assert '判为错误' not in json.dumps(messages,ensure_ascii=False)
            # 支持的是准确披露缺证的答复，不是将原始待核验主张判为真实。
            return reply('finish_review',{'status':'supported','findings':[],'gaps':[]})
        if 'set_research_plan' not in done:
            return reply('set_research_plan',{'goal':'核验版本并披露边界','input_refs':['input-A'],
                'goals':[{'goal_id':'verify','description':'核验实验环境版本','requires_fulltext':True}]})
        if 'delegate_research' not in done:
            return reply('delegate_research',{'subtasks':[{'objective':'核验版本，不预设真伪','input_refs':['input-A'],
                'mode':'investigate','goal_ids':['verify'],'verification_claims':['实验环境的版本是99']}]})
        children=next(o['data']['subtasks'] for o in obs if 'subtasks' in o['data'])
        assert children[0]['assessments'][0]['status']=='insufficient'
        assert children[0]['evidence_scope'][0]['included_physical_pages']==[1,2]
        if 'create_note' not in done:
            return reply('create_note',{'title':'核验答复','markdown':'已读两页可提取文字，无法确认版本。图片未理解，不作否证；按约束结束，未保存。',
                'evidence_ids':children[0]['evidence_ids']})
        draft=next(o['data'] for o in obs if o['data'].get('draft_id'))
        if 'review_note' not in done:
            return reply('review_note',{'draft_id':draft['draft_id'],'content_hash':draft['content_hash'],'reason':'用户要求核验且存在信息缺口'})
        return reply('finish_task',{'draft_id':draft['draft_id'],
            'goals':[{'goal_id':'verify','status':'unresolved','reason':'无法确认版本，保留来源边界','subtask_ids':[children[0]['subtask_id']]}]})


class VerificationHandoffTests(loop.ReflectionLoopTests):
    """复用隔离环境配置；本类只执行新增的跨角色场景。"""

    # 避免继承测试重复计数，父类仍在自己的模块完整执行。
    test_success_and_completed_root_recovery_no_resend = None
    test_no_progress_partial_is_readable_and_deny_no_file = None
    test_no_result_failure_has_body_but_remains_failed = None
    test_chat_explanation_and_saved_note_are_separate_deliverables = None
    test_sigkill_receipt_and_revision_resume_without_replay = None

    def test_insufficient_handoff_review_and_resume(self):
        """缺证不是否证，答复审核不使未知事实变成已证实；重放不增加请求。"""
        from fastapi.testclient import TestClient
        from research_loop.main import app
        scenario=MissingEvidenceScenario()
        executor=TaskExecutor(self.settings,self.db,scenario.factory,lambda s,f:LimitedPDF())
        with patch('research_loop.api.dependencies.get_settings',return_value=self.settings),TestClient(app) as client:
            response=client.post('/api/tasks',json={'prompt':'仅根据附件核验实验环境的版本是否为99。不要保存。',
                'save_policy':'deny','attachments':[{'input_ref':'input-A','local_path':str(self.root/'A.pdf'),'paper_id':'paper-A'}]})
            self.assertEqual(response.status_code,202);tid=response.json()['task_id']
            executor.execute(tid);view=client.get('/api/tasks/'+tid).json()
        self.assertEqual(view['status'],'partial',view)
        self.assertIn('无法确认',view['result']['answer_markdown'])
        self.assertFalse(list(self.settings.output_dir.rglob('*.md')))
        child=next(c for c in view['subtasks'] if c['role']=='researcher')
        self.assertEqual(child['assessments'][0]['basis'],'not_found')
        self.assertEqual(child['evidence_scope'][0]['asset_pages'],[1,2])
        calls=len(scenario.requests)
        self.db.update_task(tid,status='queued',result_json=None)
        executor.execute(tid,'restored')
        self.assertEqual(len(scenario.requests),calls)
