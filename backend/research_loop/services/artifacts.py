"""研究成果服务：授权原文、稿件、保存与上下文诊断投影，不触发模型请求。"""
from __future__ import annotations

import hashlib
import json

from research_loop.domain.tasks import SavePolicy
from research_loop.services.notes import NoteService
from research_loop.storage.files import FileStore


class ArtifactService:
    """单用户产品查询；文件访问通过任务中的授权身份，不开放任意路径。"""

    def __init__(self, db, settings):
        """复用已有持久层和项目内文件根目录。"""
        self.db,self.settings=db,settings
        self.files=FileStore(settings.data_dir,settings.output_dir,settings.project_root)

    def artifacts(self, task_id):
        """返回当前任务家族实际取得的论文、证据与笔记元数据，正文按需读取。"""
        task=self.db.get_task(task_id)
        if not task:
            raise KeyError('任务不存在')
        papers=self.db.fetchall('SELECT DISTINCT p.paper_id,p.metadata_json FROM papers p JOIN task_papers tp ON tp.paper_id=p.paper_id JOIN tasks t ON t.task_id=tp.task_id WHERE t.root_task_id=?',(task_id,))
        evidence=self.db.fetchall('SELECT DISTINCT e.evidence_id,e.document_id,e.page_number,d.paper_id FROM page_evidence e JOIN documents d ON d.document_id=e.document_id JOIN task_evidence te ON te.evidence_id=e.evidence_id JOIN tasks t ON t.task_id=te.task_id WHERE t.root_task_id=? ORDER BY d.paper_id,e.page_number',(task_id,))
        drafts=self.db.fetchall('SELECT draft_id,title,content_hash,created_at FROM drafts WHERE task_id=? ORDER BY created_at DESC,draft_id DESC',(task_id,))
        for row in drafts:
            row['saved']=bool(self.db.get_receipt(row['draft_id'],row['content_hash']))
        attachments=[{k:v for k,v in a.items() if k!='local_path'} for a in json.loads(task['attachments_json'])]
        for e in evidence:
            e['page_number']=e['page_number'] or None
        metadata=[json.loads(p['metadata_json']) for p in papers]
        known={p['paper_id'] for p in papers}
        # 直接上传的原文不一定经过 search_papers，不能因没有候选元数据隐藏已读页。
        # 这里只展示任务家族真实获授的页；文件名作为标签，不伪造论文标题或作者。
        for page in evidence:
            if page['paper_id'] not in known:
                attachment=next((a for a in attachments if a['input_ref']==page['paper_id']),{})
                metadata.append({'paper_id':page['paper_id'],
                    'title':attachment.get('filename') or page['paper_id'],
                    'source':'已读取的授权原文','published':None,'metadata_origin':'authorized_read'})
                known.add(page['paper_id'])
        from research_loop.services.evidence_matrix import EvidenceMatrixService
        matrices = EvidenceMatrixService(self.db).history(task_id) if task['root_task_id'] == task_id else []
        return {'papers':metadata, 'evidence':evidence,'drafts':drafts,'attachments':attachments,
                'evidence_matrices': matrices}

    def evidence(self, task_id, evidence_id):
        """证据只能来自当前任务家族；无物理页码保留null。"""
        row=self.db.fetchone('SELECT te.evidence_id FROM task_evidence te JOIN tasks t ON t.task_id=te.task_id WHERE t.root_task_id=? AND te.evidence_id=?',(task_id,evidence_id))
        if not row:
            raise KeyError('证据不属于当前任务')
        page=self.db.get_page(evidence_id)
        doc=self.db.fetchone('SELECT * FROM documents WHERE document_id=?',(page['document_id'],))
        pdf=self.files.cache_pdf_path(doc['content_sha256'])
        return {**page,'paper_id':doc['paper_id'],'pdf_available':pdf.is_file(),'gaps':json.loads(page['gaps_json'])}

    def pdf(self, task_id, evidence_id):
        """仅返回授权PDF缓存，Jina文本不伪装为PDF。"""
        page=self.evidence(task_id,evidence_id)
        doc=self.db.fetchone('SELECT * FROM documents WHERE document_id=?',(page['document_id'],))
        pdf=self.files.cache_pdf_path(doc['content_sha256'])
        if not pdf.is_file() or hashlib.sha256(pdf.read_bytes()).hexdigest()!=doc['content_sha256']:
            raise KeyError('未取得可展示的PDF缓存')
        return pdf

    def draft(self, task_id, draft_id):
        """读取具体草稿及绑定此hash的执行结果，不采信正文自称的核验状态。"""
        draft=self.db.get_draft(draft_id)
        if not draft or draft['task_id']!=task_id:
            raise KeyError('草稿不属于当前任务')
        from research_loop.agents.reflection import ReflectionService
        attempts=ReflectionService(self.db).snapshot(task_id).get('attempts',[])
        attempt=next((a for a in reversed(attempts) if a['draft_id']==draft_id
                      and a['content_hash']==draft['content_hash']),None)
        row=self.db.fetchone('SELECT result_json FROM review_results WHERE task_id=? AND draft_id=? AND content_hash=? ORDER BY created_at DESC LIMIT 1',
                             (task_id,draft_id,draft['content_hash']))
        review=(attempt.get('result') or {'status':'running'}) if attempt else json.loads(row['result_json']) if row else {'status':'not_run'}
        verification={'content_hash':draft['content_hash'],'status':review['status'],
                      'questions':attempt.get('questions',[]) if attempt else [],'gaps':review.get('gaps',[])}
        return {**draft,'saved':bool(self.db.get_receipt(draft_id,draft['content_hash'])),
                'verification':verification}

    def save_draft(self, task_id: str, draft_id: str, content_hash: str) -> dict:
        """保存用户已明确确认的终态稿件，并验证当前版本、冲突与幂等收据。

        Args:
            task_id: 稿件所属的根任务。
            draft_id: 用户当前看到的稿件身份。
            content_hash: 用户确认时绑定的精确内容版本。

        Returns:
            保存收据；旧任务的保存权限不会被这次操作改写。

        Raises:
            KeyError: 稿件不属于任务。
            ValueError: 任务仍在执行、稿件已过期或版本不匹配。
        """
        row = self.draft(task_id, draft_id)
        task = self.db.get_task(task_id)
        if task['status'] not in {'completed', 'partial', 'failed', 'cancelled'}:
            raise ValueError('任务仍在运行')
        latest = self.db.fetchone(
            'SELECT draft_id FROM drafts WHERE task_id=? ORDER BY created_at DESC,draft_id DESC LIMIT 1',
            (task_id,),
        )
        if not latest or latest['draft_id'] != draft_id:
            raise ValueError('草稿已过期，请刷新')
        if row['content_hash'] != content_hash:
            raise ValueError('版本已过期')
        result = NoteService(self.db, self.files).save(task_id, SavePolicy.allow, draft_id, content_hash)
        if result.status != 'already_saved':
            self.db.append_event(task_id, 'user_saved_note', {'draft_id': draft_id, 'content_hash': content_hash})
        return result.model_dump(mode='json')


    def context(self, task_id: str):
        """返回实际请求使用的记忆与压缩指标，不公开模型推理或整份messages。"""
        if not self.db.get_task(task_id):raise KeyError('任务不存在')
        memories=[]
        for call in self.db.list_model_calls(task_id):
            request=json.loads(call.get('request_json') or '{}')
            for message in request.get('messages',[]):
                content=message.get('content')
                if isinstance(content,str) and content.startswith('[ResearchLoop Memory v1]'):
                    start=content.find('\n{')
                    if start>=0:
                        try:
                            data=json.loads(content[start+1:])
                            memories.append({'request_number':call['sequence_no'],'preferences':data.get('preferences',{}),'related_research':data.get('related_history',[]),'history_readback':data.get('older_messages_readback')})
                        except ValueError:pass
        checkpoint=self.db.fetchone('SELECT state_json FROM checkpoints WHERE task_id=?',(task_id,))
        state=json.loads(checkpoint['state_json']) if checkpoint else {}
        return {'memory_usage':memories,'compression':state.get('context_metrics',{}),'note':'历史研究和偏好不是原文证据，也不授予保存权限'}

