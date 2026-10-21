"""临时SQLite核对同版别名的进展提交、保留和Reflection，零外网/模型。"""
import copy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from research_loop.agents.reflection import ReflectionService
from research_loop.domain.research_progress import ResearchProgressInput
from research_loop.domain.tasks import ActionRecord
from research_loop.services.research_progress import ResearchProgressService
from research_loop.storage.mysql import Database, _now


class AliasReflectionVersionTests(unittest.TestCase):
    """保留原引用和作用域，兼容显示身份而非跳过真实来源版本检查。"""

    def setUp(self):
        """每项使用独立数据库和固定合成原文，不依赖本机生产材料。"""
        temporary = tempfile.TemporaryDirectory(prefix='researchloop-alias-version-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.db = Database('sqlite:///'+str(self.directory/'test.db'))
        self.db.ensure_schema()
        self.root, self.child = 'root-A','child-A'
        self.old_id, self.new_id = 'user-url:A','2603.00595v1'
        self.db.create_task(self.root,None,'a'*64,'比较指定同版论文，不保存','deny',[])
        self.db.create_task(self.child,None,'b'*64,'读取指定第一篇，不保存','deny',[],
                            parent_task_id=self.root,root_task_id=self.root,role='researcher')
        self.db.create_task('foreign-root',None,'c'*64,'其他任务','deny',[])
        for paper_id in (self.old_id,self.new_id):
            self.db.save_paper(paper_id,{'paper_id':paper_id,'title':'Synthetic source','source':'arxiv',
                'paper_url':'https://arxiv.org/pdf/2603.00595v1','version':'v1'})
        # 原child不需要新增arXiv身份权限；它只能继续使用原来返回的这一页。
        self.db.grant_paper(self.child,self.old_id,self.root)
        self.db.grant_paper(self.root,self.old_id,self.root)
        self.db.grant_paper(self.root,self.new_id,self.root)
        source_hash = hashlib.sha256(b'synthetic-pdf').hexdigest()
        config_hash = hashlib.sha256(b'synthetic-config').hexdigest()
        cache_key = hashlib.sha256(f'{source_hash}:pymupdf:1:{config_hash}'.encode()).hexdigest()
        self.document = {'document_id':'doc:'+cache_key[:32],'paper_id':self.old_id,'content_sha256':source_hash,
            'parser_name':'pymupdf','parser_version':'1','config_sha256':config_hash,'total_pages':1,
            'cache_path':str(self.directory/'not-read.json'),'gaps':[]}
        self.db.save_document(self.document)
        self.eid = 'evidence:synthetic-A'
        self.page = {'evidence_id':self.eid,'document_id':self.document['document_id'],
                     'page_number':1,'text':'Synthetic source distinguishes reported scores from reliability.'}
        self.db.save_page(self.page)
        for task in (self.root,self.child):
            self.db.mark_task_evidence(task,self.eid)
        self.progress = ResearchProgressService(self.db)
        self.stored = self.progress.source(self.child,self.eid)
        self.target_id = 'target-A'
        current = {'target_id':self.target_id,'target_type':'answer','content_hash':'d'*64,'source_task_id':self.root,
            'items':[{'item_id':'answer','text':'Synthetic draft','sources':[copy.deepcopy(self.stored)]}],
            'origin':{},'evidence_ids':[self.eid],'goal_ids':[]}
        locations = ReflectionService(self.db)._research_refs(self.root,[self.eid])
        state = {'research_targets':{self.target_id:{'current':current,'versions':[copy.deepcopy(current)],
            'attempts':[{'sources':locations,'observed_sources':[],'result':{'status':'supported'}}]}}}
        self.state_json = json.dumps(state,ensure_ascii=False)
        self.db.execute('INSERT INTO reflection_sessions(task_id,state_json,updated_at) VALUES(?,?,?)',
                        (self.root,self.state_json,_now()))
        for owner,name in ((socket.socket,'connect'),(socket,'create_connection')):
            blocker = patch.object(owner,name,side_effect=AssertionError('NETWORK_FORBIDDEN'))
            blocker.start(); self.addCleanup(blocker.stop)

    def prepared_action(self):
        """先基于旧身份准备进展，再允许测试模拟另一路同版读取。"""
        args = ResearchProgressInput(findings=[{'text':'Synthetic bounded finding.','evidence_ids':[self.eid]}])
        value = self.progress.prepare(self.child,args)
        action = ActionRecord(task_id=self.child,sequence=1,tool_name='record_research_progress',
            arguments=args.model_dump(),status='started',observation={'ok':True,'data':{'research_progress':value}}).model_dump(mode='json')
        self.db.save_action(action)
        return action

    def rebind(self):
        """用原存储入口只改共享文档paper_id，不改变正文版本或授权。"""
        self.db.save_document({**self.document,'paper_id':self.new_id})

    def test_commit_retained_and_review_accept_alias_without_rewriting_history(self):
        """三处版本检查一致，旧child无新身份grant也能保留原授权页的旧成果。"""
        action = self.prepared_action()
        expected_ref = copy.deepcopy(action['observation']['data']['research_progress']['findings'][0]['sources'][0])
        self.rebind()
        self.progress.commit_action(action,None)
        retained = self.progress.retained(self.child)
        self.assertEqual(retained['findings'][0]['source_integrity'],'intact')
        self.assertEqual(retained['findings'][0]['sources'][0],expected_ref)
        target = ReflectionService(self.db).research_target(self.root,self.target_id,'d'*64)
        self.assertEqual(target['current']['items'][0]['sources'][0],self.stored)
        self.assertEqual(self.db.fetchone('SELECT state_json FROM reflection_sessions WHERE task_id=?',(self.root,))['state_json'],self.state_json)
        self.assertFalse(self.db.task_has_paper(self.child,self.new_id))
        self.assertEqual(self.db.get_task(self.root)['save_policy'],'deny')
        self.assertEqual(self.db.get_task(self.child)['save_policy'],'deny')
        self.assertEqual(self.db.fetchone('SELECT COUNT(*) AS n FROM save_receipts')['n'],0)

    def test_true_page_text_change_rejects_commit_and_marks_retained_stale(self):
        """真实正文变化仍阻止提交，并使已有进展明确stale。"""
        action = self.prepared_action()
        self.progress.commit_action(action,None)
        self.db.save_page({**self.page,'text':'A changed scientific statement.'})
        self.assertFalse(self.progress.source_version_matches(self.child,self.stored))
        self.assertEqual(self.progress.retained(self.child)['findings'][0]['source_integrity'],'stale')
        with self.assertRaisesRegex(ValueError,'research source version changed'):
            ReflectionService(self.db).research_target(self.root,self.target_id)

    def test_real_content_version_prevents_pending_progress_commit(self):
        """进展准备后源hash改变，不能因同版URL或alias继续提交。"""
        action = self.prepared_action()
        self.db.save_document({**self.document,'paper_id':self.new_id,'content_sha256':'e'*64})
        with self.assertRaisesRegex(ValueError,'progress source version changed'):
            self.progress.commit_action(action,None)
        self.assertEqual(self.db.list_actions(self.child)[0]['status'],'started')

    def test_missing_or_cross_task_evidence_never_becomes_authorized(self):
        """删除页授权或换根时，别名不能恢复原文访问。"""
        self.rebind()
        with self.assertRaises(PermissionError):
            self.progress.source_version_matches('foreign-root',self.stored)
        self.db.execute('DELETE FROM task_evidence WHERE task_id=? AND evidence_id=?',(self.child,self.eid))
        with self.assertRaises(PermissionError):
            self.progress.source_version_matches(self.child,self.stored)
        self.assertFalse(self.db.task_has_paper(self.child,self.new_id))

    def test_cross_version_alias_and_stale_review_hash_still_reject(self):
        """不同明确vN和旧对象hash保留原阻断，不批准新内容。"""
        self.db.save_paper(self.new_id,{'paper_id':self.new_id,'paper_url':'https://arxiv.org/pdf/2603.00595v2','version':'v2'})
        self.rebind()
        with self.assertRaisesRegex(ValueError,'research source version changed'):
            ReflectionService(self.db).research_target(self.root,self.target_id)
        with self.assertRaisesRegex(ValueError,'missing or stale'):
            ReflectionService(self.db).research_target(self.root,self.target_id,'f'*64)

    def test_missing_ref_field_and_new_parser_document_identity_fail_closed(self):
        """缺字段与新解析身份均不兼容，不改写旧ref填补信息。"""
        broken = {k:v for k,v in self.stored.items() if k!='text_sha256'}
        self.assertFalse(self.progress.source_version_matches(self.child,broken))
        changed_document = {**self.document,'document_id':'doc:new-parser-version-config'}
        self.db.save_document(changed_document)
        self.db.save_page({**self.page,'document_id':changed_document['document_id']})
        self.assertFalse(self.progress.source_version_matches(self.child,self.stored))
        with self.assertRaisesRegex(ValueError,'research source version changed'):
            ReflectionService(self.db).research_target(self.root,self.target_id)
