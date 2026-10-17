"""自然语言偏好与会话导航：真实HTTP合同，隔离SQLite，不发送模型请求。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from research_loop.config import Settings
from research_loop.main import app
from research_loop.storage.mysql import Database
from research_loop.services.conversations import ConversationService
from research_loop.services.memory import MemoryService
from research_loop.services.tasks import TaskService
from research_loop.domain.tasks import TaskCreate
from research_loop.agents.memory import ContextService


class ProfileNavigationTests(unittest.TestCase):
    """覆盖持久化、并发边界、旧记录兼容和模型上下文。"""

    def setUp(self):
        """每例新建临时库，不接生产和Worker。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        self.settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url='sqlite:///'+str(root/'test.db'))
        self.db=Database(self.settings.database_url);self.db.ensure_schema()
        p=patch('research_loop.api.dependencies.get_settings',return_value=self.settings);p.start();self.addCleanup(p.stop)
        self.client=TestClient(app);self.addCleanup(self.client.close)
        self.conv=ConversationService(self.db)
        self.cid=self.conv.create('原始名称','original')['conversation_id']

    def statement(self, content, key='statement'):
        """经新增HTTP入口提交整段原文。"""
        return self.client.post('/api/profile/preferences/statement',json={'content':content,'idempotency_key':key})

    def test_statement_verbatim_dedup_replay_and_conflict(self):
        """多句与否定原样保留；双击/重复文字不重复，换文复用键被拒。"""
        text='我关注开放词汇检测。\n默认中文，不要默认保存文件。'
        one=self.statement('  '+text+'  ').json()
        self.assertEqual(one['content'],text)
        self.assertEqual(one['source'],{'type':'profile_statement','quote':text})
        self.assertEqual(self.statement(text).json()['memory_id'],one['memory_id'])
        self.assertEqual(self.statement(text,'another').json()['memory_id'],one['memory_id'])
        self.assertEqual(self.statement('另一个内容').status_code,409)
        self.assertEqual(len(self.client.get('/api/profile').json()['explicit_preferences']),1)
        self.assertEqual(self.db.fetchall('SELECT * FROM tasks'),[])

    def test_edit_stale_revoke_and_no_resurrection(self):
        """旧编辑接口兼容自然语言记录；旧请求重放不复活撤销偏好。"""
        one=self.statement('默认简短中文').json()
        update={'slot':one['slot'],'content':'默认详细中文','expected_version':1,'idempotency_key':'edit'}
        two=self.client.post('/api/profile/preferences',json=update).json()
        self.assertEqual(two['version'],2)
        update['idempotency_key']='stale'
        self.assertEqual(self.client.post('/api/profile/preferences',json=update).status_code,409)
        self.client.post('/api/profile/forget',json={'memory_id':two['memory_id'],'expected_version':2})
        self.assertEqual(self.statement('默认简短中文').json()['status'],'superseded')
        self.assertEqual(self.client.get('/api/profile').json()['explicit_preferences'],[])

    def test_natural_preference_enters_context_without_save_authority(self):
        """后续Agent实际上下文获得原文，但旧偏好不改变本轮deny权限。"""
        text='我研究多模态大模型，习惯中文解释。'
        self.statement(text)
        tid=TaskService(self.db).create(TaskCreate(prompt='继续研究',conversation_id=self.cid,save_policy='deny',idempotency_key='task')).task_id
        context=ContextService(self.db,MemoryService(self.db),self.settings).history(tid)
        self.assertEqual(context['preferences'][0]['content'],text)
        self.assertEqual(self.db.get_task(tid)['save_policy'],'deny')
        self.assertEqual(self.db.list_model_calls(tid),[])

    def test_whitespace_and_limits(self):
        """输入边界返回422，不落空白或超长记录。"""
        for text in ('   ','x'*1001):self.assertEqual(self.statement(text).status_code,422)
        for title in (' \n ','x'*201):
            self.assertEqual(self.client.post('/api/conversations/'+self.cid+'/rename',json={'title':title,'expected_title':'原始名称','idempotency_key':str(uuid4())}).status_code,422)

    def test_rename_preserves_messages_and_rejects_stale(self):
        """改名与消息分离；重放只读当前名，不覆盖其他窗口的新名字。"""
        with self.conv.store.transaction() as tx:self.conv.add_message(tx,self.cid,'fixture','fixture-msg','user','已有消息')
        before=self.db.fetchall('SELECT * FROM conversation_messages')
        request={'title':'  开放词汇\n检测  ','expected_title':'原始名称','idempotency_key':'rename'}
        url='/api/conversations/'+self.cid+'/rename'
        self.assertEqual(self.client.post(url,json=request).json()['title'],'开放词汇 检测')
        self.assertEqual(self.client.post(url,json=request).status_code,200)
        self.assertEqual(self.client.post(url,json={**request,'idempotency_key':'stale'}).status_code,409)
        self.assertEqual(self.client.post(url,json={'title':'第二次名称','expected_title':'开放词汇 检测','idempotency_key':'new'}).status_code,200)
        self.assertEqual(self.client.post(url,json=request).json()['title'],'第二次名称')
        self.assertEqual(self.db.fetchall('SELECT * FROM conversation_messages'),before)
        self.assertEqual(self.client.post('/api/conversations/missing/rename',json=request).status_code,404)

    def test_list_last_activity_and_rename_keeps_date(self):
        """跨日活跃旧会话排前；人工改名不伪造最近交流时间。"""
        older=self.conv.create('老会话','older')['conversation_id']
        with self.conv.store.transaction() as tx:
            tx.execute('UPDATE conversations SET created_at=? WHERE conversation_id=?',('2026-10-01T00:00:00+00:00',older))
            mid=self.conv.add_message(tx,older,'fixture','fixture-date','user','继续')
            tx.execute('UPDATE conversation_messages SET created_at=? WHERE message_id=?',('2030-01-01T00:00:00+00:00',mid))
        items=self.client.get('/api/conversations').json()['conversations']
        self.assertEqual(items[0]['conversation_id'],older)
        self.client.post('/api/conversations/'+older+'/rename',json={'title':'新名字','expected_title':'老会话','idempotency_key':'rename'})
        self.assertEqual(self.conv.list()[0]['last_activity_at'],items[0]['last_activity_at'])

    def test_readonly_blocks_new_mutations(self):
        """只读演示保持禁止新增偏好和改名，返回403。"""
        with patch.dict('os.environ',{'RESEARCHLOOP_READONLY':'1'}):
            self.assertEqual(self.statement('偏好').status_code,403)
            self.assertEqual(self.client.post('/api/conversations/'+self.cid+'/rename',json={'title':'新','expected_title':'原始名称','idempotency_key':'x'}).status_code,403)
