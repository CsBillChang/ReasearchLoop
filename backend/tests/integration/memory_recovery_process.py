"""AG4隔离故障注入进程；只有替身模型，数据库必须为专用测试库。"""
import json
import os
import signal
import sys
from pathlib import Path
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.services.conversations import ConversationService
from research_loop.services.note_versions import NoteVersionService
from research_loop.domain.memory import NoteUpdate
from research_loop.storage.files import FileStore
from research_loop.workers.executor import TaskExecutor
from integration.test_memory import MemoryProvider


class LoggedProvider(MemoryProvider):
    """记录每次实际替身发送，跨进程判断是否重复请求。"""
    def send(self,*args,**kwargs):
        """先落发送计数，再返回确定性响应。"""
        with Path(os.environ['AG4_COUNTER']).open('a') as f:
            f.write('sent\n');f.flush();os.fsync(f.fileno())
        return super().send(*args,**kwargs)


def kill():
    """在指定持久化窗口真实SIGKILL。"""
    os.kill(os.getpid(),signal.SIGKILL)


if __name__=='__main__':
    url=os.environ['AG4_TEST_DATABASE_URL']
    assert url.rsplit('/',1)[-1].startswith('researchloop_test_ag4_')
    root=Path(os.environ['AG4_ROOT'])
    settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url,max_model_requests=6,reserve_model_requests=0)
    db=Database(url)
    mode=sys.argv[1]
    if mode=='receipt':
        original=Database.save_model_call
        def save(self,*args,**kwargs):
            """成功响应已提交，解析/图checkpoint之前退出。"""
            result=original(self,*args,**kwargs)
            if args[2]=='succeeded':kill()
            return result
        Database.save_model_call=save
    if mode=='projection':
        original=ConversationService.reconcile
        def reconcile(self,task_id):
            """权威终态已提交，派生会话和历史尚未写入。"""
            if self.db.get_task(task_id)['status']=='completed':kill()
            return original(self,task_id)
        ConversationService.reconcile=reconcile
    if mode=='note':
        original=FileStore.save_markdown
        def write(self,*args,**kwargs):
            """新版本文件已fsync，数据库事务提交之前退出。"""
            result=original(self,*args,**kwargs)
            kill()
            return result
        FileStore.save_markdown=write
        payload=json.loads(Path(os.environ['AG4_PAYLOAD']).read_text())
        NoteVersionService(db,FileStore(root/'data',root/'notes',root)).update(payload['note_id'],NoteUpdate.model_validate(payload['update']),True)
    else:
        tid=os.environ['AG4_TASK']
        provider=LoggedProvider(db.get_task(tid)['prompt'])
        TaskExecutor(settings,db,lambda s:provider).execute(tid,'recovery-'+mode)
