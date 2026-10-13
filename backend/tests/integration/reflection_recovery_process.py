"""隔离 SIGKILL 子进程；只调用供应商替身，保留数据库和请求计数。"""
import json
import os
from pathlib import Path
import signal
import sys

from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.workers.executor import TaskExecutor
import research_loop.workers.executor as executor_module
from test_reflection_loop import ReflectionScenario
from test_collaboration_contract import FixturePDF


class LoggedScenario(ReflectionScenario):
    """每次真正发送替身请求写入独立文件，恢复后可以核对无重发。"""

    def handler(self, request):
        """记录角色后沿同一确定性场景生成响应。"""
        data=json.loads(request.content)
        names=[t['function']['name'] for t in data.get('tools',[])]
        with Path(os.environ['AG3_COUNTER']).open('a') as f:
            f.write(json.dumps({'reviewer':'finish_review' in names,'researcher':'finish_research' in names})+'\n')
        return super().handler(request)


class KillAfterReceipt(CheckpointStore):
    """第二轮 Reviewer 响应收据已提交，但图尚未应用响应时强杀。"""

    def save(self, task_id, state):
        """必须观察到成功 receipt 才故障注入，避免只验证发送前退出。"""
        row=self.db.get_task(task_id)
        parent=row.get('parent_task_id')
        session=ReflectionService(self.db).snapshot(parent) if parent else {}
        if row['role']=='reviewer' and len(session.get('attempts',[]))==2 and any(r['status']=='succeeded' for r in self.db.list_model_calls(task_id)):
            Path(os.environ['AG3_READY']).write_text(task_id)
            os.kill(os.getpid(),signal.SIGKILL)
        return super().save(task_id,state)


if __name__ == '__main__':
    root=Path(os.environ['AG3_ROOT'])
    settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=os.environ['AG3_TEST_DATABASE_URL'])
    mode=sys.argv[1]
    if mode=='receipt':executor_module.CheckpointStore=KillAfterReceipt
    if mode=='revision':
        original=ReflectionService.revise
        def kill_revision(self, task_id, payload):
            """修订事务提交之后、工具观察保存之前强杀。"""
            result=original(self,task_id,payload)
            Path(os.environ['AG3_READY']).write_text(result['draft_id'])
            os.kill(os.getpid(),signal.SIGKILL)
        ReflectionService.revise=kill_revision
    scenario=LoggedScenario()
    TaskExecutor(settings,provider_factory=scenario.factory,pdf_factory=lambda s,f:FixturePDF()).execute(os.environ['AG3_TASK'],mode)
