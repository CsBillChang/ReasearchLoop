"""通用核验SIGKILL子进程；所有模型请求只进入HTTP替身。"""
import json
import os
from pathlib import Path
import signal
import sys

from research_loop.agents.reflection import ReflectionService
from research_loop.config import Settings
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
from integration.test_research_reflection import ResearchReflectionScenario, ScopedPDF, is_reviewer_request


def main():
    """在成功审校子结果或patch提交后终止，恢复时按收据重放。"""
    root = Path(os.environ['REFLECTION_ROOT'])
    tid = os.environ['REFLECTION_TASK']
    url = os.environ['REFLECTION_TEST_DATABASE_URL']
    if not url.rsplit('/', 1)[-1].startswith('researchloop_test_'):
        raise ValueError('isolated MySQL required')
    settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes', database_url=url)
    db = Database(url)
    scenario = ResearchReflectionScenario('answer')
    original_handler = scenario.handler
    def counted(request):
        """只记录角色计数，不记录凭据或原始私人消息。"""
        body = json.loads(request.content)
        with (root/'requests.jsonl').open('a') as file:
            file.write(json.dumps({'reviewer': is_reviewer_request(body)}) + '\n')
        return original_handler(request)
    scenario.handler = counted
    point = sys.argv[1]
    def kill():
        """标记已提交点后发送真实SIGKILL，不走异常清理模拟恢复。"""
        (root/'killed.ready').write_text(point)
        os.kill(os.getpid(), signal.SIGKILL)
    if point == 'review':
        save = db.save_subtask_result
        def saved(child_id, status, result, error=None):
            """保留成功Reviewer结果，故意在根核验会话结算前退出。"""
            save(child_id, status, result, error)
            if status == 'completed' and result.get('target_id'): kill()
        db.save_subtask_result = saved
    elif point == 'patch':
        revise = ReflectionService.revise_research
        def revised(service, task, args):
            """保留新版本，故意在工具成功行动提交前退出。"""
            result = revise(service, task, args); kill(); return result
        ReflectionService.revise_research = revised
    TaskExecutor(settings, db, scenario.factory, lambda s, f: ScopedPDF()).execute(tid, 'recovery-process')


if __name__ == '__main__':
    main()
