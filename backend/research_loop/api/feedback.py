"""反馈入口：验证 HTTP 输入和错误映射，研究版本交接留在业务服务。"""
from fastapi import APIRouter, Depends, HTTPException

from research_loop.api.dependencies import get_task_service
from research_loop.domain.tasks import Feedback
from research_loop.services.feedback import submit_feedback
from research_loop.services.tasks import TaskService

router = APIRouter()


@router.post('/feedback', status_code=202)
def feedback(payload: Feedback, service: TaskService = Depends(get_task_service)):
    """创建绑定同会话结果版本的反馈任务，不修改旧任务和旧核验结论。"""
    try:
        return submit_feedback(service, payload).model_dump(mode='json')
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
