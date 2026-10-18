"""精读节点的隔离 SIGKILL 子进程；供应商替身，绝不发真实模型请求。"""
import json
import os
import sys
import time
from pathlib import Path
import httpx
from research_loop.agents.graph import build_graph
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.services.papers import PaperService
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
from research_loop.tools.registry import ToolRegistry


def main():
    """在响应落盘但节点 checkpoint 尚未推进时暴露强杀点。"""
    db = Database(os.environ['READING_RECOVERY_DB'])
    task_id = os.environ['READING_RECOVERY_TASK']
    def handler(request):
        with Path(os.environ['READING_RECOVERY_COUNTER']).open('a') as output:
            output.write('one physical request\n')
        return httpx.Response(200,json={'id':'reading-recovery-receipt','choices':[{'finish_reason':'stop','message':{'content':'# 固定精读\n（完）'}}],
                                      'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}})
    original = db.save_model_call
    def blocked(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[2]=='succeeded':
            Path(os.environ['READING_RECOVERY_READY']).write_text(task_id)
            time.sleep(60)
        return result
    if sys.argv[1]=='crash':
        db.save_model_call=blocked
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        graph=build_graph(DeepSeekProvider('fixture',client=client),ToolRegistry([]),{'papers':PaperService(db,None,None)},db,
                          checkpoint_store=CheckpointStore(db),deep_reading=True)
        graph.invoke(db.get_checkpoint(task_id))


if __name__=='__main__':
    main()
