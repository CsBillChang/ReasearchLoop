"""SIGKILL exactly after durable tool checkpoint and before publishing needs_input."""
import json
import os
import signal
from pathlib import Path
from research_loop.config import Settings
from research_loop.providers.deepseek import DeepSeekProvider
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from research_loop.workers.executor import TaskExecutor
root=Path(os.environ['PAUSE_TEST_ROOT'])
url=os.environ['PAUSE_TEST_URL']
if 'researchloop_test_' not in url:raise RuntimeError('isolated test database required')
settings=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=url,deepseek_api_key='fixture',deepseek_model=os.environ['PAUSE_TEST_MODEL'])
class Provider:
    def complete(self,*args,**kwargs):
        with (root/'calls').open('a') as f:f.write('call\n')
        return DeepSeekProvider.parse_response({'id':'pause-receipt','usage':{'total_tokens':1},'choices':[{'finish_reason':'tool_calls','message':{
            'role':'assistant','content':'','reasoning_content':'fixture','tool_calls':[{'id':'q','type':'function','function':{
                'name':'request_clarification','arguments':json.dumps({'question':'How many?','fields':['paper_count'],'suggested_values':['5']})}}]}}]})
    def close(self):pass
def kill_at_publish(self,*args,**kwargs):
    (root/'ready').write_text('checkpoint complete')
    os.kill(os.getpid(),signal.SIGKILL)
TaskService.complete=kill_at_publish
TaskExecutor(settings,Database(url),lambda _:Provider()).execute(os.environ['PAUSE_TEST_TASK'],'kill-worker')
