"""Isolated SIGKILL fixture; no real provider requests."""
import json,os,sys,time
from pathlib import Path
from test_collaboration_contract import Scenario,FixturePDF
from research_loop.config import Settings
from research_loop.storage.checkpoints import CheckpointStore
from research_loop.storage.mysql import Database
import research_loop.workers.executor as executor_module
from research_loop.workers.executor import TaskExecutor

class PersistentScenario(Scenario):
    def handler(self,request):
        body=json.loads(request.content)
        names=[t['function']['name'] for t in body.get('tools',[])]
        with Path(os.environ['AG2_COLLAB_COUNTER']).open('a') as f:f.write(json.dumps({'reviewer':'finish_review' in names,'researcher':'finish_research' in names})+'\n')
        return super().handler(request)
class BlockReviewCheckpoint(CheckpointStore):
    def save(self,task_id,state):
        if self.db.get_task(task_id)['role']=='reviewer' and self.db.list_model_calls(task_id):
            Path(os.environ['AG2_COLLAB_READY']).write_text(task_id)
            time.sleep(60)
        return super().save(task_id,state)
root=Path(os.environ['AG2_COLLAB_ROOT'])
s=Settings(project_root=root,data_dir=root/'data',output_dir=root/'notes',database_url=os.environ['AG2_COLLAB_TEST_DATABASE_URL'])
if sys.argv[1]=='crash':executor_module.CheckpointStore=BlockReviewCheckpoint
scenario=PersistentScenario()
TaskExecutor(s,provider_factory=scenario.factory,pdf_factory=lambda s,f:FixturePDF()).execute(os.environ['AG2_COLLAB_TASK'],sys.argv[1])
