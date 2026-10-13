# MySQL 澄清事务测试：恢复回答与重入队保持一致，拒绝过期或冲突的幂等请求。
import os
import unittest
from uuid import uuid4
from urllib.parse import urlparse
from . import test_clarification_pipeline as pipeline
from research_loop.storage.mysql import Database
from research_loop.services.tasks import TaskService
from research_loop.domain.tasks import TaskCreate


@unittest.skipUnless(os.getenv('RESEARCHLOOP_RUN_MYSQL_TESTS')=='1','explicit isolated MySQL test required')
class ClarificationMySQLTests(pipeline.ClarificationPipelineTests):
    def setUp(self):
        super().setUp()
        url=os.environ.get('RESEARCHLOOP_TEST_DATABASE_URL','')
        if not urlparse(url).path.lstrip('/').startswith('researchloop_test_'):
            raise RuntimeError('isolated researchloop_test_* MySQL required')
        self.settings=self.settings.model_copy(update={'database_url':url})
        self.db=Database(url);self.db.ensure_schema();self.service=TaskService(self.db)
        self.task=self.service.create(TaskCreate(prompt='比较论文，先询问数量',idempotency_key='pause-'+uuid4().hex))
