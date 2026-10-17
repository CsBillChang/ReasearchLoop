"""明确回答引用到核验快照的合同测试；判断为显式回调，不调用模型。"""
import hashlib
from pathlib import Path
from uuid import uuid4
import unittest

from unit import test_reflection_targets as base
from research_loop.config import Settings
from research_loop.domain.reflection import ResearchReviewInput
from research_loop.domain.tasks import TaskCreate
from research_loop.services.conversations import ConversationService
from research_loop.services.tasks import TaskService


class ReviewObjectHandoffTests(unittest.TestCase):
    """核验对象由服务端加载，当前内容、来源权限与判断保持分离。"""

    setUp = base.ResearchTargetTests.setUp
    execute = base.ResearchTargetTests.execute
    revise = base.ResearchTargetTests.revise

    def seed(self, *, referenced=True, role='assistant'):
        """先建立旧回答，再通过正常任务入口引用精确内容hash。"""
        root = Path(self.temp.name)
        settings = Settings(project_root=root, data_dir=root/'data', output_dir=root/'notes', database_url=self.db.url)
        service = ConversationService(self.db, settings)
        cid = service.create('review source fixture', uuid4().hex)['conversation_id']
        self.body = '保留开头。错误句。保留结尾。'
        with service.store.transaction() as tx:
            mid = service.add_message(tx, cid, 'historical-fixture', uuid4().hex, role, self.body)
        prompt = '核验当前回答。' + ('hash=' + hashlib.sha256(self.body.encode()).hexdigest() if referenced else '')
        self.task = TaskService(self.db, settings).create(TaskCreate(prompt=prompt, conversation_id=cid, save_policy='deny')).task_id
        self.args = ResearchReviewInput(target_type='answer', reason='核对方法',
                                       verification_questions=['方法是否使用语言？'])
        return service, cid, mid

    def test_omitted_body_loads_exact_reference_and_reuses_review(self):
        """省略正文不需要模型复制；同一请求复用原核验身份。"""
        _, _, mid = self.seed()
        result = self.service.review_research(self.task, self.args, self.execute())
        self.assertEqual(result['target']['items'][0]['text'], self.body)
        origin = result['target']['origin']['answer_source']
        self.assertEqual(origin['source_id'], mid)
        self.assertFalse(origin['is_original_paper_evidence'])
        self.assertFalse(origin['save_permission_inherited'])
        self.assertEqual(result['target']['evidence_ids'], [])
        again = self.service.review_research(self.task, self.args, self.execute())
        self.assertEqual(again['review_id'], result['review_id'])
        self.assertEqual(len(self.calls), 1)

    def test_reference_keeps_current_local_revision(self):
        """再次引用旧回答不能把已修订的当前版本覆盖回旧文。"""
        self.seed()
        first = self.service.review_research(self.task, self.args, self.execute())
        changed = self.revise(first)
        result = self.service.review_research(self.task, self.args, self.execute('supported'))
        self.assertEqual(result['target']['content_hash'], changed['target']['content_hash'])
        self.assertIn('修正句。', result['target']['items'][0]['text'])

    def test_missing_and_user_message_are_not_inferred(self):
        """无明确引用或引用用户消息时，在启动Reviewer之前拒绝。"""
        for reference, role in [(False, 'assistant'), (True, 'user')]:
            with self.subTest(reference=reference, role=role):
                self.seed(referenced=reference, role=role)
                with self.assertRaisesRegex(ValueError, 'source missing or ambiguous'):
                    self.service.review_research(self.task, self.args, self.execute())
        self.assertEqual(self.calls, [])

    def test_foreign_or_future_message_is_not_accepted(self):
        """同内容hash不能跨会话取消息，也不能核验当前用户输入之后的回答。"""
        service, _, _ = self.seed()
        cid = service.create('other conversation', uuid4().hex)['conversation_id']
        self.task = TaskService(self.db).create(TaskCreate(prompt='hash=' + hashlib.sha256(self.body.encode()).hexdigest(),
            conversation_id=cid, save_policy='deny')).task_id
        with self.assertRaisesRegex(ValueError, 'source missing or ambiguous'):
            self.service.review_research(self.task, self.args, self.execute())
        with service.store.transaction() as tx:
            service.add_message(tx, cid, 'future-fixture', uuid4().hex, 'assistant', self.body)
        with self.assertRaisesRegex(ValueError, 'source missing or ambiguous'):
            self.service.review_research(self.task, self.args, self.execute())

    def test_changed_source_invalidates_existing_target(self):
        """历史回答被更改后，当前target不能继续把旧来源当有效身份。"""
        _, _, mid = self.seed()
        first = self.service.review_research(self.task, self.args, self.execute())
        self.db.execute('UPDATE conversation_messages SET content=? WHERE message_id=?', ('changed', mid))
        with self.assertRaisesRegex(ValueError, 'source missing or stale'):
            self.service.research_target(self.task, first['target_id'])
        args = self.args.model_copy(update={'target_id': first['target_id'], 'content_hash': first['content_hash']})
        with self.assertRaisesRegex(ValueError, 'source missing or stale'):
            self.service.review_research(self.task, args, self.execute())
        self.assertEqual(len(self.calls), 1)

    def test_explicit_new_answer_still_supported(self):
        """新候选正文继续沿用已有参数，不强制所有任务具有历史回答。"""
        args = ResearchReviewInput(target_type='answer', answer_markdown='新候选正文。', reason='核对',
                                   verification_questions=['当前条件有据吗？'])
        result = self.service.review_research(self.task, args, self.execute())
        self.assertEqual(result['target']['items'][0]['text'], '新候选正文。')

