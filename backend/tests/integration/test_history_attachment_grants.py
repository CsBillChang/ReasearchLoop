"""附件与历史原文授权的隔离合同：临时SQLite和PDF替身，禁止真实请求。"""
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from research_loop.domain.evidence import PaperCandidate
from research_loop.domain.memory import HistoryRead
from research_loop.domain.notes import CreateNoteInput
from research_loop.domain.tasks import SavePolicy
from research_loop.services.memory import MemoryService
from research_loop.services.notes import NoteService
from research_loop.services.papers import PaperService
from research_loop.storage.files import FileStore
from research_loop.storage.memory import dump
from research_loop.storage.mysql import Database, _now
from research_loop.tools.base import ToolContext
from research_loop.tools.memory import specs as memory_specs
from research_loop.tools.papers import specs as paper_specs
from research_loop.tools.registry import ToolRegistry
from integration.test_collaboration_contract import FixturePDF


class HistoryAttachmentGrantTests(unittest.TestCase):
    """重放U13输入形状，不改变模型计划、保存权限或旧证据。"""

    def setUp(self):
        """每例独立库和合成旧记录，PDF解析替身不会联网。"""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = Database('sqlite:///' + str(self.root / 'test.db'))
        self.db.ensure_schema()
        self.files = FileStore(self.root / 'data', self.root / 'notes', self.root)
        self.memory = MemoryService(self.db)
        self.papers = PaperService(self.db, None, FixturePDF())
        self.notes = NoteService(self.db, self.files)
        self.a = self.paper('A', '2601.12345v1')
        self.b = self.paper('B', '2601.23456v1')
        self.c = self.paper('C', '2601.34567v1')
        self.attachment = {'input_ref': 'current-A', 'paper_id': self.a.paper_id,
                           'url': self.a.pdf_url, 'filename': '当前附件A'}
        self.old = self.task('历史研究', attachments=[], policy='allow')
        self.db.grant_paper(self.old, self.b.paper_id, self.old)
        self.old_pages = self.papers.read(self.old, self.b.paper_id)['pages']
        self.record = {'task_id': self.old, 'paper_ids': [self.b.paper_id, 'missing-paper'],
                       'papers': [self.b.model_dump(mode='json')], 'answer': '历史B研究',
                       'save_policy': 'allow', 'evidence_ids': [p['evidence_id'] for p in self.old_pages]}
        self.db.execute('INSERT INTO research_memories(task_id,conversation_id,record_json,search_text,created_at) VALUES(?,?,?,?,?)',
                        (self.old, 'synthetic-conversation', dump(self.record), '历史B研究', _now()))

    def paper(self, name, arxiv):
        """只创建题录，不向当前任务授予权限。"""
        candidate = PaperCandidate(paper_id='history-' + name + '-' + uuid4().hex,
            title='Synthetic ' + name, authors=['Synthetic Author'], source='arxiv',
            paper_url='https://arxiv.org/abs/' + arxiv, pdf_url='https://arxiv.org/pdf/' + arxiv)
        self.db.save_paper(candidate.paper_id, candidate.model_dump(mode='json'))
        return candidate

    def task(self, prompt, *, attachments=None, policy='deny'):
        """直接创建隔离根任务，避免构造模型或读取生产配置。"""
        task = 'history-grant-' + uuid4().hex
        self.db.create_task(task, None, 'synthetic', prompt, policy,
                            [self.attachment] if attachments is None else attachments)
        return task

    def recall(self, task):
        """调用真实历史读回，不把历史摘要当新页证据。"""
        return self.memory.read(task, HistoryRead(source_type='research_history', source_id=self.old))

    def test_mixed_attachment_history_read_and_save_boundary(self):
        """U13形状：当前附件不排除已读B，但不继承保存或旧页证据。"""
        task = self.task('现在结合刚才读过的附件A和历史B比较，只在对话回答，不保存新文件。')
        before = self.db.get_task(task)
        result = self.recall(task)
        self.assertFalse(result['is_original_paper_evidence'])
        self.assertTrue(self.db.task_has_paper(task, self.b.paper_id))
        self.assertFalse(self.db.task_has_paper(task, self.c.paper_id))
        self.assertFalse(self.db.task_has_paper(task, 'missing-paper'))
        self.assertEqual(self.db.list_task_evidence(task), [])
        self.assertEqual(self.db.get_task(task), before)
        with self.assertRaisesRegex(ValueError, 'evidence not returned'):
            self.notes.create(task, CreateNoteInput(title='拒绝旧证据', markdown='synthetic',
                                                    evidence_ids=[self.old_pages[0]['evidence_id']]))
        read = self.papers.read(task, self.b.paper_id, page_count=1)
        self.assertEqual(read['returned_pages'], [1])
        self.assertFalse(self.db.task_has_evidence(task, self.old_pages[1]['evidence_id']))
        draft = self.notes.create(task, CreateNoteInput(title='新稿', markdown='synthetic'))
        with self.assertRaises(PermissionError):
            self.notes.save(task, SavePolicy.deny, draft.draft_id, draft.content_hash)
        self.assertEqual(self.db.get_task(task)['save_policy'], 'deny')
        self.assertEqual(list(self.files.output_dir.rglob('*.md')), [])

    def test_attachment_only_keeps_history_ungranted(self):
        """明确只用当前附件时，查看历史不能扩大论文读取权限。"""
        for prompt in ('只使用当前附件，不引用历史论文。', '仅根据我上传的PDF回答。',
                       'Use only the attached PDFs.', 'Only read current attachments.',
                       'Read the provided documents only.', 'Only from the attached PDFs.',
                       '仅根据所提供的论文。', '仅根据我上传的两篇论文。',
                       'Use only the two attached PDFs.',
                       '只使用当前附件。不要解除仅附件限制。',
                       'Use only attachments. Do not lift the attachment-only restriction.'):
            with self.subTest(prompt=prompt):
                task = self.task(prompt)
                self.recall(task)
                self.assertFalse(self.db.task_has_paper(task, self.b.paper_id))
                with self.assertRaises(PermissionError):
                    self.papers.read(task, self.b.paper_id)
                self.assertTrue(self.db.task_has_input(task, 'current-A'))

    def test_negated_only_and_output_only_do_not_limit_materials(self):
        """否定only和仅对话输出不是附件排他约束。"""
        for prompt in ('不要只使用附件，请结合已读历史B。', '不再仅依据当前附件，结合历史B。',
                       'Do not use only attached PDFs; compare with previous B.',
                       'Not only the attached PDFs; also use previous B.',
                       '只在对话中回答，不保存；结合历史B。'):
            with self.subTest(prompt=prompt):
                task = self.task(prompt)
                self.recall(task)
                self.assertTrue(self.db.task_has_paper(task, self.b.paper_id))

    def test_only_current_answered_clarification_can_release_limit(self):
        """问题、他任务回答与允许联网均不解除附件only，真实解除才生效。"""
        task = self.task('只使用当前附件。')
        clarification = str(uuid4())
        release = '解除仅附件限制，结合已读历史B。'
        self.db.save_clarification({'clarification_id': clarification, 'task_id': task, 'question': release})
        other = str(uuid4())
        self.db.save_clarification({'clarification_id': other, 'task_id': self.old, 'question': '范围？'})
        self.db.answer_clarification(other, release)
        self.recall(task)
        self.assertFalse(self.db.task_has_paper(task, self.b.paper_id))
        self.db.answer_clarification(clarification, release)
        self.recall(task)
        self.assertTrue(self.db.task_has_paper(task, self.b.paper_id))
        network_task = self.task('只使用当前附件。')
        network = str(uuid4())
        self.db.save_clarification({'clarification_id': network, 'task_id': network_task, 'question': '联网？'})
        self.db.answer_clarification(network, '允许联网。')
        self.recall(network_task)
        self.assertFalse(self.db.task_has_paper(network_task, self.b.paper_id))

    def test_negated_release_with_modifiers_keeps_existing_only(self):
        """解除前存在明确否定时，即使隔有修饰语也不新增历史身份。"""
        for answer in ('请不要擅自解除仅附件限制。',
                       'Do not ever lift the attachment-only restriction.'):
            with self.subTest(answer=answer):
                task = self.task('只使用当前附件。')
                clarification = str(uuid4())
                self.db.save_clarification({'clarification_id': clarification,
                    'task_id': task, 'question': '材料范围是否调整？'})
                self.db.answer_clarification(clarification, answer)
                self.recall(task)
                self.assertFalse(self.db.task_has_paper(task, self.b.paper_id))
                with self.assertRaises(PermissionError):
                    self.papers.read(task, self.b.paper_id)

    def test_network_denial_does_not_block_local_history(self):
        """允许本地历史身份不解除原禁止联网要求。"""
        from research_loop.domain.source_access import assert_external_access
        task = self.task('不要联网；结合当前附件与已读历史B回答。')
        self.recall(task)
        self.assertTrue(self.db.task_has_paper(task, self.b.paper_id))
        self.assertEqual(self.papers.read(task, self.b.paper_id, page_count=1)['returned_pages'], [1])
        with self.assertRaises(PermissionError):
            assert_external_access(self.db, task)

    def test_no_attachment_search_and_repeated_read_compatible(self):
        """无附件路径兼容，合法搜索可授相同历史身份，重复读回不重复授权。"""
        task = self.task('继续历史B研究。', attachments=[])
        self.memory.search('历史B研究', 'research_history', 3, task)
        self.recall(task)
        self.recall(task)
        self.assertEqual(self.db.list_task_papers(task), [self.b.paper_id])
        mixed = self.task('结合当前附件和历史B。')
        self.memory.search('历史B研究', 'research_history', 3, mixed)
        self.assertTrue(self.db.task_has_paper(mixed, self.b.paper_id))
        restricted = self.task('只使用当前附件。')
        self.memory.search('历史B研究', 'research_history', 3, restricted)
        self.assertFalse(self.db.task_has_paper(restricted, self.b.paper_id))

    def test_page_version_and_role_scopes_still_apply(self):
        """paper身份授权不扩大物理页、其他版本或子角色固定材料。"""
        task = self.task('结合当前附件与历史B；仅阅读物理第1页。')
        self.recall(task)
        with self.assertRaises(PermissionError):
            self.papers.read(task, self.b.paper_id, page_count=2)
        self.assertEqual(self.papers.read(task, self.b.paper_id, page_count=1)['returned_pages'], [1])
        alternate = self.b.model_copy(update={'paper_id': self.b.paper_id + '-v2',
                                               'pdf_url': self.b.pdf_url.replace('v1', 'v2')})
        self.db.save_paper(alternate.paper_id, alternate.model_dump(mode='json'))
        with self.assertRaises(PermissionError):
            self.papers.read(task, alternate.paper_id, page_count=1)
        registry = ToolRegistry(memory_specs() + paper_specs())
        context = ToolContext(task, SavePolicy.deny, role='researcher',
            services={'memory': self.memory, 'papers': self.papers, 'scope': {'paper_ids': [self.a.paper_id]}})
        denied = registry.execute('read_memory',
            {'source_type': 'research_history', 'source_id': self.old}, context)
        self.assertFalse(denied.ok)
        denied = registry.execute('read_paper', {'target_ref': self.b.paper_id, 'page_count': 1}, context)
        self.assertFalse(denied.ok)

    def test_message_readback_does_not_grant_paper(self):
        """消息和档案读回不因本修复变成论文原文授权。"""
        task = self.task('结合当前附件与历史B。')
        archive = 'a' * 64
        self.db.execute('INSERT INTO context_archives(snapshot_id,task_id,sequence_no,original_json,projected_json,metrics_json,created_at) VALUES(?,?,?,?,?,?,?)',
                        (archive, task, 1, dump(self.record), '{}', '{}', _now()))
        result = self.memory.read(task, HistoryRead(source_type='context_archive', source_id=archive))
        self.assertIn(self.b.paper_id, result['content'])
        self.assertFalse(self.db.task_has_paper(task, self.b.paper_id))

    def test_only_phrase_order_and_current_release(self):
        """纯函数处理明确only、否定及同条澄清内的最后有效限制。"""
        from research_loop.domain.source_access import attachment_only
        for prompt in ('仅限于当前附件。', 'Use only the attachments.',
                       'Only use these uploaded files.', '仅依据这篇PDF回答。'):
            with self.subTest(prompt=prompt):
                self.assertTrue(attachment_only([prompt]))
        self.assertFalse(attachment_only(['只用当前附件。', '解除仅附件限制，结合历史B。']))
        self.assertFalse(attachment_only(['只用当前附件。', '不要只使用附件，结合历史B。']))
        self.assertFalse(attachment_only(['Use only attachments.', 'Lift the attachment-only restriction.']))
        self.assertFalse(attachment_only(['只用当前附件。', '请现在解除仅附件限制。']))
        self.assertFalse(attachment_only(['Use only attachments.',
                                          'Please now lift the attachment-only restriction.']))
        self.assertTrue(attachment_only(['只用当前附件。', '解除仅附件限制。但仍只用当前附件。']))
        self.assertTrue(attachment_only(['只用当前附件。', '允许联网。']))
        self.assertTrue(attachment_only(['只用当前附件。', '不要解除仅附件限制。']))
        self.assertTrue(attachment_only(['Use only attachments.',
                                         'Do not lift the attachment-only restriction.']))

