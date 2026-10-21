"""URL身份交接的合成回归；所有HTTP均由MockTransport提供。"""
import unittest
from uuid import uuid4

import httpx

from integration import test_web_acquisition as fixtures

ARXIV = fixtures.ARXIV
PDF_URL = fixtures.PDF_URL
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.pdf import PDFProvider
from research_loop.services.papers import PaperService
from research_loop.tools.base import ToolContext


ORIGINAL = 'https://openreview.net/forum?id=synthetic-original'
ORIGINAL_PDF = 'https://openreview.net/pdf?id=synthetic-original'


class FallbackHandoffTests(unittest.TestCase):
    """验证合法URL能执行，同时不从题录线索推断原URL身份。"""

    add_paper = fixtures.WebAcquisitionTests.add_paper
    paid_calls = fixtures.WebAcquisitionTests.paid_calls

    def setUp(self):
        """复用隔离数据库和合成PDF夹具，不继承其他测试方法。"""
        self.original_succeeds = False
        self.cancel_after_download = False
        fixtures.WebAcquisitionTests.setUp(self)
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('请读取并解释这篇论文：' + ORIGINAL, self.task))
        self.reference = self.add_paper(paper_url=ARXIV, pdf_url=PDF_URL, version='v1', source='arxiv')

    def respond(self, request):
        """原址可控403/200，其他请求只采用已有合成响应。"""
        if str(request.url) == ORIGINAL_PDF:
            self.calls.append((request.method, str(request.url)))
            if self.cancel_after_download:
                self.db.update_task(self.task, cancel_requested=1)
            if self.original_succeeds:
                return httpx.Response(200, content=self.pdf_data, headers={'content-type':'application/pdf'})
            return httpx.Response(403, text='synthetic original access denied')
        return fixtures.WebAcquisitionTests.respond(self, request)

    def run_acquire(self, target=ORIGINAL, **arguments):
        """通过真实工具schema和服务调用合成获取链。"""
        return self.registry.execute('acquire_paper_pdf', {'target_ref':target, **arguments}, self.context)

    def save_action(self, result, arguments):
        """模拟图已完成动作的持久观察；恢复不得把已知403再发送。"""
        action_id = str(uuid4())
        self.db.save_action({'action_id':action_id,'task_id':self.task,
            'sequence':len(self.db.list_actions(self.task)) + 1,'tool_name':'acquire_paper_pdf',
            'arguments':arguments,'status':'completed' if result.ok else 'failed',
            'observation':result.model_dump(mode='json')})
        return action_id

    def test_original_url_primary_success_is_independent_identity(self):
        """用户原URL直接成功时读取原址，不别名为已有arXiv题录。"""
        self.original_succeeds = True
        result = self.run_acquire(page_count=1)
        self.assertTrue(result.ok, result.model_dump())
        self.assertEqual(result.data['returned_pages'], [1])
        self.assertTrue(result.data['paper_id'].startswith('user-url:'))
        self.assertNotEqual(result.data['paper_id'], self.reference.paper_id)
        self.assertTrue(result.data['identity_handoff']['original_locator_identity_unverified'])
        self.assertIn(('GET', ORIGINAL_PDF), self.calls)
        self.assertEqual(self.paid_calls(), [])

    def test_current_user_url_before_chinese_punctuation_is_authorized(self):
        """中文尾注紧跟URL也只授权实际给定地址，不吞入后续指令。"""
        tails = ('）。请根据原文回答。', '，请读取。', '。请读取。',
                 '；然后解释。', '！请读取。', '？请读取。', '】请读取。',
                 '》请读取。', '”请读取。', '’请读取。')
        for tail in tails:
            with self.subTest(tail=tail):
                # 每个变体使用新任务，防止前例授予的身份使坏边界误通过。
                task = 'url-boundary-' + uuid4().hex
                self.db.create_task(task, None, 'synthetic',
                    '请读取论文（' + PDF_URL + tail, 'deny', [])
                context = ToolContext(task, SavePolicy.deny, services={'papers': self.service})
                result = self.registry.execute('acquire_paper_pdf',
                    {'target_ref': PDF_URL, 'page_count': 1}, context)
                self.assertTrue(result.ok, result.model_dump())
                self.assertEqual(result.data['returned_pages'], [1])
                self.assertTrue(result.data['paper_id'].startswith('user-url:'))
                self.assertEqual(self.db.list_task_papers(task), [result.data['paper_id']])
        self.assertEqual(self.paid_calls(), [])

    def test_chinese_boundary_keeps_exact_user_version_and_child_scope(self):
        """修正分界不授权别的版本、任意域名或固定材料之外的子角色。"""
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?',
            ('请读取（' + PDF_URL + '）。只读本篇。', self.task))
        for target in (PDF_URL.replace('v1', 'v2'), 'https://example.org/2601.12345v1.pdf'):
            with self.subTest(target=target):
                result = self.run_acquire(target)
                self.assertFalse(result.ok)
        with self.assertRaises(PermissionError):
            self.service._resolve_acquisition_target(self.task, PDF_URL, {}, ['assigned-other'], lambda: False)
        self.assertEqual(self.calls, [])

    def test_only_current_answered_clarification_grants_delimited_url(self):
        """模型问题和其他任务的回答不授权，本任务真实澄清中的URL才可用。"""
        task = 'url-clarification-' + uuid4().hex
        self.db.create_task(task, None, 'synthetic', '请读取我将明确提供的论文。', 'deny', [])
        context = ToolContext(task, SavePolicy.deny, services={'papers': self.service})
        clarification = str(uuid4())
        answer = '请读取（' + PDF_URL + '）。按这份原文回答。'
        self.db.save_clarification({'clarification_id': clarification, 'task_id': task, 'question': answer})
        other = str(uuid4())
        self.db.save_clarification({'clarification_id': other, 'task_id': self.task, 'question': '地址？'})
        self.db.answer_clarification(other, answer)
        denied = self.registry.execute('acquire_paper_pdf', {'target_ref': PDF_URL}, context)
        self.assertFalse(denied.ok)
        self.assertEqual(self.calls, [])
        self.db.answer_clarification(clarification, answer)
        allowed = self.registry.execute('acquire_paper_pdf', {'target_ref': PDF_URL, 'page_count': 1}, context)
        self.assertTrue(allowed.ok, allowed.model_dump())
        self.assertEqual(allowed.data['returned_pages'], [1])
        self.assertEqual(self.paid_calls(), [])

    def test_url_hint_explicit_metadata_and_resume_paid_success(self):
        """连续URL重试保留未知关联；显式题录经Fetch确认后读取独立版本。"""
        self._check_recovery(use_resolved_id=False)

    def test_resolved_placeholder_id_supports_metadata_and_resume(self):
        """首轮返回的临时ID也可带metadata_ref修正，不丢失身份交接。"""
        self._check_recovery(use_resolved_id=True)

    def _check_recovery(self, *, use_resolved_id):
        """失败观察→显式线索→新服务恢复；已付费操作物理POST只发送一次。"""
        first = self.run_acquire()
        self.assertFalse(first.ok)
        self.assertEqual(first.kind, 'validation_error')
        self.assertEqual(first.data['status'], 'metadata_reference_required')
        self.assertEqual(first.data['primary_failure']['stage'], 'primary_pdf')
        self.assertEqual(first.data['primary_failure']['error_code'], 'PDF_HTTP_ERROR')
        self.assertEqual(first.data['diagnostic']['request_state'], 'not_requested')
        self.assertEqual(first.data['diagnostic']['authorized_metadata_refs'][0]['paper_id'], self.reference.paper_id)
        placeholder = first.data['identity_handoff']['resolved_paper_id']
        self.assertEqual(self.db.get_paper(placeholder)['metadata']['title'], '')
        self.assertEqual(self.paid_calls(), [])
        self.save_action(first, {'target_ref':ORIGINAL})
        target = placeholder if use_resolved_id else ORIGINAL
        args = {'target_ref':target, 'metadata_ref':self.reference.paper_id}
        result = self.run_acquire(target, metadata_ref=self.reference.paper_id)
        self.assertTrue(result.ok, result.model_dump())
        self.assertTrue(result.data['identity_handoff']['original_locator_identity_unverified'])
        self.assertTrue(result.data['alternate_version'])
        self.assertNotIn(result.data['paper_id'], [placeholder, self.reference.paper_id])
        self.assertEqual(result.data['returned_pages'], [1, 2])
        self.assertEqual(self.db.get_paper(placeholder)['metadata']['title'], '')
        self.assertEqual(len(self.paid_calls()), 2)
        self.assertEqual(self.calls.count(('GET', ORIGINAL_PDF)), 1)
        self.save_action(result, args)
        resumed = PaperService(self.db, self.discovery, PDFProvider(self.files, client=self.client), web=self.service.web)
        data = resumed.acquire_pdf(self.task, target, metadata_ref=self.reference.paper_id)
        self.assertEqual(data['document_id'], result.data['document_id'])
        self.assertTrue(data['identity_handoff']['original_locator_identity_unverified'])
        self.assertEqual(len(self.paid_calls()), 2)
        self.assertEqual(self.calls.count(('GET', ORIGINAL_PDF)), 1)

    def test_exact_authorized_locator_maps_only_exact_version(self):
        """精确v1详情映射已授权ID，v2既不别名也不继承未给定URL授权。"""
        result = self.run_acquire(ARXIV, page_count=1)
        self.assertTrue(result.ok, result.model_dump())
        self.assertEqual(result.data['paper_id'], self.reference.paper_id)
        before = len(self.calls)
        denied = self.run_acquire(ARXIV.replace('v1','v2'))
        self.assertFalse(denied.ok)
        self.assertEqual(len(self.calls), before)

    def test_arbitrary_url_and_unmentioned_canonical_url_not_granted(self):
        """网页线索或未由用户给定的论文URL不能自授读取权。"""
        for url in ('https://example.org/random.pdf', 'https://arxiv.org/abs/2601.99999v1'):
            with self.subTest(url=url):
                result = self.run_acquire(url)
                self.assertFalse(result.ok)
        self.assertEqual(self.calls, [])
        rows = self.db.fetchall('SELECT paper_id FROM task_papers WHERE task_id=?', (self.task,))
        self.assertEqual([row['paper_id'] for row in rows], [self.reference.paper_id])

    def test_child_cannot_borrow_root_other_url_or_metadata(self):
        """根含A/B也不扩大只获派A的子角色材料；metadata_ref同样受限。"""
        other = 'https://openreview.net/forum?id=synthetic-other'
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('请比较 ' + ORIGINAL + ' 和 ' + other, self.task))
        child = 'handoff-child-' + uuid4().hex
        self.db.create_task(child,None,'synthetic','只分析分派材料','deny',[],parent_task_id=self.task,root_task_id=self.task)
        assigned = self.add_paper(paper_id='assigned:' + child, paper_url=ORIGINAL, source='openreview')
        self.db.grant_paper(child,assigned.paper_id,child)
        self.db.grant_paper(child,self.reference.paper_id,child)
        context = ToolContext(child,SavePolicy.deny,services={'papers':self.service,'scope':{'paper_ids':[assigned.paper_id]}})
        for arguments in ({'target_ref':other}, {'target_ref':ORIGINAL,'metadata_ref':self.reference.paper_id}):
            result = self.registry.execute('acquire_paper_pdf',arguments,context)
            self.assertFalse(result.ok)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.db.list_task_evidence(child), [])

    def test_metadata_other_task_and_explicit_version_mismatch(self):
        """其他任务题录不能作线索；同论文v2不能覆盖用户显式v1。"""
        foreign = self.add_paper(paper_id='foreign:' + uuid4().hex)
        self.db.execute('DELETE FROM task_papers WHERE task_id=? AND paper_id=?',(self.task,foreign.paper_id))
        denied = self.run_acquire(metadata_ref=foreign.paper_id)
        self.assertFalse(denied.ok)
        v2 = self.add_paper(paper_id='v2:' + self.task,paper_url=ARXIV.replace('v1','v2'),version='v2',source='arxiv')
        mismatch = self.run_acquire(ARXIV,metadata_ref=v2.paper_id)
        self.assertFalse(mismatch.ok)
        self.assertEqual(mismatch.data['error_code'],'PDF_METADATA_VERSION_MISMATCH')
        self.assertEqual(self.calls, [])

    def test_unknown_publication_source_rejected_before_identity_grant(self):
        """根明确URL也不证明CVPR发表范围；未知来源无请求且不授临时ID。"""
        self.context.services['scope'] = {'sources':['cvpr']}
        result = self.run_acquire()
        self.assertFalse(result.ok)
        self.assertEqual(result.data['error_code'],'PDF_PUBLICATION_SOURCE_UNVERIFIED')
        self.assertEqual(result.data['acquisition']['request_state'],'not_requested')
        self.assertEqual(self.calls, [])
        rows = self.db.fetchall('SELECT paper_id FROM task_papers WHERE task_id=?',(self.task,))
        self.assertEqual([row['paper_id'] for row in rows],[self.reference.paper_id])

    def test_page_range_and_external_denial_preserved(self):
        """原URL交接不绕过用户物理页范围或明确禁止外搜。"""
        for prompt, arguments in (('只读取物理第1页。' + ORIGINAL, {'start_page':2,'page_count':1}),
                                  ('Do not search external sources; use the uploaded paper only. ' + ORIGINAL, {})):
            with self.subTest(prompt=prompt):
                self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?',(prompt,self.task))
                result = self.run_acquire(**arguments)
                self.assertFalse(result.ok)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.db.list_task_evidence(self.task), [])

    def test_lease_loss_before_url_grant_and_cancel_after_pdf(self):
        """租约写前再查，下载后取消不产生任何页证据。"""
        checks = []
        def lost_before_grant():
            """第三次检查发生于URL身份持久授权之前。"""
            checks.append(True)
            return len(checks) >= 3
        self.context.cancelled = lost_before_grant
        result = self.run_acquire()
        self.assertFalse(result.ok)
        self.assertEqual(self.calls, [])
        rows = self.db.fetchall('SELECT paper_id FROM task_papers WHERE task_id=?',(self.task,))
        self.assertEqual([row['paper_id'] for row in rows],[self.reference.paper_id])
        self.context.cancelled = lambda:False
        self.original_succeeds = True
        self.cancel_after_download = True
        cancelled = self.run_acquire(page_count=1)
        self.assertFalse(cancelled.ok)
        self.assertIn(('GET',ORIGINAL_PDF),self.calls)
        self.assertEqual(self.db.list_task_evidence(self.task),[])

    def test_primary_observation_is_historical_reference_and_no_rpc_repeat(self):
        """10项旧RPC观察只读引用保留action_id，不误记本轮再次执行。"""
        source_results = [{'source':'synthetic-' + str(index),'status':'failed' if index < 7 else 'empty'} for index in range(10)]
        action_id = str(uuid4())
        self.db.save_action({'action_id':action_id,'task_id':self.task,'sequence':1,'tool_name':'search_papers',
            'arguments':{'query':'original query'},'status':'failed',
            'observation':{'ok':False,'kind':'execution_error','message':'historical','data':{'source_results':source_results}}})
        result = self.registry.execute('search_web_papers',{'query':'original query'},self.context)
        history = result.data['primary_source_observations'][0]
        self.assertEqual(history['action_id'],action_id)
        self.assertEqual(history['record_kind'],'historical_readonly_reference')
        self.assertFalse(history['executed_now'])
        self.assertTrue(history['read_only'])
        self.assertEqual(history['data']['source_results'],source_results)
        self.discovery.search.assert_not_called()
        self.assertEqual(len(self.paid_calls()),1)

    def test_web_schema_and_publication_scope_are_separate(self):
        """网页平台筛选独立于分派会议；未知发表身份不授予新候选。"""
        self.context.services['scope'] = {'sources':['icml']}
        result = self.registry.execute('search_web_papers',{'query':'research','sources':[' ARXIV.ORG ']},self.context)
        self.assertEqual(result.data['assigned_publication_sources'],['icml'])
        self.assertEqual(result.data['candidates'],[])
        self.assertEqual(result.data['results'][0]['publication_source_status'],'unverified_for_assigned_sources')
        before = len(self.paid_calls())
        invalid = self.registry.execute('search_web_papers',{'query':'research','sources':['SOSP']},self.context)
        self.assertFalse(invalid.ok)
        self.assertEqual(invalid.kind,'validation_error')
        self.assertEqual(invalid.data['request_state'],'not_requested')
        self.assertFalse(invalid.data['tool_executed'])
        self.assertEqual(len(self.paid_calls()),before)


if __name__ == '__main__':
    unittest.main()
