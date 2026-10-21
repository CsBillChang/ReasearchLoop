"""备用获取的合成HTTP/真实PDF隔离验证；禁止调用模型或真实网页。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from unittest.mock import Mock

import httpx
import pymupdf

from research_loop.config import Settings
from research_loop.domain.evidence import PaperCandidate, SearchResult, SourceResult
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.pdf import PDFProvider
from research_loop.providers.web_search import WebSearchProvider
from research_loop.providers.web_fetch import WebFetchProvider
from research_loop.services.papers import PaperService
from research_loop.services.web_acquisition import WebAcquisitionService, build_web_acquisition
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database
from research_loop.storage.web_requests import WebRequestStore
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry


TITLE = 'Synthetic Hierarchical Research'
AUTHOR = 'Synthetic Author'
ARXIV = 'https://arxiv.org/abs/2601.12345v1'
PDF_URL = 'https://arxiv.org/pdf/2601.12345v1'


def synthetic_pdf():
    """生成可公开的两页PDF；所有内容均为合成文字。"""
    document = pymupdf.open()
    for number in range(2):
        page = document.new_page()
        text = TITLE + '\n' + '\n'.join(f'{number}:{index} Synthetic methods explain a reproducible algorithm and evaluation.' for index in range(45))
        page.insert_textbox(pymupdf.Rect(30, 30, 560, 800), text, fontsize=8)
    data = document.tobytes()
    document.close()
    return data


class WebAcquisitionTests(unittest.TestCase):
    """默认SQLite，也可显式指向专用测试MySQL；网络只有MockTransport。"""

    def setUp(self):
        """每项使用独立任务身份和临时文件，不能改生产数据或旧账本。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Database(os.environ.get('RESEARCHLOOP_TEST_WEB_DATABASE_URL') or 'sqlite:///' + self.temp.name + '/web.db')
        self.db.ensure_schema()
        self.task = 'web-test-' + uuid4().hex
        self.db.create_task(self.task, None, 'synthetic', 'Find relevant papers and explain their methods.', 'deny', [])
        self.files = FileStore(Path(self.temp.name) / 'data', Path(self.temp.name) / 'out', self.temp.name)
        self.calls = []
        self.pdf_data = synthetic_pdf()
        self.fetch_content = AUTHOR + '\n[PDF](' + PDF_URL + ')'
        self.fetch_title = TITLE
        self.search_results = [{'title': TITLE, 'url': ARXIV, 'snippet': 'synthetic summary', 'summary': 'not original text', 'date': '2026-10-19T00:00:00Z'}]
        self.download_fails = False
        self.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.client.close)
        web = WebAcquisitionService(self.db, WebSearchProvider('synthetic-key', client=self.client),
                                    WebFetchProvider('synthetic-key', client=self.client))
        self.discovery = Mock()
        self.discovery.search.return_value = SearchResult(source_results=[SourceResult(source='sosp', status='failed', message='HTTP403 original')])
        self.service = PaperService(self.db, self.discovery, PDFProvider(self.files, client=self.client), web=web)
        self.context = ToolContext(self.task, SavePolicy.deny, services={'papers': self.service})
        self.registry = ToolRegistry(specs(web_search_enabled=True, web_fetch_enabled=True))

    def respond(self, request):
        """仅合成的Search/Fetch、认证失败和PDF字节；未知请求直接报错。"""
        self.calls.append((request.method, str(request.url)))
        if request.url.path == '/v1/tools/search':
            body = json.loads(request.content)
            self.assertFalse(body['need_content'])
            return httpx.Response(200, json={'results': self.search_results})
        if request.url.path == '/v1/tools/fetch':
            return httpx.Response(200, json={'title': self.fetch_title, 'content': self.fetch_content})
        if request.url.host == 'r.jina.ai':
            return httpx.Response(401, text='credential missing')
        if request.url.host == 'dl.acm.org' or (self.download_fails and request.url.host == 'arxiv.org'):
            return httpx.Response(403, text='access denied')
        if str(request.url) == PDF_URL or str(request.url) == 'https://arxiv.org/pdf/2601.12345v2':
            return httpx.Response(200, content=self.pdf_data, headers={'content-type': 'application/pdf'})
        raise AssertionError('unapproved synthetic HTTP request: ' + str(request.url))

    def add_paper(self, **changes):
        """持久化已授权的合成旧题录，不扩大其他任务权限。"""
        candidate = PaperCandidate(paper_id='synthetic:' + self.task, title=TITLE, authors=[AUTHOR], source='sosp',
                                   paper_url='https://dl.acm.org/doi/10.0000/synthetic')
        candidate = candidate.model_copy(update=changes)
        self.db.save_paper(candidate.paper_id, candidate.model_dump(mode='json'))
        self.db.grant_paper(self.task, candidate.paper_id, self.task)
        return candidate

    def paid_calls(self):
        """按实际物理POST计数，不能把生成Token填零代替费用记录。"""
        return [item for item in self.calls if item[0] == 'POST']

    def test_default_off_has_no_tools_no_requests_and_no_table_initialization(self):
        """关闭开关甚至不构建网页适配器；默认schema不包含备用工具。"""
        db = Mock()
        self.assertIsNone(build_web_acquisition(db, Settings(project_root=Path(self.temp.name), data_dir=Path(self.temp.name) / 'data', output_dir=Path(self.temp.name) / 'out')))
        db.execute.assert_not_called()
        self.assertNotIn('search_web_papers', ToolRegistry(specs()).names())
        self.assertNotIn('acquire_paper_pdf', ToolRegistry(specs()).names())
        self.assertEqual(self.calls, [])

    def test_empty_primary_dated_query_can_select_candidate_and_parse_pdf(self):
        """原RPC失败独立保留；有日期的备用候选可读，但不得宣称已通过发表日期筛选。"""
        primary = self.registry.execute('search_papers', {'query': 'research', 'sources': ['sosp']}, self.context)
        self.assertFalse(primary.ok)
        self.db.save_action({'action_id': str(uuid4()), 'task_id': self.task, 'sequence': 1, 'tool_name': 'search_papers', 'arguments': {'query': 'research'}, 'status': 'failed', 'observation': primary.model_dump(mode='json')})
        result = self.registry.execute('search_web_papers', {'query': 'research', 'date_from': '2023-01-01'}, self.context)
        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.data['date_filter_status'], 'publication_date_unverified')
        self.assertEqual(result.data['primary_source_observations'][0]['data']['source_results'][0]['status'], 'failed')
        candidate = result.data['candidates'][0]
        self.assertIsNone(candidate['published'])
        self.assertTrue(candidate['metadata']['publication_date_unverified'])
        read = self.registry.execute('read_paper', {'target_ref': candidate['paper_id'], 'page_count': 1}, self.context)
        self.assertTrue(read.ok, read.message)
        self.assertEqual(read.data['returned_pages'], [1])
        self.assertEqual(read.data['read_coverage']['locator_kind'], 'physical_page')
        self.assertNotIn('not original text', read.data['pages'][0]['text'])
        self.assertEqual(len(self.paid_calls()), 1)
        self.assertEqual(len([url for method, url in self.calls if 'r.jina.ai' in url]), 0)

    def test_blocked_publisher_to_verified_alternate_pdf_is_separate_version(self):
        """ACM拒绝后只取公开官方替代；独立版本/原失败与物理证据均保留。"""
        candidate = self.add_paper()
        result = self.registry.execute('acquire_paper_pdf', {'target_ref': candidate.paper_id}, self.context)
        self.assertTrue(result.ok, result.message)
        self.assertTrue(result.data['alternate_version'])
        self.assertNotEqual(result.data['paper_id'], candidate.paper_id)
        self.assertEqual(result.data['requested_target_ref'], candidate.paper_id)
        self.assertEqual(result.data['returned_pages'], [1, 2])
        self.assertEqual(result.data['acquisition']['primary_failure']['error_code'], 'PDF_HTTP_ERROR')
        self.assertFalse(result.data['acquisition']['fallback']['same_version_confirmed'])
        self.assertEqual(len(self.paid_calls()), 2)
        self.assertEqual(len([url for method, url in self.calls if 'r.jina.ai' in url]), 1)
        original = self.db.get_paper(candidate.paper_id)['metadata']
        self.assertEqual(original['paper_url'], candidate.paper_url)
        # 另一服务实例模拟恢复，Search与Fetch的已付费成功结果不重复发送。
        resumed = PaperService(self.db, self.discovery, PDFProvider(self.files, client=self.client), web=self.service.web)
        second = resumed.acquire_pdf(self.task, candidate.paper_id)
        self.assertEqual(second['document_id'], result.data['document_id'])
        self.assertEqual(len(self.paid_calls()), 2)

    def test_final_redirect_identity_and_explicit_host_before_page_grant(self):
        """Fetch返回v1链接仍须核对真实PDF终点；明确异论文/版本或域名不得授页。"""
        cases = (
            ('https://arxiv.org/pdf/2601.12345v2','只使用指定版本v1。','paper_or_version_mismatch'),
            ('https://arxiv.org/pdf/2601.99999v1','只使用指定版本v1。','paper_or_version_mismatch'),
            ('https://export.arxiv.org/pdf/2601.12345v1','只使用指定版本v1。Only use arxiv.org for downloads.','outside_explicit_host_scope'),
            ('https://export.arxiv.org/pdf/2601.12345v1','只使用指定版本v1。',None),
        )
        for final_url,prompt,error in cases:
            with self.subTest(final_url=final_url,prompt=prompt):
                task = 'web-test-' + uuid4().hex
                self.db.create_task(task,None,'synthetic',prompt,'deny',[])
                candidate = self.add_paper(paper_url=ARXIV,version='v1',source='arxiv')
                self.db.grant_paper(task,candidate.paper_id,task)
                attempts = []
                def handler(request):
                    """首取404，备用PDF真实302；全部字节均为合成PDF。"""
                    if request.url.host == 'r.jina.ai':
                        return httpx.Response(200,text='short abstract only')
                    if str(request.url) == PDF_URL:
                        attempts.append(True)
                        if len(attempts) == 1:
                            return httpx.Response(404,text='not found')
                        return httpx.Response(302,headers={'location':final_url})
                    if str(request.url) == final_url:
                        return httpx.Response(200,content=self.pdf_data,headers={'content-type':'application/pdf'})
                    raise AssertionError('unexpected synthetic PDF URL: ' + str(request.url))
                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    service = PaperService(self.db,self.discovery,PDFProvider(self.files,client=client),web=self.service.web)
                    context = ToolContext(task,SavePolicy.deny,services={'papers':service})
                    result = self.registry.execute('acquire_paper_pdf',{'target_ref':candidate.paper_id},context)
                if error:
                    self.assertFalse(result.ok,result.data)
                    self.assertEqual(result.data['error_code'],'PDF_FINAL_LOCATOR_MISMATCH')
                    self.assertFalse(result.data['acquisition']['fallback']['same_version_confirmed'])
                    final = result.data['acquisition']['pdf_failure']['final_locator_validation']
                    self.assertEqual(final['status'],error)
                    self.assertEqual(final['final_url'],final_url)
                    self.assertEqual(self.db.list_task_evidence(task),[])
                else:
                    self.assertTrue(result.ok,result.message)
                    self.assertTrue(result.data['acquisition']['fallback']['same_version_confirmed'])
                    self.assertEqual(result.data['acquisition']['pdf']['final_locator_validation']['status'],'same_official_locator')

    def test_redirect_mismatch_cannot_reenter_via_url_cache(self):
        """URL缓存恢复实际下载终点；首次拒绝的异版本不能第二次变成合法原文。"""
        from research_loop.providers.paper_acquisition import PaperAcquisitionError
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?',('只使用指定版本v1。',self.task))
        candidate = self.add_paper(paper_url=ARXIV,version='v1',source='arxiv')
        downloads = []
        def handler(request):
            """直接原路径转去v2；第二次应读缓存且仍拒绝授权。"""
            if request.url.host == 'r.jina.ai':
                return httpx.Response(200,text='short abstract only')
            downloads.append(str(request.url))
            if str(request.url) == PDF_URL:
                return httpx.Response(302,headers={'location':'https://arxiv.org/pdf/2601.12345v2'})
            return httpx.Response(200,content=self.pdf_data,headers={'content-type':'application/pdf'})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            pdf = PDFProvider(self.files,client=client)
            service = PaperService(self.db,self.discovery,pdf,web=self.service.web)
            for _ in range(2):
                with self.assertRaises(PaperAcquisitionError) as raised:
                    service.read(self.task,candidate.paper_id)
                self.assertEqual(raised.exception.code,'PDF_FINAL_LOCATOR_MISMATCH')
                self.assertEqual(raised.exception.acquisition['final_locator_validation']['status'],'paper_or_version_mismatch')
                self.assertEqual(self.db.list_task_evidence(self.task),[])
            self.assertEqual(pdf.last_fetch['kind'],'text_cache')
        self.assertEqual(len(downloads),2)

    def test_jina_401_does_not_block_original_site_404_fallback_fetch(self):
        """Jina认证失败只属于代理；原站404仍允许合法同版本详情Fetch。"""
        candidate = self.add_paper(paper_url=ARXIV,version='v1',source='arxiv')
        attempts = []
        def handler(request):
            """仅首个原站PDF404，其他请求继续使用原有合成HTTP响应。"""
            if str(request.url) == PDF_URL:
                attempts.append(True)
                if len(attempts) == 1:
                    return httpx.Response(404,text='not found')
            return self.respond(request)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            service = PaperService(self.db,self.discovery,PDFProvider(self.files,client=client),web=self.service.web)
            context = ToolContext(self.task,SavePolicy.deny,services={'papers':service})
            result = self.registry.execute('acquire_paper_pdf',{'target_ref':candidate.paper_id},context)
        self.assertTrue(result.ok,result.data)
        primary = result.data['acquisition']['primary_failure']['acquisition']
        self.assertEqual(primary['status'],404)
        self.assertEqual(primary['jina_attempts'][0]['http_status'],401)
        self.assertEqual(len(primary['jina_attempts']),1)
        self.assertTrue(any('/v1/tools/fetch' in url for method,url in self.calls))
        self.assertTrue(result.data['pages'])

    def test_known_access_denial_is_reused_before_backup_and_recovery(self):
        """实际read工具已返回403时，备用工具和恢复不得原样重取拒绝地址。"""
        candidate = self.add_paper()
        arguments = {'target_ref':candidate.paper_id}
        original = self.registry.execute('read_paper',arguments,self.context)
        self.assertFalse(original.ok)
        self.assertEqual(original.data['acquisition']['status'],403)
        self.db.save_action({'action_id':str(uuid4()),'task_id':self.task,'sequence':1,
            'tool_name':'read_paper','arguments':arguments,'status':'failed','observation':original.model_dump(mode='json')})
        denied_requests = len(self.calls)
        first = self.registry.execute('acquire_paper_pdf',arguments,self.context)
        self.assertTrue(first.ok,first.message)
        self.assertIn('reused_failure_action',first.data['acquisition']['primary_failure'])
        self.assertEqual(len([url for _,url in self.calls if 'dl.acm.org' in url]),denied_requests)
        resumed = PaperService(self.db,self.discovery,PDFProvider(self.files,client=self.client),web=self.service.web)
        second = resumed.acquire_pdf(self.task,candidate.paper_id)
        self.assertTrue(second['pages'])
        self.assertEqual(len([url for _,url in self.calls if 'dl.acm.org' in url]),denied_requests)
        self.assertEqual(len(self.paid_calls()),2)

    def test_paid_success_survives_later_pdf_failure(self):
        """下载后续失败不能使成功Search/Fetch重发计费。"""
        candidate = self.add_paper()
        self.download_fails = True
        first = self.registry.execute('acquire_paper_pdf', {'target_ref': candidate.paper_id}, self.context)
        self.assertFalse(first.ok)
        self.assertEqual(first.data['error_code'], 'PDF_HTTP_ERROR')
        self.assertEqual(len(self.paid_calls()), 2)
        self.download_fails = False
        second = self.registry.execute('acquire_paper_pdf', {'target_ref': candidate.paper_id}, self.context)
        self.assertTrue(second.ok, second.message)
        self.assertEqual(len(self.paid_calls()), 2)

    def test_no_external_policy_covers_primary_and_backup_but_attachment_presence_does_not(self):
        """当前原指令禁止外搜时两个入口均拒绝；附件存在不会自动禁止。"""
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('仅使用附件，禁止外部搜索。', self.task))
        for name, arguments in [('search_papers', {'query': 'research'}), ('search_web_papers', {'query': 'research'})]:
            result = self.registry.execute(name, arguments, self.context)
            self.assertFalse(result.ok)
            self.assertIn('forbid external', result.message)
        self.discovery.search.assert_not_called()
        self.assertEqual(self.calls, [])
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('Use this attachment and find relevant papers.', self.task))
        self.context.attachments = {'attachment': {'local_path': '/synthetic.pdf'}}
        self.assertTrue(self.registry.execute('search_web_papers', {'query': 'research'}, self.context).ok)

    def test_exact_version_and_page_restrictions_do_not_substitute(self):
        """已指定版本或物理页不允许换载体；不会发送Fetch来绕过该限制。"""
        for prompt in ('只使用指定版本v1。', '只读物理第1页。'):
            with self.subTest(prompt=prompt):
                self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', (prompt, self.task))
                candidate = self.add_paper()
                result = self.registry.execute('acquire_paper_pdf', {'target_ref': candidate.paper_id}, self.context)
                self.assertFalse(result.ok)
                self.assertEqual(result.data['status'], 'unconfirmed')
        self.assertFalse(any('/fetch' in url for method, url in self.calls))

    def test_unconfirmed_identity_challenge_and_summary_never_create_evidence(self):
        """不相似题名/作者、挑战页和仅摘要都只能形成诊断。"""
        candidate = self.add_paper()
        for content in ('Another Author\n[PDF](' + PDF_URL + ')', 'Verify you are human [PDF](' + PDF_URL + ')', AUTHOR + '\nAbstract only'):
            with self.subTest(content=content):
                # 每种服务响应单独任务，不能篡改已有完成收据。
                task = 'web-test-' + uuid4().hex
                self.db.create_task(task, None, 'synthetic', 'Find this paper.', 'deny', [])
                self.db.grant_paper(task, candidate.paper_id, task)
                self.fetch_content = content
                result = self.service.acquire_pdf(task, candidate.paper_id)
                self.assertEqual(result['status'], 'unconfirmed')
                self.assertEqual(self.db.list_task_evidence(task), [])

    def test_unknown_started_and_known_http_failure_are_not_resent(self):
        """超时未知、崩溃started与HTTP403失败分别保留；都不能盲重发。"""
        store = WebRequestStore(self.db)
        from research_loop.providers.paper_acquisition import PaperAcquisitionError
        for code in ('timeout', '403'):
            calls = []
            def send():
                """注入一次未知传输或已知HTTP失败。"""
                calls.append('sent')
                if code == 'timeout':
                    raise httpx.ReadTimeout('synthetic')
                raise PaperAcquisitionError('WEB_HTTP_ERROR', 'HTTP403', acquisition={'http_status': 403})
            for _ in range(2):
                with self.assertRaises(PaperAcquisitionError):
                    store.run(self.task, 'search', {'query': code}, send, lambda: True)
            self.assertEqual(len(calls), 1)
            row = self.db.fetchone('SELECT * FROM web_request_receipts WHERE root_task_id=? AND request_json=?',
                                  (self.task, json.dumps({'query': code}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))))
            self.assertEqual(row['status'], 'unknown' if code == 'timeout' else 'failed')
        # 未发送失权有单独状态，不标记为供应商未知。
        active = iter([True, False])
        with self.assertRaises(PermissionError):
            store.run(self.task, 'fetch', {'url': ARXIV}, lambda: self.fail('must not send'), lambda: next(active))
        row = self.db.fetchone('SELECT * FROM web_request_receipts WHERE root_task_id=? AND operation=?', (self.task, 'fetch'))
        self.assertEqual(row['status'], 'not_sent')
        self.db.execute('UPDATE web_request_receipts SET status=? WHERE receipt_id=?', ('started', row['receipt_id']))
        with self.assertRaises(PaperAcquisitionError):
            store.run(self.task, 'fetch', {'url': ARXIV}, lambda: self.fail('must not resend'), lambda: True)

    def test_permission_word_order_and_venue_paths(self):
        """常用中英文禁搜语序和同域名不同会议均不能放宽授权。"""
        from research_loop.domain.source_access import assert_external_access, source_allowed
        for prompt in ('Do not search the web. Answer only from this PDF.', '仅使用这份PDF，不要搜索网络。',
                       'Only use attachments.', 'Use only the uploaded PDF.'):
            self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', (prompt, self.task))
            with self.subTest(prompt=prompt), self.assertRaises(PermissionError):
                assert_external_access(self.db, self.task)
        self.assertFalse(source_allowed('https://openaccess.thecvf.com/content/ICCV2025/html/Author_Synthetic_paper.html', ['CVPR']))
        self.assertTrue(source_allowed('https://openaccess.thecvf.com/content/CVPR2026/html/Author_Synthetic_paper.html', ['CVPR']))
        self.assertEqual(self.calls, [])

    def test_explicit_provided_material_and_external_search_phrases(self):
        """明确仅给定论文/不外搜的同类措辞阻断原检索与备用检索，不泛化任意否定。"""
        denied = (
            '不要检索外部文献，只分析我给的论文。',
            '只根据已提供的论文回答，不要去网上找。',
            'Do not search external sources; use the uploaded paper only.',
            '禁止额外查找外部论文，基于已有材料解释。',
            '仅参考我提供的文献。',
            "Don't search for external references; analyze the provided document only.",
        )
        for prompt in denied:
            self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', (prompt,self.task))
            for name in ('search_papers','search_web_papers'):
                with self.subTest(prompt=prompt,tool=name):
                    result = self.registry.execute(name,{'query':'research'},self.context)
                    self.assertFalse(result.ok)
                    self.assertIn('forbid external',result.message)
        self.discovery.search.assert_not_called()
        self.assertEqual(self.calls,[])
        from research_loop.domain.source_access import assert_external_access
        for prompt in ('不要重新精读，找两篇相关论文。', '不要改写标题，允许搜索相关论文。',
                       'Do not rewrite the summary; search for related papers.',
                       '不要检索外部变量的名字；寻找相关论文。',
                       'Do not search external variables; find related papers.'):
            self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', (prompt,self.task))
            with self.subTest(allowed_prompt=prompt):
                self.assertEqual(assert_external_access(self.db,self.task),self.task)
        self.assertEqual(self.calls,[])

    def test_detail_cannot_grant_unrelated_pdf_even_from_allowed_host(self):
        """同站其他论文及跨站PDF都不因页面标题正确取得身份授权。"""
        detail = 'https://openaccess.thecvf.com/content/CVPR2026/html/Author_Synthetic_paper.html'
        candidate = self.add_paper(source='cvpr', paper_url=detail)
        self.search_results = [{'title': TITLE, 'url': detail}]
        for wrong in ('https://openreview.net/pdf?id=DIFFERENT',
                      'https://openaccess.thecvf.com/content/CVPR2026/papers/Other_Different_paper.pdf'):
            task = 'web-test-' + uuid4().hex
            self.db.create_task(task, None, 'synthetic', 'Only use openaccess.thecvf.com for this paper.', 'deny', [])
            self.db.grant_paper(task, candidate.paper_id, task)
            self.fetch_content = AUTHOR + '\n[PDF](' + wrong + ')'
            result = self.service.web.locate_pdf(task, candidate)
            self.assertIsNone(result['pdf_url'])
            self.assertTrue(any(item['status'] in {'detail_pdf_identity_mismatch', 'no_unique_pdf'} for item in result['diagnostics']))
            self.assertEqual(self.db.list_task_evidence(task), [])

    def test_known_arxiv_title_wrapper_and_response_resource_guard(self):
        """只剥离当前arXiv ID外壳；响应字节上限是显式网络资源保护。"""
        candidate = self.add_paper()
        self.fetch_title = '[2601.12345v1] ' + TITLE + ' | arXiv'
        result = self.service.acquire_pdf(self.task, candidate.paper_id)
        self.assertTrue(result['pages'])
        from research_loop.providers.paper_acquisition import PaperAcquisitionError
        provider = WebSearchProvider('synthetic-key', client=self.client, max_response_bytes=16)
        with self.assertRaises(PaperAcquisitionError) as raised:
            provider.request(provider.payload('new synthetic query'))
        self.assertEqual(raised.exception.code, 'WEB_RESPONSE_TOO_LARGE')
        self.assertEqual(raised.exception.acquisition['max_response_bytes'], 16)

    def test_competing_receipt_sends_once_and_root_cancellation_blocks_request(self):
        """并发占位只有赢家发送；父任务取消也禁止子任务新增付费请求。"""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from research_loop.providers.paper_acquisition import PaperAcquisitionError
        entered, release = Event(), Event()
        sends = []
        def send():
            """在发送窗口保留并发竞争，不能依赖线程执行顺序。"""
            sends.append(True)
            entered.set()
            release.wait(3)
            return {'results': []}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.service.web.receipts.run, self.task, 'search', {'query': 'concurrent'}, send, lambda: True)
            self.assertTrue(entered.wait(3))
            try:
                with self.assertRaises(PaperAcquisitionError):
                    self.service.web.receipts.run(self.task, 'search', {'query': 'concurrent'}, send, lambda: True)
            finally:
                release.set()
            self.assertFalse(first.result()['reused'])
        self.assertEqual(sends, [True])
        child = 'web-test-' + uuid4().hex
        self.db.create_task(child, None, 'synthetic', 'Find papers.', 'deny', [], parent_task_id=self.task, root_task_id=self.task)
        self.db.update_task(self.task, cancel_requested=1)
        with self.assertRaises(PaperAcquisitionError) as cancelled:
            self.service.search_web(child, 'blocked after parent cancellation')
        self.assertEqual(cancelled.exception.code, 'WEB_REQUEST_NOT_SENT')
        self.assertEqual(cancelled.exception.acquisition['request_state'], 'not_sent')
        self.assertIsNone(cancelled.exception.acquisition['receipt_id'])
        self.assertNotIn('http_status', cancelled.exception.acquisition)
        self.assertEqual(self.calls, [])

    def test_official_parameter_and_role_boundaries(self):
        """Unicode字符/50条、HTTP无凭据及Supervisor角色使用同一工具合同。"""
        self.assertEqual(WebSearchProvider.payload('汉' * 128, 50)['count'], 50)
        for query, count in [(' ', 1), ('a' * 129, 1), ('valid', 51)]:
            with self.assertRaises(ValueError):
                WebSearchProvider.payload(query, count)
        for url in ('https://name:secret@example.com/paper', 'file:///tmp/paper', 'https://127.0.0.1/paper'):
            with self.assertRaises(ValueError):
                WebFetchProvider.payload(url)
        self.context.role = 'researcher'
        denied = self.registry.execute('search_web_papers', {'query': 'research'}, self.context)
        self.assertFalse(denied.ok)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
