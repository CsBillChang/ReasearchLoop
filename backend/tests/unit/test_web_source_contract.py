"""网页来源筛选与失败归因合同；全部HTTP为替身，默认临时SQLite，可显式使用专用测试MySQL。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import httpx
from pydantic import TypeAdapter

from research_loop.domain.evidence import PaperCandidate
from research_loop.domain.source_access import (
    SOURCE_HOSTS, WebSource, WebSourceValidationError, normalize_web_sources, source_allowed,
)
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.providers.web_fetch import WebFetchProvider
from research_loop.providers.web_search import WebSearchProvider
from research_loop.services.web_acquisition import WebAcquisitionService
from research_loop.storage.mysql import Database


ARXIV = 'https://arxiv.org/abs/2601.12345v1'
PDF = 'https://arxiv.org/pdf/2601.12345v1'
OPENREVIEW = 'https://openreview.net/forum?id=SyntheticPaper'


class WebSourceContractTests(unittest.TestCase):
    """直接服务入口也必须发送前校验，不能依赖模型或工具schema自律。"""

    def setUp(self):
        """不读生产配置、真实模型或来源，只有独立测试任务与显式HTTP替身。"""
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.db = Database(os.environ.get('RESEARCHLOOP_TEST_WEB_DATABASE_URL') or 'sqlite:///' + str(Path(tmp.name)/'sources.db'))
        self.db.ensure_schema()
        self.task = 'source-test-' + uuid4().hex
        self.db.create_task(self.task, None, 'synthetic', 'Find this paper using allowed public sources.', 'deny', [])
        self.calls = []
        self.search_results = [{'url': OPENREVIEW, 'title': 'Synthetic Paper'}]
        self.search_status = self.fetch_status = 200
        self.search_unknown = False
        self.fetch_content = 'Synthetic Author\n[PDF](' + PDF + ')'
        self.http = httpx.Client(transport=httpx.MockTransport(self.respond)); self.addCleanup(self.http.close)
        self.service = WebAcquisitionService(self.db, WebSearchProvider('fixture', client=self.http),
                                             WebFetchProvider('fixture', client=self.http))

    def respond(self, request):
        """记录每次实际物理POST；所有地址均通过MockTransport本地返回。"""
        self.calls.append((str(request.url), json.loads(request.content)))
        if request.url.path.endswith('/search'):
            if self.search_unknown:
                raise httpx.ReadTimeout('synthetic response unknown', request=request)
            return httpx.Response(self.search_status, json={'results': self.search_results})
        if request.url.path.endswith('/fetch'):
            return httpx.Response(self.fetch_status, json={'title': 'Synthetic Paper', 'content': self.fetch_content})
        self.fail('unexpected synthetic request')

    def receipts(self):
        """只读现有独立收据，不补写usage或请求。"""
        return self.db.fetchall('SELECT receipt_id,operation,status,diagnostic_json FROM web_request_receipts WHERE root_task_id=? ORDER BY operation,receipt_id', (self.task,))

    def candidate(self, **changes):
        """元数据仅描述当前已知论文，不授予任何跨论文替换权限。"""
        return PaperCandidate(paper_id='paper', title='Synthetic Paper', authors=['Synthetic Author'],
                              source='arxiv', paper_url=ARXIV, version='v1').model_copy(update=changes)

    def test_shared_enum_aliases_and_host_mapping_match(self):
        """schema值全部有实际映射，平台OpenReview与会议来源分开。"""
        self.assertEqual({source.value for source in WebSource}, set(SOURCE_HOSTS))
        schema = TypeAdapter(WebSource).json_schema()
        self.assertEqual(set(schema['enum']), set(SOURCE_HOSTS))
        self.assertEqual(normalize_web_sources([' OpenReview.NET ', 'OPENREVIEW', 'arXiv.org', 'NIPS']),
                         ['openreview', 'arxiv', 'neurips'])
        self.assertTrue(source_allowed(OPENREVIEW, ['OpenReview']))
        self.assertFalse(source_allowed(ARXIV, ['openreview']))
        for invalid in [['sosp'], ['unknown-source'], [None], 'openreview']:
            with self.subTest(invalid=invalid), self.assertRaises(WebSourceValidationError):
                normalize_web_sources(invalid)

    def test_unknown_source_direct_service_rejected_before_paid_request(self):
        """原RPC合法的SOSP没有网页filter映射，应明确本地拒绝而非先付费再过滤。"""
        with self.assertRaises(PaperAcquisitionError) as raised:
            self.service.search(self.task, 'synthetic topic', sources=['sosp'])
        self.assertEqual(raised.exception.code, 'WEB_SOURCE_INVALID')
        details = raised.exception.acquisition
        self.assertEqual(details['stage'], 'parameter_validation')
        self.assertEqual(details['request_state'], 'not_requested')
        self.assertIsNone(details['receipt_id'])
        self.assertNotIn('http_status', details)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.receipts(), [])

    def test_query_validation_has_no_request_or_receipt(self):
        """查询长度错误同样是参数错误，不归因为来源站点不可用。"""
        with self.assertRaises(PaperAcquisitionError) as raised:
            self.service.search(self.task, 'x'*129, sources=['openreview'])
        self.assertEqual(raised.exception.acquisition['stage'], 'parameter_validation')
        self.assertEqual(raised.exception.acquisition['request_state'], 'not_requested')
        self.assertEqual(self.calls, [])
        self.assertEqual(self.receipts(), [])

    def test_openreview_platform_filter_accepts_results_and_reuses_receipt(self):
        """平台别名可用，筛选不改变原查询，也不重复支付同一Search。"""
        first = self.service.search(self.task, 'Original query unchanged', sources=['OpenReview.net'])
        second = self.service.search(self.task, 'Original query unchanged', sources=['openreview'])
        self.assertEqual(first['sources'], ['openreview'])
        self.assertEqual(first['results'][0]['paper_identity'], ['openreview', 'SyntheticPaper', None])
        self.assertEqual(first['diagnostic']['stage'], 'search')
        self.assertEqual(first['diagnostic']['request_state'], 'completed')
        self.assertEqual(second['receipt']['receipt_id'], first['receipt']['receipt_id'])
        self.assertTrue(second['receipt']['reused'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1]['query'], 'Original query unchanged')

    def test_tool_filter_never_replaces_explicit_user_domain_scope(self):
        """工具自己的结果过滤与用户域名授权取交集，不因openreview合法而扩大访问。"""
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', ('Only use arxiv.org for this research.', self.task))
        self.search_results = [{'url': OPENREVIEW, 'title': 'Synthetic Paper'}, {'url': ARXIV, 'title': 'Synthetic Paper'}]
        result = self.service.search(self.task, 'domain scope', sources=['openreview'])
        self.assertEqual(result['status'], 'filtered_empty')
        reasons = {item['url']: item['reason'] for item in result['rejected']}
        self.assertEqual(reasons[OPENREVIEW], 'outside_explicit_host_scope')
        self.assertEqual(reasons[ARXIV], 'outside_tool_source_filter')
        self.assertEqual(result['diagnostic']['explicit_host_scope'], ['arxiv.org'])
        without_filter = self.service.search(self.task, 'domain scope', sources=None)
        self.assertEqual([item['url'] for item in without_filter['results']], [ARXIV])
        self.assertEqual(len(self.calls), 1)

    def test_empty_and_all_filtered_are_different_completed_search_outcomes(self):
        """远端空返回与本地全部筛除均保留Search成功收据，但原因不同。"""
        self.search_results = []
        empty = self.service.search(self.task, 'remote empty', sources=['openreview'])
        self.search_results = [{'url': ARXIV, 'title': 'Synthetic Paper'}]
        filtered = self.service.search(self.task, 'local filter empty', sources=['openreview'])
        self.assertEqual((empty['status'], empty['diagnostic']['stage']), ('empty', 'search'))
        self.assertEqual((filtered['status'], filtered['diagnostic']['stage']), ('filtered_empty', 'filtered_empty'))
        self.assertEqual((empty['diagnostic']['received_count'], filtered['diagnostic']['received_count']), (0, 1))
        self.assertEqual([row['status'] for row in self.receipts()], ['completed', 'completed'])
        self.assertNotIn('http_status', filtered['diagnostic'])

    def test_known_search_http_failure_keeps_its_operation_and_single_receipt(self):
        """Search真实HTTP500不与原检索500混算，恢复只引用本次失败收据。"""
        self.search_status = 500
        errors = []
        for _ in range(2):
            with self.assertRaises(PaperAcquisitionError) as raised:
                self.service.search(self.task, 'known HTTP failure', sources=['openreview'])
            errors.append(raised.exception)
        self.assertEqual(errors[0].acquisition['http_status'], 500)
        self.assertEqual(errors[0].acquisition['stage'], 'search')
        self.assertEqual(errors[1].acquisition['request_state'], 'failed')
        self.assertEqual(errors[1].acquisition['http_status'], 500)
        self.assertEqual(errors[1].acquisition['original_error_code'], 'WEB_HTTP_ERROR')
        self.assertEqual(errors[0].acquisition['receipt_id'], errors[1].acquisition['receipt_id'])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.receipts()), 1)
        self.assertEqual(self.receipts()[0]['status'], 'failed')

    def test_unknown_search_response_stays_unknown_without_resend(self):
        """超时未知不伪装空结果、HTTP403或站点挑战，也不重复发送。"""
        self.search_unknown = True
        for _ in range(2):
            with self.assertRaises(PaperAcquisitionError) as raised:
                self.service.search(self.task, 'unknown response', sources=['openreview'])
            self.assertEqual(raised.exception.acquisition['stage'], 'search')
            self.assertEqual(raised.exception.acquisition['request_state'], 'unknown')
            self.assertNotIn('http_status', raised.exception.acquisition)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.receipts()[0]['status'], 'unknown')

    def test_failed_queries_keep_case_sensitive_receipt_identity_on_mysql(self):
        """MySQL文本排序规则不能把两个精确请求误归为同一失败收据。"""
        self.search_status = 500
        receipt_ids = []
        for query in ['Synthetic Query CASE', 'synthetic query case']:
            with self.assertRaises(PaperAcquisitionError) as raised:
                self.service.search(self.task, query, sources=['openreview'])
            receipt_ids.append(raised.exception.acquisition['receipt_id'])
        self.assertEqual(len(set(receipt_ids)), 2)
        self.assertEqual(set(receipt_ids), {row['receipt_id'] for row in self.receipts()})
        self.assertEqual(len(self.calls), 2)

    def test_disabled_search_is_not_a_request_failure(self):
        """未启用组件不生成HTTP失败或收据，明确没有外部请求。"""
        self.service.search_provider = None
        with self.assertRaises(PaperAcquisitionError) as raised:
            self.service.search(self.task, 'disabled search', sources=['openreview'])
        self.assertEqual(raised.exception.code, 'WEB_SEARCH_DISABLED')
        self.assertEqual(raised.exception.acquisition['request_state'], 'not_requested')
        self.assertEqual(self.calls, [])
        self.assertEqual(self.receipts(), [])

    def test_missing_authentication_stays_not_sent_after_recovery(self):
        """本地未配置认证不能被恢复包装成远端HTTP403或未知计费发送。"""
        self.service.search_provider.api_key = None
        for _ in range(2):
            with self.assertRaises(PaperAcquisitionError) as raised:
                self.service.search(self.task, 'unconfigured search', sources=['openreview'])
            self.assertEqual(raised.exception.acquisition['request_state'], 'not_sent')
            self.assertEqual(raised.exception.acquisition['original_error_code'], 'WEB_NOT_CONFIGURED')
            self.assertNotIn('http_status', raised.exception.acquisition)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.receipts()), 1)

    def test_fetch_failure_preserves_successful_search_and_primary_pdf_reason(self):
        """已成功Search、Fetch500与原PDF403分别归因，原失败不冒认Search失败。"""
        self.search_results = [{'url': ARXIV, 'title': 'Synthetic Paper'}]
        self.fetch_status = 500
        primary = {'error_code': 'PDF_HTTP_ERROR', 'acquisition': {'http_status': 403,
                   'requested_source': 'https://publisher.invalid/main.pdf'}}
        with self.assertRaises(PaperAcquisitionError) as raised:
            self.service.locate_pdf(self.task, self.candidate(), primary_failure=primary)
        details = raised.exception.acquisition
        self.assertEqual((details['stage'], details['request_state'], details['http_status']), ('fetch', 'failed', 500))
        self.assertEqual(details['primary_failure'], primary)
        self.assertEqual(details['search']['diagnostic']['request_state'], 'completed')
        self.assertNotEqual(details['receipt_id'], details['search']['receipt']['receipt_id'])
        self.assertEqual({row['operation']: row['status'] for row in self.receipts()}, {'search': 'completed', 'fetch': 'failed'})

    def test_identity_rejection_cites_fetch_receipt_without_claiming_http_failure(self):
        """HTTP已成功但标题/主文身份不符时是本地核对拒绝，不虚构挑战页。"""
        self.search_results = [{'url': ARXIV, 'title': 'Synthetic Paper'}]
        self.fetch_content = 'Synthetic Author\n[PDF](https://arxiv.org/pdf/2601.99999v1)'
        result = self.service.locate_pdf(self.task, self.candidate())
        self.assertIsNone(result['pdf_url'])
        rejection = next(item for item in result['diagnostics'] if item['status'] == 'detail_pdf_identity_mismatch')
        self.assertEqual((rejection['stage'], rejection['request_state']), ('identity_validation', 'completed'))
        self.assertIsNotNone(rejection['receipt_id'])
        self.assertNotIn('http_status', rejection)
        self.assertEqual(result['diagnostic']['stage'], 'identity_validation')
        self.assertIsNone(result['diagnostic']['operation'])
        self.assertIsNone(result['diagnostic']['receipt_id'])
        self.assertEqual(result['diagnostic']['search_receipt_id'], result['search']['receipt']['receipt_id'])
        self.assertEqual(result['diagnostic']['fetch_receipt_ids'], [rejection['receipt_id']])
        self.assertEqual(result['diagnostic']['fetch_request_state'], 'completed')
        self.assertTrue(all(row['status'] == 'completed' for row in self.receipts()))

    def test_lease_loss_after_receipt_reservation_keeps_not_sent_diagnostic(self):
        """占位后失权不发POST，工具层仍能取得取消归因和原收据，恢复不重发。"""
        checks = []
        def cancelled():
            """第一次允许占位，第二次代表发送前租约已经丢失。"""
            checks.append(True)
            return len(checks) > 1
        for callback in [cancelled, lambda: False]:
            with self.assertRaises(PaperAcquisitionError) as raised:
                self.service.search(self.task, 'lease lost before POST', cancelled=callback)
            details = raised.exception.acquisition
            self.assertEqual(raised.exception.code, 'WEB_REQUEST_NOT_SENT')
            self.assertEqual((details['stage'], details['request_state']), ('search', 'not_sent'))
            self.assertIsNotNone(details['receipt_id'])
            self.assertEqual(details['receipt_status'], 'not_sent')
            self.assertNotIn('http_status', details)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.receipts()[0]['status'], 'not_sent')

    def test_known_sosp_candidate_keeps_existing_cross_platform_alternate(self):
        """SOSP发表来源不是网页filter；合法同论文arXiv载体仍按原身份合同核对。"""
        self.search_results = [{'url': ARXIV, 'title': 'Synthetic Paper'}]
        result = self.service.locate_pdf(self.task, self.candidate(source='sosp',
            paper_url='https://publisher.invalid/doi/10.0000/synthetic', version=None), sources=['sosp'])
        self.assertEqual(result['status'], 'located')
        self.assertTrue(result['alternate_version'])
        self.assertEqual(result['search']['sources'], None)
        self.assertEqual(result['search']['requested_paper_sources'], ['sosp'])
        self.assertEqual(result['pdf_url'], PDF)
        self.assertEqual(len(self.calls), 2)

    def test_provisional_metadata_reference_is_not_original_locator_confirmation(self):
        """metadata_ref只验证候选替代文档，原URL未读时关联仍保留未核实标记。"""
        self.search_results = [{'url': ARXIV, 'title': 'Synthetic Paper'}]
        reference = {'paper_id': 'returned-metadata', 'canonical_locator': ARXIV}
        candidate = self.candidate(paper_url=OPENREVIEW, version=None, metadata={
            'original_locator_identity_unverified': True, 'metadata_reference': reference})
        result = self.service.locate_pdf(self.task, candidate)
        self.assertEqual(result['status'], 'located')
        self.assertTrue(result['alternate_version'])
        self.assertFalse(result['same_version_confirmed'])
        self.assertTrue(result['original_locator_identity_unverified'])
        self.assertEqual(result['metadata_reference'], reference)

    def test_provisional_same_locator_still_requires_all_metadata_authors(self):
        """同原URL和题名不能证明hint关系；缺少任一题录作者时保留未知。"""
        reference = {'paper_id': 'metadata-hint', 'canonical_locator': ARXIV}
        candidate = self.candidate(source='user_url', paper_url=OPENREVIEW, version=None,
            authors=['Synthetic Author', 'Second Author'], metadata={
                'original_locator_identity_unverified': True, 'metadata_reference': reference})
        self.fetch_content = 'Synthetic Author\n[PDF](https://openreview.net/pdf?id=SyntheticPaper)'
        primary = {'error_code': 'PDF_HTTP_ERROR', 'stage': 'primary_pdf',
                   'acquisition': {'http_status': 404, 'requested_source': OPENREVIEW}}
        result = self.service.locate_pdf(self.task, candidate, primary_failure=primary)
        self.assertEqual(result['status'], 'unconfirmed_identity_or_pdf')
        self.assertIsNone(result['pdf_url'])
        self.assertTrue(result['original_locator_identity_unverified'])
        self.assertEqual(result['metadata_reference'], reference)
        rejection = result['diagnostics'][0]
        self.assertEqual(rejection['status'], 'provisional_metadata_unconfirmed')
        self.assertEqual(rejection['request_state'], 'completed')
        self.assertNotIn('http_status', rejection)
        self.assertEqual(len(self.calls), 2)

    def test_provisional_same_locator_full_metadata_does_not_confirm_hint_version(self):
        """作者完整匹配只允许继续主文获取，不能静默确认metadata_ref与原URL同版。"""
        reference = {'paper_id': 'metadata-hint', 'canonical_locator': ARXIV}
        candidate = self.candidate(source='user_url', paper_url=OPENREVIEW, version=None,
            authors=['Synthetic Author', 'Second Author'], metadata={
                'original_locator_identity_unverified': True, 'metadata_reference': reference})
        self.fetch_content = 'Synthetic Author; Second Author\n[PDF](https://openreview.net/pdf?id=SyntheticPaper)'
        primary = {'error_code': 'PDF_HTTP_ERROR', 'stage': 'primary_pdf',
                   'acquisition': {'http_status': 404, 'requested_source': OPENREVIEW}}
        result = self.service.locate_pdf(self.task, candidate, primary_failure=primary)
        self.assertEqual(result['status'], 'located')
        self.assertTrue(result['diagnostic']['same_locator'])
        self.assertFalse(result['same_version_confirmed'])
        self.assertTrue(result['original_locator_identity_unverified'])
        self.assertEqual(result['metadata_reference'], reference)
        self.assertEqual(result['identity_basis'], 'exact_official_locator_title_and_provisional_metadata')
        self.assertTrue(candidate.metadata['original_locator_identity_unverified'])
        self.assertEqual(len(self.calls), 2)
