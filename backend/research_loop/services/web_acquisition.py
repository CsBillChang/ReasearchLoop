"""可选备用获取：独立网页收据、题录线索和严格同论文PDF定位。"""
from __future__ import annotations

import html
import hashlib
import json
import re
from typing import Any, Callable

from research_loop.domain.source_access import assert_external_access, canonical_paper_url, same_paper_locator, source_allowed, alternate_metadata_matches, exact_material_required, explicit_source_hosts, matched_detail_title, same_pdf_for_detail, normalize_web_sources, WebSourceValidationError, source_error_diagnostic
from research_loop.providers.paper_acquisition import PaperAcquisitionError, access_challenge_reason, blocked_primary_hosts, pdf_from_detail
from research_loop.providers.web_search import WebSearchProvider, public_http_url
from research_loop.providers.web_fetch import WebFetchProvider
from research_loop.storage.web_requests import WebRequestStore


def build_web_acquisition(db, settings):
    """关闭时不建表、不建连接、不发送请求；启用时复用已有DeepSeek认证配置。"""
    search = bool(getattr(settings, 'web_search_enabled', False))
    fetch = bool(getattr(settings, 'web_fetch_enabled', False))
    if not search and not fetch:
        return None
    arguments = {'api_key': settings.deepseek_api_key, 'base_url': settings.deepseek_base_url,
                 'timeout': getattr(settings, 'web_tools_timeout_seconds', 30.0),
                 'max_response_bytes': getattr(settings, 'web_tools_max_response_bytes', 2 * 1024 * 1024)}
    return WebAcquisitionService(db, WebSearchProvider(**arguments) if search else None,
                                 WebFetchProvider(**arguments) if fetch else None)


class WebAcquisitionService:
    """仅提供备用路径，原来源RPC结果与原PDF失败均由调用方独立保留。"""

    def __init__(self, db, search: WebSearchProvider | None = None, fetch: WebFetchProvider | None = None):
        """接收可替换HTTP适配和任务数据库，用独立表记录付费操作。"""
        self.db, self.search_provider, self.fetch_provider = db, search, fetch
        self.receipts = WebRequestStore(db)

    def close(self):
        """关闭本组件所持HTTP适配器。"""
        for provider in (self.search_provider, self.fetch_provider):
            if provider:
                provider.close()

    def _active(self, task_id: str, cancelled: Callable[[], bool]) -> bool:
        """发送窗口再次检查取消和当前任务；租约校验由执行器回调补充。"""
        task = self.db.get_task(task_id)
        root = self.db.get_task((task or {}).get('root_task_id') or task_id)
        terminal = {'cancelled', 'failed', 'completed', 'partial'}
        return bool(task and root and not task.get('cancel_requested') and not root.get('cancel_requested')
                    and task.get('status') not in terminal and root.get('status') not in terminal and not cancelled())

    def _run_request(self, root: str, task_id: str, operation: str, payload: dict,
                     provider, cancelled: Callable[[], bool]) -> dict:
        """补充本次操作的真实收据归因，不重发或改写独立HTTP账本。

        Args:
            root: 已解析的当前根任务。
            task_id: 当前执行角色。
            operation: search或fetch。
            payload: 经过发送前校验的无密钥请求。
            provider: 对应单次HTTP适配器。
            cancelled: 当前取消及租约检查。

        Returns:
            原收据结果及操作诊断；成功重放仍引用同一receipt_id。

        Raises:
            PaperAcquisitionError: 保持原错误码、HTTP归属和持久收据状态。
        """
        try:
            result = self.receipts.run(root, operation, payload, lambda: provider.request(payload),
                                       lambda: self._active(task_id, cancelled))
        except (PaperAcquisitionError, PermissionError) as exc:
            request_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            receipt_id = hashlib.sha256((root + '\n' + operation + '\n' + request_json).encode()).hexdigest()
            # 复用账本同一精确身份；MySQL文本排序规则可能把大小写不同的query视为相等。
            row = self.db.fetchone('SELECT receipt_id,status,diagnostic_json FROM web_request_receipts WHERE receipt_id=?',
                                   (receipt_id,))
            saved = json.loads(row['diagnostic_json']) if row else {}
            # 恢复时保留最初HTTP失败归属；UNRESOLVED只描述禁止重发，不能抹掉原HTTP500。
            saved_acquisition = saved.get('acquisition') or {}
            cancelled_before_send = isinstance(exc, PermissionError) or (row and row['status'] == 'not_sent')
            code = 'WEB_REQUEST_NOT_SENT' if cancelled_before_send else exc.code
            acquisition = getattr(exc, 'acquisition', {})
            state = row['status'] if row else 'not_requested'
            if cancelled_before_send or saved.get('error_code', code) == 'WEB_NOT_CONFIGURED':
                state = 'not_sent'
            details = {**saved_acquisition, **acquisition, 'stage': operation, 'operation': operation,
                       'request_state': state, 'receipt_id': row['receipt_id'] if row else acquisition.get('receipt_id'),
                       'receipt_status': row['status'] if row else None,
                       'original_error_code': code if cancelled_before_send else saved.get('error_code', code)}
            raise PaperAcquisitionError(code, str(exc), retryable=False if cancelled_before_send else exc.retryable,
                                        acquisition=details) from exc
        result['diagnostic'] = {'stage': operation, 'operation': operation, 'request_state': 'completed',
                                'receipt_id': result['receipt_id'], 'reused': result['reused']}
        return result

    def search(self, task_id: str, query: str, count: int = 5, *, date_from=None, date_to=None,
               sources=None, cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
        """返回保留原查询/日期的题录线索；网页时间不充当论文发表日期。"""
        try:
            sources = normalize_web_sources(sources)
        except WebSourceValidationError as exc:
            raise PaperAcquisitionError('WEB_SOURCE_INVALID', str(exc), acquisition=source_error_diagnostic(exc)) from exc
        root = assert_external_access(self.db, task_id)
        if not self.search_provider:
            raise PaperAcquisitionError('WEB_SEARCH_DISABLED', 'Search备用能力未启用',
                acquisition={'stage': 'search', 'operation': 'search', 'request_state': 'not_requested', 'receipt_id': None})
        try:
            payload = self.search_provider.payload(query, count)
        except ValueError as exc:
            raise PaperAcquisitionError('WEB_PARAMETER_INVALID', str(exc), acquisition={
                'stage': 'parameter_validation', 'operation': 'search', 'request_state': 'not_requested', 'receipt_id': None}) from exc
        receipt = self._run_request(root, task_id, 'search', payload, self.search_provider, cancelled)
        results, rejected = [], []
        explicit_hosts = explicit_source_hosts(self.db, task_id)
        for item in receipt['response']['results']:
            try:
                url = public_http_url(item['url'])
            except ValueError:
                rejected.append({'reason': 'invalid_result_url'})
                continue
            from urllib.parse import urlsplit
            if explicit_hosts and urlsplit(url).hostname not in explicit_hosts:
                rejected.append({'url': url, 'reason': 'outside_explicit_host_scope', 'stage': 'filtered_empty'})
                continue
            if not source_allowed(url, sources):
                rejected.append({'url': url, 'reason': 'outside_tool_source_filter', 'stage': 'filtered_empty'})
                continue
            anchor = canonical_paper_url(url)
            results.append({'title': item['title'], 'url': url, 'snippet': item.get('snippet', ''),
                            'summary': item.get('summary', ''), 'web_date': item.get('date'),
                            'paper_publication_date': None, 'paper_identity': list(anchor) if anchor else None,
                            'status': 'canonical_candidate' if anchor else 'unconfirmed_candidate',
                            'is_original_evidence': False})
        result_status = 'results' if results else 'filtered_empty' if rejected else 'empty'
        diagnostic = {**receipt['diagnostic'], 'stage': 'filtered_empty' if result_status == 'filtered_empty' else 'search',
                      'result_status': result_status, 'received_count': len(receipt['response']['results']),
                      'returned_count': len(results), 'rejected_count': len(rejected),
                      'tool_source_filter': sources, 'explicit_host_scope': sorted(explicit_hosts) if explicit_hosts else None}
        return {'query': query, 'date_from': date_from, 'date_to': date_to, 'sources': sources,
                'status': result_status, 'diagnostic': diagnostic,
                'transport_resource_guard': {'max_response_bytes': self.search_provider.max_response_bytes},
                'date_filter_status': 'publication_date_unverified' if date_from or date_to else 'not_requested',
                'results': results, 'rejected': rejected, 'receipt': {k: v for k, v in receipt.items() if k != 'response'},
                'limitation': '网页摘要及时间不是论文原文或发表日期；候选身份仍须原文获取确认。'}

    def locate_pdf(self, task_id: str, candidate, *, cancelled: Callable[[], bool] = lambda: False,
                   sources=None, primary_failure=None) -> dict[str, Any]:
        """搜索已授权论文的同版本详情，Fetch仅定位主文链接；不读HTML作为正文。"""
        assert_external_access(self.db, task_id)
        if not self.fetch_provider:
            raise PaperAcquisitionError('WEB_FETCH_DISABLED', 'Fetch备用能力未启用',
                acquisition={'stage': 'fetch', 'operation': 'fetch', 'request_state': 'not_requested', 'receipt_id': None})
        original = candidate.paper_url or candidate.pdf_url or ''
        anchor = canonical_paper_url(original)
        # 无稳定身份不能以标题近似度替换已指定论文，也不能静默去掉显式版本。
        if not anchor and not (candidate.doi or candidate.authors):
            return {'status': 'unconfirmed_identity', 'target_ref': candidate.paper_id, 'pdf_url': None,
                    'diagnostics': ['原题录缺少可核对身份及DOI/作者；保留候选，不授予替代原文权限。'],
                    'diagnostic': {'stage': 'identity_validation', 'operation': None, 'request_state': 'not_requested', 'receipt_id': None}}
        query = candidate.title
        if anchor and anchor[0] == 'arxiv':
            query = anchor[1] + (anchor[2] or '') + ' paper pdf'
        if len(query) > 128:
            return {'status': 'query_too_long', 'pdf_url': None, 'diagnostics': ['原题名超过Search上限；未截断或付费请求。'],
                    'diagnostic': {'stage': 'parameter_validation', 'operation': 'search', 'request_state': 'not_requested', 'receipt_id': None}}
        try:
            search = self.search(task_id, query, 5, sources=None, cancelled=cancelled)
        except PaperAcquisitionError as exc:
            raise PaperAcquisitionError(exc.code, str(exc), retryable=exc.retryable, acquisition={
                **exc.acquisition, 'primary_failure': primary_failure}) from exc
        search['requested_paper_sources'] = sources
        search['source_scope_note'] = '已知论文的来源限制核对原题录；下载域名不能替代发表场所。'
        diagnostics = []
        exact_required = exact_material_required(self.db, task_id, candidate)
        root = assert_external_access(self.db, task_id)
        from urllib.parse import urlsplit
        # 只根据实际PDF/详情请求归属阻止代理重取；Jina的401属于r.jina.ai，不能迁给论文站点。
        blocked_hosts = blocked_primary_hosts(primary_failure)
        for item in search['results']:
            url = item['url']
            if urlsplit(url).hostname in blocked_hosts:
                diagnostics.append({'url': url, 'status': 'original_access_denied_no_proxy_bypass', 'stage': 'primary_pdf',
                                    'operation': 'fetch', 'request_state': 'not_requested', 'receipt_id': None})
                continue
            exact_locator = same_paper_locator(original, url, candidate.version)
            alternate_anchor = canonical_paper_url(url)
            if not exact_locator and (exact_required or not alternate_anchor):
                diagnostics.append({'url': url, 'status': 'identity_or_version_unconfirmed', 'stage': 'identity_validation',
                                    'operation': 'fetch', 'request_state': 'not_requested', 'receipt_id': None,
                                    'search_receipt_id': search['receipt']['receipt_id']})
                continue
            payload = self.fetch_provider.payload(url)
            try:
                receipt = self._run_request(root, task_id, 'fetch', payload, self.fetch_provider, cancelled)
            except PaperAcquisitionError as exc:
                raise PaperAcquisitionError(exc.code, str(exc), retryable=exc.retryable, acquisition={
                    **exc.acquisition, 'primary_failure': primary_failure, 'search': search,
                    'diagnostics': diagnostics}) from exc
            body = receipt['response']
            challenge = access_challenge_reason(url, body['content'], title=body['title'])
            if challenge:
                diagnostics.append({'url': url, 'status': 'access_challenge', 'challenge_reason': challenge,
                                    'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            # Fetch摘要本身不能证明元数据；规范地址仍需与当前题名精确一致。
            verified_title = matched_detail_title(candidate, url, body['title'], body['content'])
            if not verified_title:
                diagnostics.append({'url': url, 'status': 'title_unconfirmed', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            provisional_reference = bool(candidate.metadata.get('original_locator_identity_unverified'))
            # 同一URL只确认地址；待核metadata_ref仍须完整作者/DOI，不能借exact分支升级关系。
            if (not exact_locator or provisional_reference) and not alternate_metadata_matches(candidate, verified_title, body['content']):
                diagnostics.append({'url': url, 'status': 'provisional_metadata_unconfirmed' if provisional_reference else 'alternate_metadata_unconfirmed', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            links = re.findall(r'\[([^\]\n]{1,80})\]\(([^\s)]+)\)', body['content'])
            markup = ''.join('<a href="' + html.escape(link, quote=True) + '">' + html.escape(label) + '</a>' for label, link in links)
            pdf = pdf_from_detail(body['content'] + markup, url)
            if not pdf:
                diagnostics.append({'url': url, 'status': 'no_unique_pdf', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            try:
                public_http_url(pdf)
            except ValueError:
                diagnostics.append({'url': url, 'status': 'invalid_pdf_url', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            if not same_pdf_for_detail(url, pdf):
                diagnostics.append({'url': pdf, 'status': 'detail_pdf_identity_mismatch', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            restricted_hosts = explicit_source_hosts(self.db, task_id)
            if restricted_hosts and urlsplit(pdf).hostname not in restricted_hosts:
                diagnostics.append({'url': pdf, 'status': 'pdf_outside_explicit_host_scope', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            # arXiv/OpenReview仍核对主文链接，避免同页面列出的另一篇或另一版PDF。
            pdf_identity_source = original if exact_locator else url
            if alternate_anchor and alternate_anchor[0] in {'arxiv', 'openreview'} and not same_paper_locator(pdf_identity_source, pdf, candidate.version if exact_locator else alternate_anchor[2]):
                diagnostics.append({'url': pdf, 'status': 'pdf_version_unconfirmed', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            if exact_locator and candidate.version and (not anchor or anchor[0] != 'arxiv'):
                diagnostics.append({'url': pdf, 'status': 'explicit_version_unconfirmed', 'stage': 'identity_validation', 'operation': 'fetch', 'request_state': 'completed', 'receipt_id': receipt['receipt_id']})
                continue
            identity_basis = ('exact_official_locator_title_and_provisional_metadata' if exact_locator and provisional_reference
                              else 'exact_official_locator_and_title' if exact_locator
                              else 'official_detail_exact_title_and_original_doi_or_authors')
            return {'status': 'located', 'pdf_url': pdf, 'detail_url': url, 'identity_basis': identity_basis,
                    'same_version_confirmed': exact_locator and not provisional_reference, 'alternate_version': not exact_locator,
                    'alternate_identity': list(alternate_anchor) if alternate_anchor else None,
                    'search': search, 'fetch_receipt': {k: v for k, v in receipt.items() if k != 'response'},
                    'diagnostics': diagnostics, 'fetch_content_is_original_evidence': False,
                    'diagnostic': {**receipt['diagnostic'], 'stage': 'identity_validation', 'result_status': 'located',
                        'search_receipt_id': search['receipt']['receipt_id'], 'same_locator': exact_locator},
                    'original_locator_identity_unverified': bool(candidate.metadata.get('original_locator_identity_unverified')),
                    'metadata_reference': candidate.metadata.get('metadata_reference')}
        fetch_receipts = list(dict.fromkeys(item['receipt_id'] for item in diagnostics if item.get('receipt_id')))
        # 本地身份检查不是一次Search或Fetch；各外部操作的完成事实单独引用，不改写归属。
        diagnostic = ({'stage': 'identity_validation', 'operation': None, 'request_state': 'not_requested',
                       'receipt_id': None} if diagnostics else dict(search['diagnostic']))
        diagnostic.update({'result_status': 'unconfirmed_identity_or_pdf',
                           'search_receipt_id': search['receipt']['receipt_id'], 'search_request_state': 'completed',
                           'fetch_receipt_ids': fetch_receipts,
                           'fetch_request_state': 'completed' if fetch_receipts else 'not_requested'})
        return {'status': 'unconfirmed_identity_or_pdf', 'pdf_url': None, 'search': search, 'diagnostics': diagnostics,
                'diagnostic': diagnostic,
                'original_locator_identity_unverified': bool(candidate.metadata.get('original_locator_identity_unverified')),
                'metadata_reference': candidate.metadata.get('metadata_reference')}
