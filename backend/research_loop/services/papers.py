"""论文服务：串接检索、授权、PDF 解析与页证据持久化；只向任务返回它有权读取的内容。"""
from __future__ import annotations

from contextlib import nullcontext
import json
import hashlib
import re
import unicodedata
from typing import Any

from research_loop.domain.evidence import CitationStats, Document, PaperCandidate, SearchResult, source_reference, validate_source_window, explicit_versioned_paper_locator
from research_loop.domain.reading import UserPageRange, user_page_range, material_page_scopes, MaterialScopeDenied
from research_loop.domain.source_access import assert_external_access, canonical_paper_url, explicit_source_hosts, final_pdf_locator, current_instructions
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.paper_acquisition import PaperAcquisitionError, blocked_primary_hosts, resolve_pdf_url
from research_loop.providers.pdf import PDFProvider, PaperReadCancelled
from research_loop.storage.mysql import Database


class PaperService:
    def __init__(self, db: Database, discovery: DiscoveryProvider, pdf: PDFProvider, citations: Any = None, *, web: Any = None):
        self.db = db
        self.discovery = discovery
        self.pdf = pdf
        self.citations = citations
        self.web = web
        self.candidates: dict[str, PaperCandidate] = {}

    def search(self, query: str, date_from: str | None = None, date_to: str | None = None,
               sources: list[str] | None = None, limit: int = 5, task_id: str | None = None) -> SearchResult:
        """输入查询和筛选，输出候选及逐源状态；把返回候选存入数据库并授予当前任务读取权限。"""
        if task_id:
            assert_external_access(self.db, task_id)
        result = self.discovery.search(query, date_from, date_to, sources, limit)
        for candidate in result.candidates:
            self.candidates[candidate.paper_id] = candidate
            self.db.save_paper(candidate.paper_id, candidate.model_dump(mode="json"))
            if task_id:
                self.db.grant_paper(task_id, candidate.paper_id, task_id)
        return result

    def search_web(self, task_id: str, query: str, count: int = 5, *, date_from=None, date_to=None,
                   sources=None, assigned_sources=None, cancelled=lambda: False) -> dict[str, Any]:
        """选择性备用发现；只为规范官方论文地址建立候选权限，不把摘要当原文。

        Args:
            task_id: 当前持久任务。
            query: 原始Search查询；不静默截断。
            count: 本次结果数。
            date_from: 原论文发表日期下限，网页日期不能替代。
            date_to: 原论文发表日期上限。
            sources: 明确的网页结果筛选，与论文发表来源范围独立。
            assigned_sources: 角色获派的论文发表来源；未确认时不得授新材料权限。
            cancelled: 取消及租约检测回调。

        Returns:
            新候选、未确认线索及独立收据，原来源观察保持原样。
        """
        assert_external_access(self.db, task_id)
        if not self.web:
            raise PaperAcquisitionError('WEB_SEARCH_DISABLED', 'Search备用能力未启用')
        result = self.web.search(task_id, query, count, date_from=date_from, date_to=date_to,
                                 sources=sources, cancelled=cancelled)
        if cancelled():
            raise PermissionError('web search completed but execution lease/cancellation prevents granting candidates')
        candidates = []
        permitted_metadata = self._metadata_references(task_id) if assigned_sources else []
        for item in result['results']:
            anchor = canonical_paper_url(item['url'])
            # 规范论文候选可供取文核对日期，但网页日期不能通过发表日期筛选。
            if not anchor:
                continue
            if assigned_sources:
                known = next((paper for paper in permitted_metadata
                    if paper.source.lower() in {source.lower() for source in assigned_sources}
                    and anchor in [canonical_paper_url(url) for url in (paper.paper_url,paper.pdf_url) if url]), None)
                if not known:
                    item['publication_source_status'] = 'unverified_for_assigned_sources'
                    continue
                item['paper_id'] = known.paper_id
                candidates.append(known.model_dump(mode='json'))
                continue
            import hashlib
            identity = ':'.join(part or '' for part in anchor)
            paper_id = 'web:' + hashlib.sha256(identity.encode()).hexdigest()[:32]
            candidate = PaperCandidate(paper_id=paper_id, title=item['title'], source='web',
                paper_url=item['url'], version=anchor[2], source_id=anchor[1],
                metadata={'discovery_provider': 'deepseek_search', 'query': query,
                    'identity_basis': 'canonical_official_paper_locator', 'metadata_unverified': True,
                    'web_date': item['web_date'], 'search_receipt': result['receipt']['receipt_id'],
                    'is_original_evidence': False, 'publication_date_unverified': True,
                    'requested_date_from': date_from, 'requested_date_to': date_to, 'requested_sources': sources})
            self.candidates[paper_id] = candidate
            self.db.save_paper(paper_id, candidate.model_dump(mode='json'))
            self.db.grant_paper(task_id, paper_id, task_id)
            candidates.append(candidate.model_dump(mode='json'))
            item['paper_id'] = paper_id
        result['assigned_publication_sources'] = assigned_sources
        result['candidates'] = candidates
        result['primary_source_observations'] = [{**action['observation'],
            'record_kind': 'historical_readonly_reference', 'action_id': action['action_id'],
            'task_id': action['task_id'], 'tool_name': 'search_papers', 'executed_now': False,
            'read_only': True} for action in self.db.list_actions(task_id) if action['tool_name'] == 'search_papers']
        return result

    def _known_access_failure(self, task_id: str, target_ref: str, source: str) -> dict[str, Any] | None:
        """复用当前任务确切地址的访问阻断，不借恢复重发原址。

        旧观察可能把200验证页记为PDF_LINK_UNAVAILABLE；按实际下载诊断
        重新识别阻断，但不修改历史观察，也不将Jina认证错误归给原论文站。
        """
        for action in reversed(self.db.list_actions(task_id)):
            if action['tool_name'] not in {'read_paper', 'acquire_paper_pdf'} or action['arguments'].get('target_ref') != target_ref:
                continue
            observation = action.get('observation') or {}
            data = observation.get('data') or {}
            acquisition = data.get('acquisition') or {}
            primary = acquisition.get('primary_failure')
            if not primary and data.get('error_code'):
                primary = {'error_code': data['error_code'], 'message': observation.get('message', ''),
                           'acquisition': acquisition}
            detail = (primary or {}).get('acquisition') or {}
            if detail.get('requested_source') == source and blocked_primary_hosts(primary):
                return {**primary, 'reused_failure_action': action['action_id']}
        return None

    def _acquisition_active(self, task_id: str, cancelled) -> bool:
        """写授权及发送前复查任务和根取消；租约由执行器回调验证。"""
        task = self.db.get_task(task_id)
        root = self.db.get_task((task or {}).get('root_task_id') or task_id)
        terminal = {'completed', 'partial', 'failed', 'cancelled'}
        return bool(task and root and not task.get('cancel_requested') and not root.get('cancel_requested')
                    and task.get('status') not in terminal and root.get('status') not in terminal and not cancelled())

    def _metadata_references(self, task_id: str, allowed_refs=None) -> list[PaperCandidate]:
        """只枚举当前任务与当前角色获派题录，不读取其他任务的参考身份。"""
        rows = self.db.fetchall('SELECT p.metadata_json FROM papers p JOIN task_papers t ON p.paper_id=t.paper_id WHERE t.task_id=?', (task_id,))
        return [candidate for row in rows
                if (candidate := PaperCandidate.model_validate(json.loads(row['metadata_json'])))
                and (not allowed_refs or candidate.paper_id in allowed_refs)]

    def _automatic_metadata_reference(self, task_id: str, allowed_refs: list[str] | None = None
                                      ) -> tuple[PaperCandidate | None, dict[str, Any]]:
        """只从当前用户明确身份和本角色授权题录中选择唯一参考。

        支持单个书名号/引号题名附完整作者列表，或单个明确 DOI。
        不能确定语法、同时出现多个目标、候选有多个版本时保留补参出口；
        不用长期记忆、模型计划或候选列表只有一条来证明论文身份。

        Args:
            task_id: 当前任务，用户依据由持久根指令及已提交澄清读取。
            allowed_refs: 后端传入的本角色材料范围。

        Returns:
            唯一参考题录和可读匹配诊断；不确定时题录为 None。
        """
        _, instructions = current_instructions(self.db, task_id)

        def normalized(value: str) -> str:
            """统一大小写和标点，不做模糊词义或作者姓氏匹配。"""
            return re.sub(r'\W+', '', unicodedata.normalize('NFKC', value)).casefold()

        def doi_value(value: str) -> str:
            """规范 DOI 前缀；只容忍用户句末标点。"""
            return re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value, flags=re.I).rstrip('。，,;；').casefold()

        locators, titles, dois, author_lists = set(), set(), set(), []
        positive_titles, positive_dois = set(), set()
        uncertain_identity_claim = False
        positive_locator = False
        title_pattern = re.compile(r'《([^《》\n]+)》|“([^“”\n]+)”|"([^"\n]+)"|‘([^‘’\n]+)’')
        action = r'(?:请)?(?:核验|读取|阅读|获取|查找|分析|检索|研究|解释|核对)'
        for instruction_index, text in enumerate(instructions):
            has_identity_claim = False
            # 正向句式之外若又明确撤回关联，不能继续沿用早先的身份。
            # 按分句检查，仅针对身份语义；“不保存文件”等执行约束不是身份否定。
            for clause in re.split(r'[。；;\n!?！？，,]', text):
                if (re.search(r'题录|题名|作者|身份|目标|匹配|绑定|关联|对应|URL|原址|上述|\b(?:identity|metadata|author|reference|target)\b', clause, re.I)
                        and re.search(r'不是|并非|不属于|不对应|不匹配|不要|不得|不能|不应|撤销|取消|背景|另一篇|\b(?:not|never|background)\b|\banother\s+(?:paper|reference|target)\b', clause, re.I)):
                    uncertain_identity_claim = True
            for url_match in re.finditer(r'https?://[^\s<>"\']+', text):
                anchor = canonical_paper_url(url_match[0].rstrip('。，；！？、,.);]>）'))
                if not anchor:
                    continue
                locators.add(anchor)
                prefix = re.split(r'[。；;\n!?！？，,]', text[:url_match.start()])[-1].strip()
                # 只接续明确的原址/同一目标动作，不能把“另一篇论文 URL”绑定到前文背景。
                locator_frame = (re.fullmatch(r'(?:原址|原文地址|论文地址|链接)\s*[:：]?', prefix)
                    or re.fullmatch(r'(?:请)?(?:随后|然后|先|优先)?(?:尝试|读取|阅读|获取|核验|下载)(?:原址|原文)?\s*[:：]?', prefix))
                positive_locator = positive_locator or bool(locator_frame)
                if not locator_frame:
                    uncertain_identity_claim = True
            for doi_match in re.finditer(r'(?<!\w)10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text, re.I):
                has_identity_claim = True
                doi = doi_value(doi_match[0])
                dois.add(doi)
                prefix = re.split(r'[。；;\n!?！？，,]', text[:doi_match.start()])[-1].strip()
                suffix = re.split(r'[。；;\n!?！？，,]', text[doi_match.end():])[0].strip()
                # 明确取文动作才是肯定身份；背景引用或否定DOI必须走显式补参。
                if (not suffix and re.fullmatch(action + r'\s*(?:DOI\s*[:：]?\s*|https?://(?:dx\.)?doi\.org/)?', prefix, re.I)):
                    positive_dois.add(doi)
                else:
                    uncertain_identity_claim = True
            for match in title_pattern.finditer(text):
                has_identity_claim = True
                title = normalized(next(value for value in match.groups() if value is not None))
                titles.add(title)
                prefix = re.split(r'[。；;\n!?！？]', text[:match.start()])[-1].strip()
                suffix = re.split(r'[。；;\n!?！？]', text[match.end():])[0].strip()
                before = re.fullmatch(action + r'\s*(.+?)\s*(?:的|所著的?)', prefix)
                english_prefix = re.fullmatch(r'(?:please\s+)?(?:read|verify|retrieve|analyze)\s*', prefix, re.I)
                after = re.fullmatch(r'by\s+(.+)', suffix, re.I) if english_prefix else None
                authors = before[1] if before and not suffix else after[1] if after else None
                if authors:
                    names = [normalized(value.strip()) for value in re.split(r'、|，|,|\band\b|和|与|及|&', authors, flags=re.I)]
                    if all(names):
                        author_lists.append(set(names))
                        positive_titles.add(title)
                    else:
                        uncertain_identity_claim = True
                else:
                    uncertain_identity_claim = True
            # 后续涉及身份的用户澄清必须重新明确肯定目标，不能沿用已撤回的旧陈述。
            if instruction_index and not has_identity_claim and re.search(
                    r'题录|题名|作者|身份|匹配|绑定|关联|对应|背景|另一篇|上述|\b(?:identity|metadata|author|reference|another paper)\b', text, re.I):
                uncertain_identity_claim = True
            # 明确题名字段也参与冲突诊断，但本轮不猜测此类开放句式的肯定取文含义。
            for value in re.findall(r'(?:题名|标题|\btitle)\s*[:：]\s*([^。；;\n]+)', text, re.I):
                titles.add(normalized(value.strip()))
                uncertain_identity_claim = True
        diagnostic = {'mode': 'current_user_authorized_exact_match', 'request_state': 'not_requested'}
        if len(locators) > 1 or len(titles) > 1 or len(dois) > 1:
            return None, {**diagnostic, 'status': 'multiple_explicit_targets'}
        if uncertain_identity_claim or not positive_locator:
            return None, {**diagnostic, 'status': 'identity_intent_requires_explicit_reference'}
        if not positive_dois and (not positive_titles or not author_lists):
            return None, {**diagnostic, 'status': 'explicit_title_authors_or_doi_required'}
        matches = []
        for reference in self._metadata_references(task_id, allowed_refs):
            if not reference.title or reference.metadata.get('metadata_unverified'):
                continue
            if titles and normalized(reference.title) not in titles:
                continue
            if dois and (not reference.doi or doi_value(reference.doi) not in dois):
                continue
            expected_authors = {normalized(author) for author in reference.authors if author.strip()}
            if author_lists and (not expected_authors or any(authors != expected_authors for authors in author_lists)):
                continue
            matches.append(reference)
        if len(matches) != 1:
            return None, {**diagnostic, 'status': 'ambiguous_authorized_metadata' if matches else 'no_authorized_metadata_match',
                          'matching_paper_ids': [reference.paper_id for reference in matches]}
        return matches[0], {**diagnostic, 'status': 'unique_match', 'metadata_ref': matches[0].paper_id,
                            'basis': 'current_user_doi' if dois else 'current_user_exact_title_and_complete_authors'}

    @staticmethod
    def _attach_metadata_reference(reference: PaperCandidate, identity_handoff: dict[str, Any]) -> None:
        """保留显式版本检查；参考线索不改写原 URL 的未知身份。

        Args:
            reference: 当前角色已授权且由显式参数或用户身份选定的题录。
            identity_handoff: 本次获取的观察，原地附加参考信息。

        Raises:
            PaperAcquisitionError: 同一论文地址指定了不同显式版本。
        """
        original_anchor = canonical_paper_url(identity_handoff['requested_url'])
        reference_anchor = canonical_paper_url(reference.paper_url or reference.pdf_url or '')
        if (original_anchor and reference_anchor and original_anchor[:2] == reference_anchor[:2]
                and original_anchor[2] != reference_anchor[2]):
            raise PaperAcquisitionError('PDF_METADATA_VERSION_MISMATCH', 'metadata_ref与原URL是不同显式版本，未将其映射为同一身份',
                retryable=True, acquisition={'stage': 'identity_resolution', 'request_state': 'not_requested',
                    'metadata_ref': reference.paper_id, 'identity_handoff': identity_handoff})
        identity_handoff['metadata_reference'] = {'paper_id': reference.paper_id, 'title': reference.title,
            'authors': reference.authors, 'paper_url': reference.paper_url, 'version': reference.version}

    def _assert_acquisition_source(self, candidate: PaperCandidate, sources) -> None:
        """发表范围必须由题录证明；用户给定URL本身不证明所属会议。"""
        if not sources:
            return
        if candidate.source == 'user_url':
            raise PaperAcquisitionError('PDF_PUBLICATION_SOURCE_UNVERIFIED',
                '原URL的发表来源尚未核实，不能在限定发表来源的角色中授予读取权限',
                acquisition={'stage':'identity_resolution','request_state':'not_requested',
                    'publication_source_status':'unverified_for_assigned_sources','assigned_sources':sources})
        if candidate.source.lower() not in {source.lower() for source in sources}:
            raise PermissionError('original paper venue exceeds assigned source scope')

    def _resolve_acquisition_target(self, task_id: str, target_ref: str, attachments, allowed_refs, cancelled, sources=None):
        """将用户规范URL交接为独立身份；绝不凭同题或近似URL别名为另一论文。"""
        if not target_ref.startswith(('http://', 'https://')):
            if allowed_refs and target_ref not in allowed_refs:
                raise PermissionError('fallback target exceeds assigned references')
            if not self.db.task_has_paper(task_id, target_ref):
                raise PermissionError('fallback requires an already-authorized paper_id or current-user canonical paper URL')
            row = self.db.get_paper(target_ref)
            if not row:
                raise PaperAcquisitionError('PAPER_REFERENCE_INVALID', '备用获取所需题录不存在')
            candidate = PaperCandidate.model_validate(row['metadata'])
            self._assert_acquisition_source(candidate, sources)
            if candidate.metadata.get('original_locator_identity_unverified'):
                return candidate, {'requested_url':candidate.paper_url,'resolved_paper_id':candidate.paper_id,
                    'identity_basis':candidate.metadata.get('identity_basis','current_user_exact_locator'),
                    'original_locator_identity_unverified':True}
            return candidate, None
        anchor = canonical_paper_url(target_ref)
        if not anchor:
            raise PaperAcquisitionError('PDF_URL_IDENTITY_INVALID', 'URL不是受支持的规范论文地址，未发送请求',
                acquisition={'stage':'parameter_validation','request_state':'not_requested','target_ref':target_ref})
        for candidate in self._metadata_references(task_id, allowed_refs):
            locators = [canonical_paper_url(url) for url in (candidate.paper_url, candidate.pdf_url) if url]
            if anchor in locators:
                self._assert_acquisition_source(candidate, sources)
                return candidate, {'requested_url':target_ref,'resolved_paper_id':candidate.paper_id,
                    'identity_basis':candidate.metadata.get('identity_basis','authorized_exact_locator'),
                    'original_locator_identity_unverified':bool(candidate.metadata.get('original_locator_identity_unverified') or not candidate.title)}
        assigned_input = any(ref in (allowed_refs or attachments or {}) and self.db.task_has_input(task_id, ref)
            and canonical_paper_url(item.get('url') or '') == anchor
            for ref, item in (attachments or {}).items())
        if allowed_refs and not assigned_input:
            # 根指令包含B，不代表只获派A的子任务也能取得B。
            raise PermissionError('canonical URL does not match this role assigned references')
        _, instructions = current_instructions(self.db, task_id)
        # 中文句读常与URL紧邻，不能将“）。请读取”吞成路径而漏掉真实用户授权。
        # 这里只修文本分界；规范身份、精确版本与本角色材料范围仍由原检查决定。
        explicit = any(canonical_paper_url(url.rstrip('。，；！？、,.);]>）')) == anchor
            for text in instructions for url in re.findall(r'https?://[^\s<>"\'，。；！？、（）【】《》“”‘’]+', text))
        if not explicit and not assigned_input:
            raise PermissionError('canonical URL was not explicitly supplied by the current user or assigned material')
        paper_id = 'user-url:' + hashlib.sha256(json.dumps(anchor, separators=(',', ':')).encode()).hexdigest()[:32]
        candidate = PaperCandidate(paper_id=paper_id,title='',authors=[],source='user_url',paper_url=target_ref,
            source_id=anchor[1],version=anchor[2],metadata={'identity_basis':'current_user_exact_locator',
                'metadata_unverified':True,'original_locator_identity_unverified':True})
        self._assert_acquisition_source(candidate, sources)
        if not self._acquisition_active(task_id, cancelled):
            raise PermissionError('fallback cancelled or execution lease lost before granting URL identity')
        self.db.save_paper(paper_id, candidate.model_dump(mode='json'))
        self.db.grant_paper(task_id, paper_id, task_id)
        self.candidates[paper_id] = candidate
        return candidate, {'requested_url':target_ref,'resolved_paper_id':paper_id,
            'identity_basis':'current_user_exact_locator','original_locator_identity_unverified':True}

    def acquire_pdf(self, task_id: str, target_ref: str, start_page: int = 1, page_count=None,
                    attachments=None, *, cancelled=lambda: False, sources=None, metadata_ref: str | None = None,
                    allowed_refs: list[str] | None = None) -> dict[str, Any]:
        """已知论文的可选备用获取；先保留原路径失败，再核对同版本PDF。

        Args:
            task_id: 当前任务。
            target_ref: 已授权题录身份，或当前用户明确给定的规范论文URL。
            start_page: 当前读取的物理起始页。
            page_count: 可选读取长度，用户页限制仍由read校验。
            attachments: 当前绑定附件，不自动授予全网权限。
            cancelled: 当前执行取消及租约检测回调。
            sources: 分派的来源范围。
            metadata_ref: 可选显式授权题录线索；缺省时仅唯一明确用户身份可自动衔接。
            allowed_refs: 后端提供的本角色材料范围，不能由模型指定。

        Returns:
            原有页级证据结果或明确未确认诊断；Fetch内容绝不存为证据。
        """
        assert_external_access(self.db, task_id)
        if not self.web:
            raise PaperAcquisitionError('WEB_SEARCH_DISABLED', '备用获取未启用')
        if not self._acquisition_active(task_id, cancelled):
            raise PermissionError('fallback cancelled or execution lease lost')
        reference = None
        if metadata_ref:
            if not self.db.task_has_paper(task_id, metadata_ref) or (allowed_refs and metadata_ref not in allowed_refs):
                raise PermissionError('metadata_ref is not authorized within this role assigned references')
            reference_row = self.db.get_paper(metadata_ref)
            if not reference_row:
                raise PaperAcquisitionError('PAPER_REFERENCE_INVALID', 'metadata_ref题录不存在')
            reference = PaperCandidate.model_validate(reference_row['metadata'])
        requested_ref = target_ref
        candidate, identity_handoff = self._resolve_acquisition_target(task_id, target_ref, attachments, allowed_refs, cancelled, sources)
        target_ref = candidate.paper_id
        if reference and identity_handoff:
            self._attach_metadata_reference(reference, identity_handoff)
        if sources and candidate.source != 'user_url' and candidate.source.lower() not in {source.lower() for source in sources}:
            raise PermissionError('original paper venue exceeds assigned source scope')
        # 即使复用已知拒绝，也须先检查当前用户页限制，不能让恢复跳过授权。
        user_scope = self.user_scope(task_id, target_ref=target_ref)
        if user_scope:
            selected_start = user_scope.start if start_page == 1 and page_count is None else start_page
            selected_count = page_count if page_count is not None else user_scope.end - selected_start + 1
            if selected_start < user_scope.start or selected_count < 1 or selected_start + selected_count - 1 > user_scope.end:
                raise PermissionError('requested pages exceed the user-specified physical page range')
        primary_source = resolve_pdf_url(candidate.pdf_url, candidate.paper_url)
        primary = self._known_access_failure(task_id, requested_ref, primary_source)
        if not primary and identity_handoff:
            primary = self._known_access_failure(task_id, identity_handoff['requested_url'], primary_source)
        if not primary:
            if not self._acquisition_active(task_id, cancelled):
                raise PermissionError('fallback cancelled or execution lease lost before primary PDF')
            try:
                primary_result = self.read(task_id, target_ref, start_page, page_count, attachments,
                    _cancelled=lambda: not self._acquisition_active(task_id, cancelled))
                if identity_handoff:
                    primary_result['identity_handoff'] = identity_handoff
                    primary_result['requested_target_ref'] = requested_ref
                return primary_result
            except PaperAcquisitionError as failure:
                # 不可变解析快照损坏属于本地合同错误，换URL不能修复或绕过同内容缓存。
                if failure.code == 'PDF_CACHE_INVALID':
                    raise
                primary = {'error_code': failure.code, 'message': str(failure), 'acquisition': failure.acquisition, 'stage':'primary_pdf'}
        if not self._acquisition_active(task_id, cancelled):
            raise PermissionError('fallback cancelled or execution lease lost')
        auto_reference = None
        if not candidate.title and not reference and identity_handoff:
            # 原址可成功时不需要身份提示；只有失败后才尝试确定性的参考交接。
            reference, auto_reference = self._automatic_metadata_reference(task_id, allowed_refs)
            identity_handoff['automatic_metadata_reference'] = auto_reference
            if reference:
                self._attach_metadata_reference(reference, identity_handoff)
        if not candidate.title and not reference:
            suggestions = [{'paper_id':item.paper_id,'title':item.title,'paper_url':item.paper_url}
                           for item in self._metadata_references(task_id, allowed_refs) if item.title]
            return {'status':'metadata_reference_required','target_ref':requested_ref,'pages':[],
                'identity_handoff':identity_handoff,'primary_failure':primary,'acquisition':{'primary_failure':primary},
                'diagnostic':{'stage':'identity_resolution','request_state':'not_requested',
                    'operation':'search_fetch','error_code':'PDF_METADATA_REFERENCE_REQUIRED','required_parameter':'metadata_ref',
                    'automatic_metadata_reference':auto_reference,
                    'authorized_metadata_refs':suggestions,
                    'correction':{'target_ref':requested_ref,'metadata_ref':'从authorized_metadata_refs明确选择；没有合适题录时先检索，不猜关联'}},
                'is_original_evidence':False}
        if reference and identity_handoff and identity_handoff['original_locator_identity_unverified']:
            # 仅本次定位使用明确题录线索；不把它写回原URL的未知题名/作者。
            candidate = candidate.model_copy(update={'title':reference.title,'authors':reference.authors,'doi':reference.doi,
                'metadata':{**candidate.metadata,'metadata_reference':identity_handoff['metadata_reference'],
                            'original_locator_identity_unverified':True}})
        located = self.web.locate_pdf(task_id, candidate, sources=sources, cancelled=cancelled, primary_failure=primary)
        if identity_handoff:
            located['identity_handoff'] = identity_handoff
        if not located.get('pdf_url'):
            return {'status': 'unconfirmed', 'target_ref': requested_ref, 'primary_failure': primary,
                    'identity_handoff':identity_handoff,'fallback': located, 'pages': [], 'is_original_evidence': False}
        if not self._acquisition_active(task_id, cancelled):
            raise PermissionError('fallback cancelled or execution lease lost')
        selected_ref = target_ref
        if located.get('alternate_version'):
            # 同论文异载体/异版本保留独立题录身份和内容hash，不回写原题录。
            import hashlib
            selected_ref = 'webvariant:' + hashlib.sha256((target_ref + '\n' + located['detail_url']).encode()).hexdigest()[:32]
            identity = located.get('alternate_identity') or []
            variant = candidate.model_copy(update={'paper_id': selected_ref, 'paper_url': located['detail_url'],
                'pdf_url': located['pdf_url'], 'version': identity[2] if len(identity) > 2 else None,
                'source': identity[0] if identity else 'web', 'source_id': identity[1] if len(identity) > 1 else None,
                'published': None, 'date_precision': 'unknown', 'doi': None,
                'metadata': {**candidate.metadata, 'alternate_for': target_ref, 'original_source': candidate.source,
                             'original_published': candidate.published, 'original_doi': candidate.doi, 'same_version_confirmed': False,
                             'identity_basis': located['identity_basis']}})
            self.db.save_paper(selected_ref, variant.model_dump(mode='json'))
            self.db.grant_paper(task_id, selected_ref, task_id)
            self.candidates[selected_ref] = variant
        try:
            result = self.read(task_id, selected_ref, start_page, page_count, attachments,
                               _acquired_pdf=located['pdf_url'], _cancelled=lambda: not self._acquisition_active(task_id, cancelled))
        except PaperAcquisitionError as failure:
            if failure.code == 'PDF_FINAL_LOCATOR_MISMATCH':
                located['same_version_confirmed'] = False
                located['diagnostics'].append(failure.acquisition.get('final_locator_validation', {}))
            raise PaperAcquisitionError(failure.code, str(failure), retryable=failure.retryable,
                acquisition={'primary_failure': primary, 'fallback': located,
                             'pdf_failure': failure.acquisition}) from failure
        final_validation = result['acquisition'].get('final_locator_validation')
        if final_validation and final_validation['status'] != 'same_official_locator':
            located['same_version_confirmed'] = False
            located['diagnostics'].append(final_validation)
        result['requested_target_ref'] = requested_ref
        if identity_handoff:
            result['identity_handoff'] = identity_handoff
        result['alternate_version'] = located.get('alternate_version', False)
        result['acquisition'] = {'primary_failure': primary, 'fallback': located,
                                 'pdf': result['acquisition']}
        return result

    def citation_stats(self, task_id: str, paper_ids: list[str]) -> list[CitationStats]:
        if not self.citations:
            return [CitationStats(paper_id=paper_id, status="unavailable", message="OpenAlex not configured")
                    for paper_id in paper_ids]
        output = []
        for paper_id in paper_ids:
            if not self.db.task_has_paper(task_id, paper_id):
                output.append(CitationStats(paper_id=paper_id, status="unavailable", message="paper is not authorized"))
                continue
            row = self.db.get_paper(paper_id)
            if not row:
                output.append(CitationStats(paper_id=paper_id, status="unmatched", message="paper metadata missing"))
                continue
            stat = self.citations.lookup(PaperCandidate.model_validate(row["metadata"]))
            self.db.save_citation_stats(stat.model_dump(mode="json"))
            output.append(stat)
        return output

    def _scope_context(self, task_id: str, tx=None):
        """读回真实根指令、材料名称及已验证读取映射，不读取模型计划。"""
        from pathlib import Path
        one, rows = (tx.one, tx.all) if tx else (self.db.fetchone, self.db.fetchall)
        task = one('SELECT * FROM tasks WHERE task_id=?', (task_id,))
        if not task:
            return {}, [], {}, {}
        root = one('SELECT * FROM tasks WHERE task_id=?', (task.get('root_task_id') or task_id,)) or task
        brief = json.loads(root.get('brief_json') or '{}')
        instructions = brief.get('material_scope_instructions') or [root['prompt']]
        answers = rows("SELECT answer FROM clarifications WHERE task_id=? AND status='answered' ORDER BY answered_at,clarification_id", (root['task_id'],))
        instructions = instructions + [r['answer'] for r in answers if r.get('answer')]
        materials = {}
        for item in json.loads(root.get('attachments_json') or '[]'):
            ref = item['input_ref']
            names = [item.get('paper_id'), item.get('filename')]
            if item.get('filename'):
                names.append(Path(item['filename']).stem)
            # input-A是既有附件工具的显式身份格式，不按附件列表次序赋A/B。
            if ref.startswith('input-'):
                names.append(ref[6:])
            materials[ref] = [n for n in names if n]
        for row in rows('SELECT DISTINCT p.paper_id,p.metadata_json FROM papers p JOIN task_papers g ON g.paper_id=p.paper_id JOIN tasks t ON t.task_id=g.task_id WHERE t.root_task_id=? OR t.task_id=?', (root['task_id'], task_id)):
            metadata = json.loads(row['metadata_json'])
            materials.setdefault(row['paper_id'], []).extend(n for n in (metadata.get('title'), metadata.get('paper_url'), metadata.get('pdf_url')) if n)
        for ref, binding in brief.get('material_read_bindings', {}).items():
            materials.setdefault(ref, [binding['paper_id']])
        # 只接受用户原句明确括号绑定且名称唯一命中；重名保留歧义。
        for text in instructions:
            for match in re.finditer(r'附件\s*([^\s（(，,。；;]{1,80})\s*[（(]([^）)]+)[）)]', text):
                label, description = match.groups()
                owners = [ref for ref, names in materials.items() if any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(n) + r'(?![A-Za-z0-9_])', description, re.I) for n in [ref, *names])]
                if len(owners) == 1:
                    materials[owners[0]].append(label)
        return root, instructions, materials, brief

    def material_scopes(self, task_id: str, tx=None):
        """返回服务端已知各材料的当前范围，pending不会阻止其他材料。"""
        _, instructions, materials, _ = self._scope_context(task_id, tx)
        return material_page_scopes(instructions, materials)

    def user_scope(self, task_id: str, tx=None, *, target_ref: str | None = None) -> UserPageRange | None:
        """先验证具体目标授权，再从根真实指令解析该材料的当前页范围。

        未传目标只保留旧全局helper合同；新业务入口必须传已授权材料身份。
        """
        one = tx.one if tx else self.db.fetchone
        _, instructions, materials, _ = self._scope_context(task_id, tx)
        if target_ref is None:
            return user_page_range(instructions)
        if not (one('SELECT 1 AS ok FROM task_inputs WHERE task_id=? AND input_ref=?', (task_id, target_ref))
                or one('SELECT 1 AS ok FROM task_papers WHERE task_id=? AND paper_id=?', (task_id, target_ref))):
            raise PermissionError('material is not authorized for this task')
        materials.setdefault(target_ref, [])
        return material_page_scopes(instructions, materials)[target_ref].require_range()

    def _bind_material(self, task_id: str, target_ref: str, document: dict, pages: list[dict]) -> None:
        """在现有事务合并实际读取身份；不缓存权限，也不覆盖其他根JSON字段。"""
        from research_loop.storage.memory import MemoryStore, dump
        with MemoryStore(self.db).transaction() as tx:
            task = tx.one('SELECT * FROM tasks WHERE task_id=?', (task_id,))
            root_id = task.get('root_task_id') or task_id
            root = tx.one('SELECT * FROM tasks WHERE task_id=?' + (' FOR UPDATE' if self.db.kind == 'mysql' else ''), (root_id,))
            brief = json.loads(root.get('brief_json') or '{}')
            bindings = brief.setdefault('material_read_bindings', {})
            identity = {key: document[key] for key in ('document_id', 'content_sha256', 'paper_id')}
            old = bindings.get(target_ref)
            if old and any(old[key] != identity[key] for key in identity):
                raise ValueError('material document/version changed; rebind with a new input_ref')
            binding = bindings.setdefault(target_ref, {**identity, 'evidence_ids': []})
            binding['evidence_ids'] = sorted(set(binding['evidence_ids']) | {p['evidence_id'] for p in pages})
            # 用户澄清可能与解析并发；提交前在同一事务复查每一页最新权限。
            scope = self.user_scope(task_id, tx, target_ref=target_ref)
            if scope and any(p['page_number'] is None or not scope.start <= p['page_number'] <= scope.end for p in pages):
                raise PermissionError('source outside user physical page range')
            tx.execute('UPDATE tasks SET brief_json=? WHERE task_id=?', (dump(brief), root_id))

    def _source_material(self, task_id: str, document: dict, page: dict, tx=None, target_ref=None):
        """以实际页映射选择来源，不因同hash或document显示paper_id漂移合并材料。"""
        _, instructions, materials, brief = self._scope_context(task_id, tx)
        scopes = material_page_scopes(instructions, materials)
        bindings = brief.get('material_read_bindings', {})
        one = tx.one if tx else self.db.fetchone
        rows = tx.all if tx else self.db.fetchall
        role_refs = {r['input_ref'] for r in rows('SELECT input_ref FROM task_inputs WHERE task_id=?', (task_id,))}
        role_refs.update(r['paper_id'] for r in rows('SELECT paper_id FROM task_papers WHERE task_id=?', (task_id,)))
        candidates = [(ref, item) for ref, item in bindings.items()
                      if item['document_id'] == document['document_id']
                      and item['content_sha256'] == document['content_sha256']
                      and page['evidence_id'] in item['evidence_ids'] and (target_ref is None or ref == target_ref)
                      and (not role_refs or ref in role_refs)]
        if not candidates:
            if any(item['document_id'] == document['document_id'] for item in bindings.values()):
                raise PermissionError('source material outside current role or actual read mapping')
            # 历史记录只能按明确paper_id对应材料，不能用同hash推导旧别名。
            refs = [ref for ref, names in materials.items() if document['paper_id'] == ref or document['paper_id'] in names]
            if len(refs) > 1:
                raise PermissionError('ambiguous historical material identity')
            if refs and role_refs and refs[0] not in role_refs:
                raise PermissionError('historical material outside current role references')
            if target_ref is not None and (not refs or target_ref not in refs):
                raise PermissionError('source material binding unavailable')
            if not refs:
                scope = user_page_range(instructions)
                refs = [None]
            else:
                try:
                    scope = scopes[refs[0]].require_range()
                except ValueError as exc:
                    raise MaterialScopeDenied('source material pending clarification') from exc
            candidates = [(refs[0], {'paper_id': document['paper_id'], 'legacy_scope': scope})]
        if not role_refs and len(candidates) > 1:
            raise PermissionError('ambiguous evidence-only material binding')
        for ref, binding in candidates:
            try:
                scope = binding['legacy_scope'] if 'legacy_scope' in binding else scopes[ref].require_range()
            except ValueError:
                continue
            if scope and (page['page_number'] is None or not scope.start <= page['page_number'] <= scope.end):
                continue
            display_id = binding['paper_id']
            if display_id != document['paper_id'] and self.paper_alias_matches(display_id, document['paper_id'], tx):
                display_id = document['paper_id']
            return ref, {**document, 'paper_id': display_id}
        raise MaterialScopeDenied('source outside user physical page range or material pending clarification')

    def authorized_source(self, task_id: str, evidence_id: str, tx=None, *, with_readback: bool = True, target_ref: str | None = None) -> dict:
        """共同读回当前角色已授权的页、内容版本与可用入口，不新增权限。

        Args:
            task_id: 实际角色任务，不能用根任务替代页授权。
            evidence_id: 已取得的确切页。
            tx: 可选事务，提交时所有来源和用户范围在同一事务读回。
            with_readback: 仅版本检查时省略入口查询。

        Returns:
            reference、page、document及readback；页授权本身不生成题录/附件授权。

        Raises:
            PermissionError: 页未授权或已不在用户允许范围。
            ValueError: 文档、页定位或版本合同损坏。
        """
        one = tx.one if tx else self.db.fetchone
        page = one('SELECT e.* FROM page_evidence e JOIN task_evidence t ON t.evidence_id=e.evidence_id WHERE t.task_id=? AND e.evidence_id=?', (task_id, evidence_id))
        if not page:
            raise PermissionError('evidence outside current task: ' + evidence_id)
        page = dict(page)
        page['page_number'] = page['page_number'] or None
        document = one('SELECT * FROM documents WHERE document_id=?', (page['document_id'],))
        if not document:
            raise ValueError('source document unavailable: ' + evidence_id)
        material_ref, document = self._source_material(task_id, document, page, tx, target_ref)
        number, total = page['page_number'], document['total_pages']
        if (total and (type(number) is not int or not 1 <= number <= total)) or (not total and number is not None):
            raise ValueError('source physical page identity invalid')
        result = {'reference': source_reference(page, document), 'page': page, 'document': document}
        if with_readback:
            result['readback'] = self.source_readback(task_id, document, number, tx, material_ref=material_ref)
        return result

    def paper_alias_matches(self, original_id: str, current_id: str, tx=None) -> bool:
        """比较显示题录身份；同hash/同题名不能替代双方明确同版证据。"""
        if original_id == current_id:
            return True
        one = tx.one if tx else self.db.fetchone
        def locator(paper_id):
            """题录在当前事务只读解析，损坏元数据无法证明别名。"""
            row = one('SELECT metadata_json FROM papers WHERE paper_id=?', (paper_id,))
            try:
                return explicit_versioned_paper_locator(paper_id, json.loads(row['metadata_json'])) if row else None
            except (ValueError, TypeError):
                return None
        original = locator(original_id)
        return original is not None and original == locator(current_id)

    def source_readback(self, task_id: str, document: dict, page_number: int | None, tx=None, *, material_ref=None) -> dict:
        """按read相同优先级选择本角色可尝试入口，不以SQL页授权保证载体存在。"""
        one = tx.one if tx else self.db.fetchone
        all_rows = tx.all if tx else self.db.fetchall
        task = one('SELECT attachments_json FROM tasks WHERE task_id=?', (task_id,)) or {}
        attachments = json.loads(task.get('attachments_json') or '[]')
        paper_id, target = document['paper_id'], None
        unavailable_reason = 'no_current_role_read_target'
        def same_paper(other):
            """不同别名必须用348明确版本证明，不按共享内容推断。"""
            if other == paper_id:
                return True
            return self.paper_alias_matches(other, paper_id, tx)
        for attached in attachments:
            ref = attached.get('input_ref')
            if material_ref is not None and ref != material_ref:
                continue
            # read会读回同名持久题录并优先验证paper grant，交给下方题录分支处理。
            collision = one('SELECT paper_id FROM papers WHERE paper_id=?', (ref,)) if ref else None
            if (ref and not collision and same_paper(attached.get('paper_id') or ref)
                    and one('SELECT 1 AS ok FROM task_inputs WHERE task_id=? AND input_ref=?', (task_id, ref))):
                source = attached.get('local_path') or attached.get('url')
                problem = PDFProvider.local_source_problem(source) if source else 'PDF_URL_MISSING'
                if problem:
                    unavailable_reason = problem
                    continue
                target = ref
                break
        if target is None:
            for row in all_rows('SELECT p.paper_id,p.metadata_json FROM papers p JOIN task_papers t ON t.paper_id=p.paper_id WHERE t.task_id=? ORDER BY p.paper_id', (task_id,)):
                if material_ref is not None and row['paper_id'] != material_ref:
                    continue
                try:
                    # 与read一致：非授权附件入口优先用本服务已返回的候选，否则读持久题录。
                    input_attached = any(item.get('input_ref') == row['paper_id'] for item in attachments)
                    input_granted = input_attached and one('SELECT 1 AS ok FROM task_inputs WHERE task_id=? AND input_ref=?', (task_id, row['paper_id']))
                    candidate = None if input_granted else self.candidates.get(row['paper_id'])
                    if candidate is None:
                        candidate = PaperCandidate.model_validate(json.loads(row['metadata_json']))
                except (ValueError, TypeError):
                    continue
                if candidate.paper_id != row['paper_id']:
                    continue
                if not same_paper(row['paper_id']):
                    continue
                try:
                    source = resolve_pdf_url(candidate.pdf_url, candidate.paper_url)
                    problem = PDFProvider.local_source_problem(source) if source else 'PDF_URL_MISSING'
                except (ValueError, TypeError):
                    problem = 'PDF_URL_MISSING'
                if problem:
                    unavailable_reason = problem
                    continue
                target = row['paper_id']
                break
        if target is None:
            return {'available': False, 'target_ref': None, 'reason': unavailable_reason}
        if page_number is None:
            return {'available': True, 'target_ref': target, 'reason': 'authorized_document_text'}
        return {'available': True, 'target_ref': target, 'start_page': page_number, 'page_count': 1,
                'reason': 'authorized_read_target'}

    def validate_source_observation(self, task_id: str, item: dict, data: dict, *, kind: str = 'page') -> dict:
        """按共同权限与版本合同校验原始返回窗口，供覆盖、上下文和检索复用。"""
        source = self.authorized_source(task_id, item.get('evidence_id', ''), target_ref=item.get('target_ref') or data.get('target_ref'))
        validate_source_window(item, data, source['page'], source['document'], kind=kind)
        observed, current = item.get('paper_id') or data.get('paper_id'), source['document']['paper_id']
        proven = bool(observed) and self.paper_alias_matches(observed, current)
        status = ('not_recorded' if not observed else 'same' if observed == current else
                  'explicit_same_version_alias' if proven else 'unverified_difference')
        source['display_identity'] = {'status': status, 'observed_paper_id': observed,
                                      'current_paper_id': current, 'paper_id': current if proven else None}
        return source

    def read(self, task_id: str, target_ref: str, start_page: int = 1, page_count: int | None = None,
             attachments: dict[str, dict[str, Any]] | None = None, *,
             _acquired_pdf: str | None = None, _cancelled=lambda: False) -> dict[str, Any]:
        """读取授权论文，默认向调用方返回完整的逐页解析正文。

        Args:
            task_id: 接收证据的任务 ID。
            target_ref: 已授权的 paper_id 或附件 input_ref。
            start_page: 本次读取的物理起始页，默认第一页。
            page_count: 可选片段长度；省略时读到末页或用户指定边界。
            attachments: 后端已绑定的附件信息。
            _cancelled: 当前执行取消或租约失效回调，贯通在途获取和授权边界。

        Returns:
            页级正文、源 hash、实际覆盖范围及解析缺口；缓存完整和本次返回完整分别记录。

        Raises:
            PermissionError: 论文未授权或请求超出用户明确限定页范围。
            PaperAcquisitionError: 论文身份不存在、获取地址缺失或 PDF 获取/解析失败。
            ValueError: 页码非法或用户范围存在歧义。
        """
        def cancelled() -> bool:
            """将执行器取消/租约与任务族状态合并，避免取文后继续授权。"""
            return not self._acquisition_active(task_id, _cancelled)

        def check_cancelled() -> None:
            """在不可撤销的证据提交前拒绝已取消读取。"""
            if cancelled():
                raise PaperReadCancelled("论文读取已取消或执行权失效")

        check_cancelled()
        attached = attachments and attachments.get(target_ref)
        attached_authorized = bool(attached and self.db.task_has_input(task_id, target_ref))
        candidate = None if attached_authorized else self.candidates.get(target_ref)
        if candidate and not self.db.task_has_paper(task_id, target_ref):
            raise PermissionError("paper is not authorized for this task")
        if not candidate:
            stored = self.db.get_paper(target_ref)
            if stored:
                if not self.db.task_has_paper(task_id, target_ref):
                    raise PermissionError("paper is not authorized for this task")
                candidate = PaperCandidate.model_validate(stored["metadata"])
        source = None
        paper_id = target_ref
        if candidate:
            # 论文 的显式 PDF 优先及来源转换规则；合法详情页不再被误报为非法身份。
            source = resolve_pdf_url(candidate.pdf_url, candidate.paper_url)
        elif attached_authorized:
            source = attached.get("local_path") or attached.get("url")
            paper_id = attached.get("paper_id") or target_ref
        if not source:
            if candidate or attached_authorized:
                raise PaperAcquisitionError("PDF_URL_MISSING", "论文/附件身份有效，但未提供 PDF 或详情页地址")
            raise PaperAcquisitionError("PAPER_REFERENCE_INVALID", "target_ref is not a returned paper_id or an authorized input_ref")
        if _acquired_pdf:
            if not candidate or not self.db.task_has_paper(task_id, target_ref):
                raise PermissionError("alternate PDF requires an authorized paper candidate")
            source = _acquired_pdf
        user_scope = self.user_scope(task_id, target_ref=target_ref)
        if user_scope:
            if start_page == 1 and page_count is None:
                start_page = user_scope.start
            if page_count is None:
                page_count = user_scope.end - start_page + 1
            if start_page < user_scope.start or page_count < 1 or start_page + page_count - 1 > user_scope.end:
                raise PermissionError("requested pages exceed the user-specified physical page range")
        page_request = _acquired_pdf is not None or user_scope is not None or start_page != 1 or page_count is not None
        ingest = getattr(self.pdf, "ingest_pages", self.pdf.ingest) if page_request else self.pdf.ingest
        # 作用域不改已有 ingest 两位置参数，兼容本地附件适配器及合成替身。
        scope = getattr(self.pdf, "cancellation_scope", None)
        with scope(cancelled) if callable(scope) else nullcontext():
            check_cancelled()
            document = ingest(source, paper_id)
        check_cancelled()
        acquisition = getattr(self.pdf, 'last_fetch', {})
        if candidate and acquisition.get('final_url'):
            # 详情及初始PDF链接合法仍不能授权重定向后的另一论文/版本。
            final_identity = final_pdf_locator(source, acquisition['final_url'], explicit_source_hosts(self.db, task_id))
            acquisition['final_locator_validation'] = final_identity
            if final_identity['status'] in {'paper_or_version_mismatch', 'outside_explicit_host_scope'}:
                raise PaperAcquisitionError('PDF_FINAL_LOCATOR_MISMATCH', 'PDF最终下载地址与论文/版本或用户域名限制不符，未授予页证据',
                                            acquisition=dict(acquisition))
        check_cancelled()
        self.db.save_document(document.model_dump(mode="json"))
        count = page_count if page_count is not None else (document.total_pages - start_page + 1 if document.total_pages else None)
        pages = self.pdf.pages(document, start_page, count)
        self._bind_material(task_id, target_ref, document.model_dump(mode="json"), [p.model_dump(mode="json") for p in pages])
        for page in pages:
            check_cancelled()
            self.db.save_page(page.model_dump(mode="json"))
            check_cancelled()
            self.db.mark_task_evidence(task_id, page.evidence_id)
        check_cancelled()
        for page in pages:
            self.authorized_source(task_id, page.evidence_id, with_readback=False, target_ref=target_ref)
        next_page = pages[-1].page_number + 1 if pages and document.total_pages and pages[-1].page_number < document.total_pages else None
        return {
            "paper_id": paper_id,
            "target_ref": target_ref,
            "document_id": document.document_id,
            "content_sha256": document.content_sha256,
            "total_pages": document.total_pages,
            "returned_pages": [page.page_number for page in pages],
            "pages": [page.model_dump(mode="json") for page in pages],
            "next_page": next_page,
            "acquisition": dict(getattr(self.pdf, "last_fetch", {})),
            "parse_coverage": {"total_pages": document.total_pages, "cached": True},
            "read_coverage": {
                "returned_pages": [page.page_number for page in pages],
                "full_document_returned": ([page.page_number for page in pages] == list(range(1, document.total_pages + 1))) if document.total_pages else None,
                "all_extracted_text_returned": True if not document.total_pages else count == document.total_pages,
                "locator_kind": "physical_page" if document.total_pages else "document_text",
                "scope_origin": "user" if user_scope else "full_document",
                "user_page_range": {"start": user_scope.start, "end": user_scope.end,
                                    "instruction": user_scope.instruction} if user_scope else None,
            },
            "known_gaps": document.gaps + sorted({gap for page in pages for gap in page.gaps}),
        }

    def evidence_bundle(self, task_id: str, evidence_ids: list[str]) -> list[dict[str, Any]]:
        """组装授权原文及解析限制；仅选中的页进入包，不宣称整份 PDF 已理解。"""
        output = []
        documents = {}
        for ref in dict.fromkeys(evidence_ids):
            source = self.authorized_source(task_id, ref)
            page, document = source['page'], source['document']
            paper_id = document['paper_id']
            documents[page['document_id']] = document
            target = source['readback']['target_ref']
            output.append({"evidence_id": ref, "paper_id": paper_id, "target_ref": target, "readback": source["readback"],
                           "document_id": page["document_id"], "source_sha256": document["content_sha256"],
                           "page_number": page["page_number"], "text": page["text"],
                           "parsed_markdown": page.get("parsed_markdown", ""),
                           "assets": json.loads(page.get("assets_json") or "[]"),
                           "gaps": json.loads(document.get("gaps_json") or "[]") + json.loads(page.get("gaps_json") or "[]")})
        # 授权池是当前角色累计取得/继承的证据，选中包只是本次核验输入，二者不能互相替代。
        available = {}
        for ref in self.db.list_task_evidence(task_id):
            try:
                evidence = self.authorized_source(task_id, ref, with_readback=False)['page']
            except PermissionError:
                continue
            available.setdefault(evidence['document_id'], set()).add(evidence['page_number'])
        for page in output:
            doc = documents[page["document_id"]]
            selected = sorted({p["page_number"] for p in output if p["document_id"] == page["document_id"] and p["page_number"] is not None})
            total = doc["total_pages"] or None
            page["read_scope"] = {"total_physical_pages": total, "included_physical_pages": selected,
                "all_physical_pages_included": selected == list(range(1, total + 1)) if total else None,
                "locator_kind": "physical_page" if total else "document_text",
                "visual_content_interpreted": False,
                "scope_kind": "selected_evidence_bundle",
                "limitation": "这是本次选入核验包的页，不是任务累计阅读范围；含全部页也不代表图片、图表、公式完整可读。"}
            authorized = sorted(n for n in available.get(page['document_id'], set()) if n is not None)
            page['task_evidence_scope'] = {'authorized_physical_pages': authorized,
                'all_physical_pages_available': authorized == list(range(1,total+1)) if total else None,
                'provenance': 'persisted_task_evidence',
                'limitation': '任务已取得或从子任务继承的文字证据；不等于每个Agent均阅读过或理解所有图表。'}
        return output

    def evidence_scope(self, task_id: str) -> list[dict[str, Any]]:
        """从真实授权页汇总交接范围，不采用模型自报的全文/图表理解程度。"""
        docs = {}
        refs = []
        for ref in self.db.list_task_evidence(task_id):
            try:
                self.authorized_source(task_id, ref, with_readback=False)
                refs.append(ref)
            except PermissionError:
                continue
        for page in self.evidence_bundle(task_id, refs):
            item = docs.setdefault(page["document_id"], {"document_id": page["document_id"],
                "source_sha256": page["source_sha256"], **page["read_scope"], "gaps": [], "asset_pages": []})
            item["gaps"] = list(dict.fromkeys(item["gaps"] + page["gaps"]))
            if page["assets"] and page["page_number"] not in item["asset_pages"]:
                item["asset_pages"].append(page["page_number"])
        return list(docs.values())

    def inspect_media(self, task_id: str, target_ref: str, start_page: int, page_count: int | None, attachments: dict) -> dict:
        """复用授权读取取得物理 PDF，再调用 论文 图表适配；无视觉模型时如实返回限制。"""
        from research_loop.providers.paper_media import PaperMediaProvider
        # 显式起始范围触发 PDF 页级路径，不能把 Jina 文本 hash 当 PDF hash。
        scope = self.user_scope(task_id, target_ref=target_ref)
        start = scope.start if scope and start_page == 1 else start_page
        first = self.read(task_id, target_ref, start, 1, attachments)
        total = first['total_pages']
        end = start + page_count - 1 if page_count is not None else (scope.end if scope else total)
        if not total or end > total or end < start or (scope and end > scope.end):
            raise PermissionError('media pages exceed PDF or user scope')
        files = self.pdf.file_store
        result = PaperMediaProvider(files).extract(files.cache_pdf_path(first['content_sha256']), first['content_sha256'], list(range(start, end + 1)))
        return {**result, 'target_ref': target_ref, 'document_id': first['document_id'],
                'gaps': ['图表文件已提取，但当前文字模型没有接收图片；不能宣称视觉核验通过。']}

    def visual_input_for_evidence(self, task_id: str, args):
        """从已授权的真实页派生视觉身份，不取新论文或让模型填写hash。

        Args:
            task_id: 当前Supervisor任务。
            args: VisualDelegationInput，指定实际取得的evidence_id和问题。

        Returns:
            与原独立组件相同的VisualInput；原正文与来源不改。

        Raises:
            PermissionError: 页不属于本任务或超出用户范围。
            ValueError: 无物理页、PDF/解析缓存缺失或来源版本变化。
        """
        from hashlib import sha256
        from pathlib import Path
        from research_loop.domain.visual_reading import VisualInput
        task = self.db.get_task(task_id)
        if not task or task['cancel_requested'] or task['status'] != 'running':
            raise PermissionError('visual delegation requires an active task')
        page = self.evidence_bundle(task_id, [args.evidence_id])[0]
        number = page['page_number']
        if not isinstance(number, int) or number < 1:
            raise ValueError('visual reading requires physical PDF page evidence; text-only source has no page image')
        scope = self.user_scope(task_id, target_ref=page['target_ref'])
        if scope and not scope.start <= number <= scope.end:
            raise PermissionError('visual page exceeds user-specified physical page range')
        pdf = self.pdf.file_store.cache_pdf_path(page['source_sha256'])
        if not pdf.is_file() or sha256(pdf.read_bytes()).hexdigest() != page['source_sha256']:
            raise ValueError('visual PDF cache missing or changed; no automatic fetch')
        attached = next((a for a in json.loads(task['attachments_json'] or '[]')
                         if a['input_ref'] == page['target_ref']), None)
        if attached and attached.get('local_path'):
            source = Path(attached['local_path'])
            if not source.is_file() or sha256(source.read_bytes()).hexdigest() != page['source_sha256']:
                raise ValueError('visual attachment version changed')
        config_hash = sha256(json.dumps(self.pdf.parser_config, sort_keys=True).encode()).hexdigest()
        key = sha256(f'{page["source_sha256"]}:pymupdf:{self.pdf.parser_version}:{config_hash}'.encode()).hexdigest()
        cache = self.pdf.file_store.cache_parse_path(key)
        if page['document_id'] != 'doc:' + key[:32] or not cache.is_file():
            raise ValueError('visual parsed cache identity missing or changed')
        cached = next((p for p in json.loads(cache.read_text())['pages'] if p['page_number'] == number), None)
        if not cached or cached['document_id'] != page['document_id'] or cached['text'] != page['text']:
            raise ValueError('visual retained page differs from original parsed cache')
        return VisualInput(target_ref=page['target_ref'], document_id=page['document_id'],
            pdf_sha256=page['source_sha256'], page_text_sha256=sha256(page['text'].encode()).hexdigest(),
            physical_page=number, question=args.question, group='visual', bbox=args.bbox)

    def prepare_visual(self, task_id: str, request) -> dict:
        """只从当前任务绑定附件与既有PDF缓存准备视觉组件输入，不联网取文。

        Args:
            task_id: 当前组件任务；继承的附件须已持久绑定且授权。
            request: VisualInput，包含冻结来源、原文hash及单个物理页。

        Returns:
            原页文字、图片文件清单及不含图片字节的来源manifest。

        Raises:
            PermissionError: 取消、附件未绑定、页范围或来源身份不一致。
            ValueError: 缓存缺失或损坏；不自动重检索、重解析。
        """
        from hashlib import sha256
        from pathlib import Path
        from research_loop.domain.visual_reading import VisualInput
        from research_loop.providers.paper_media import PaperMediaProvider
        item = VisualInput.model_validate(request)
        task = self.db.get_task(task_id)
        if not task or task['cancel_requested'] or task['status'] in {'cancelled', 'failed', 'completed', 'partial'}:
            raise PermissionError('visual task is not active')
        attachments = {a['input_ref']: a for a in json.loads(task['attachments_json'] or '[]')}
        attached = attachments.get(item.target_ref)
        if not attached or not self.db.task_has_input(task_id, item.target_ref):
            raise PermissionError('visual input is not an assigned attachment')
        scope = self.user_scope(task_id, target_ref=item.target_ref)
        if scope and not scope.start <= item.physical_page <= scope.end:
            raise PermissionError('visual page exceeds user-specified physical page range')
        source = attached.get('local_path')
        if not source or not Path(source).is_file():
            raise ValueError('visual component requires an existing local PDF')
        files = self.pdf.file_store
        pdf = files.cache_pdf_path(item.pdf_sha256)
        if not pdf.is_file() or sha256(pdf.read_bytes()).hexdigest() != item.pdf_sha256:
            raise ValueError('visual PDF cache missing or changed')
        if sha256(Path(source).read_bytes()).hexdigest() != item.pdf_sha256:
            raise ValueError('visual attachment version changed')
        # 与既有PDFProvider缓存键一致；先确认缓存存在，禁止回退重解析全文。
        config_hash = sha256(json.dumps(self.pdf.parser_config, sort_keys=True).encode()).hexdigest()
        cache_key = sha256(f'{item.pdf_sha256}:pymupdf:{self.pdf.parser_version}:{config_hash}'.encode()).hexdigest()
        cache = files.cache_parse_path(cache_key)
        if item.document_id != 'doc:' + cache_key[:32] or not cache.is_file():
            raise ValueError('visual parsed cache identity missing or changed')
        cached = json.loads(cache.read_text())
        page = next((p for p in cached['pages'] if p['page_number'] == item.physical_page), None)
        if not page or page['document_id'] != item.document_id or sha256(page['text'].encode()).hexdigest() != item.page_text_sha256:
            raise ValueError('visual page text identity mismatch')
        # 日常子节点已继承选中页，直接使用同版缓存；尤其不能因paper_id再次走网络获取。
        retained = self.db.get_page(page['evidence_id']) if self.db.task_has_evidence(task_id, page['evidence_id']) else None
        if retained:
            if retained['document_id'] != item.document_id or retained['text'] != page['text']:
                raise ValueError('visual authorized page differs from original cache')
            first = {'document_id': item.document_id, 'content_sha256': item.pdf_sha256, 'pages': [retained]}
        else:
            first = self.read(task_id, item.target_ref, item.physical_page, 1, attachments)
        if (first['document_id'], first['content_sha256']) != (item.document_id, item.pdf_sha256):
            raise ValueError('visual source identity changed while reading')
        text = first['pages'][0]['text']
        if sha256(text.encode()).hexdigest() != item.page_text_sha256:
            raise ValueError('visual text changed while reading')
        rendered = PaperMediaProvider(files).render(pdf, item.pdf_sha256, item.physical_page, item.bbox)
        manifest = {**item.model_dump(mode='json'), 'assets': rendered['assets'],
                    'render_options': rendered['render_options'], 'evidence_id': first['pages'][0]['evidence_id'],
                    'kind': 'model_observation', 'semantically_verified': False}
        return {'input': item.model_dump(mode='json'), 'text': text, 'assets': rendered['assets'],
                'manifest': manifest, 'render_seconds': rendered['render_seconds']}
