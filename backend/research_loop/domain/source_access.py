"""来源访问边界：读取当前根指令与澄清，识别显式禁止联网和规范论文地址。"""
from __future__ import annotations

import re
from enum import Enum
from urllib.parse import parse_qs, urlsplit


SOURCE_HOSTS = {
    'arxiv': {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org'},
    'openreview': {'openreview.net', 'www.openreview.net'},
    'cvpr': {'openaccess.thecvf.com'}, 'iccv': {'openaccess.thecvf.com'},
    'eccv': {'ecva.net', 'www.ecva.net'}, 'acl': {'aclanthology.org'},
    'emnlp': {'aclanthology.org'}, 'naacl': {'aclanthology.org'},
    'iclr': {'openreview.net', 'proceedings.iclr.cc'},
    'neurips': {'papers.neurips.cc', 'proceedings.neurips.cc'},
    'icml': {'proceedings.mlr.press'}, 'aistats': {'proceedings.mlr.press'},
}


class WebSource(str, Enum):
    """网页结果筛选的已实现来源集合；不等同于用户授权或原RPC目录。"""

    arxiv = 'arxiv'
    openreview = 'openreview'
    cvpr = 'cvpr'
    iccv = 'iccv'
    eccv = 'eccv'
    acl = 'acl'
    emnlp = 'emnlp'
    naacl = 'naacl'
    iclr = 'iclr'
    neurips = 'neurips'
    icml = 'icml'
    aistats = 'aistats'


WEB_SOURCE_ALIASES = {'nips': 'neurips', 'openreview.net': 'openreview',
                      'www.openreview.net': 'openreview', 'arxiv.org': 'arxiv',
                      'www.arxiv.org': 'arxiv'}


class WebSourceValidationError(ValueError):
    """来源筛选参数在发送前被拒绝，不代表站点返回空结果或访问错误。"""


def source_error_diagnostic(error: str | Exception) -> dict:
    """返回无外部请求的本地参数诊断，不虚构HTTP状态或付费收据。"""
    return {'stage': 'parameter_validation', 'operation': None, 'request_state': 'not_requested',
            'receipt_id': None, 'error_code': 'WEB_SOURCE_INVALID', 'message': str(error),
            'supported_sources': [source.value for source in WebSource]}


def normalize_web_sources(sources: list[str] | None) -> list[str] | None:
    """归一化已声明别名，拒绝未知或尚无网页映射的来源。

    Args:
        sources: 可选网页结果筛选清单；None/空清单不增加工具筛选。

    Returns:
        按原顺序去重的稳定来源名；该结果不授予站点访问权限。

    Raises:
        WebSourceValidationError: 输入形状错误或来源尚不受支持。
    """
    if sources is None:
        return None
    if not isinstance(sources, list):
        raise WebSourceValidationError('sources must be a list of supported web source names')
    normalized = []
    for value in sources:
        if not isinstance(value, str):
            raise WebSourceValidationError('each source must be a supported web source name')
        key = value.strip().casefold()
        key = WEB_SOURCE_ALIASES.get(key, key)
        if key not in SOURCE_HOSTS:
            raise WebSourceValidationError('unsupported web source filter: ' + value + '; use: ' + ', '.join(source.value for source in WebSource))
        if key not in normalized:
            normalized.append(key)
    return normalized


# 只匹配明确的外搜动作或用户已提供材料对象；“不重新精读”等否定不代表禁止联网。
_EXPLICIT_EXTERNAL_RESTRICTION = re.compile(
    r'(?:不要|禁止|不得|不允许|不能|无需)\s*(?:再|额外|另外)?\s*'
    r'(?:搜索|检索|查找|查询|获取)\s*(?:(?:外部|站外)(?:的)?(?:文献|论文|资料|来源)|(?:网络|网上|互联网))'
    r'|(?:不要|禁止|不得|不允许|不能)\s*(?:再|额外)?\s*(?:去|到|从)?\s*'
    r'(?:网上|网络上|互联网上)\s*(?:找|搜索|检索|查找)'
    r'|(?:只|仅)\s*(?:根据|依据|基于|使用|参考|分析|阅读|读取)\s*'
    r'(?:我(?:已经)?(?:给|提供|上传)(?:你)?的|已(?:经)?(?:提供|上传)的?|所(?:提供|上传)的)'
    r'\s*(?:论文|文献|PDF|附件|文件|资料)'
    r'|\b(?:do not|don[\'’]t|never)\s+(?:search|browse|query|look\s+for)\s+'
    r'(?:(?:for|in|from)\s+)?(?:(?:external|outside)\s+(?:sources?|papers?|literature|references?|documents?)'
    r'|(?:online|web|internet)(?:\s+(?:sources?|papers?|literature|references?|documents?))?)\b'
    r'|\b(?:use|read|analy[sz]e|answer\s+(?:using|from))\s+(?:the\s+)?'
    r'(?:uploaded|provided|attached)\s+(?:papers?|pdfs?|documents?|files?)\s+only\b',
    re.IGNORECASE,
)


def current_instructions(db, task_id: str) -> tuple[str, list[str]]:
    """从持久根任务和已提交澄清读回权限，不从模型计划或长期记忆继承。"""
    task = db.get_task(task_id)
    if not task:
        raise PermissionError('external access requires a persisted current task')
    root_id = task.get('root_task_id') or task_id
    root = db.get_task(root_id)
    if not root:
        raise PermissionError('current root task is missing')
    answers = db.fetchall("SELECT answer FROM clarifications WHERE task_id=? AND status='answered' ORDER BY answered_at,clarification_id", (root_id,))
    return root_id, [root['prompt']] + [row['answer'] for row in answers if row.get('answer')]


# 材料排他与禁止外网是不同权限：“不要联网”不排除本地历史原文。
# 这里只识别明确的附件only句式，不从模型计划或“仅在对话回答”推导材料限制。
_EN_ATTACHMENTS = (r'(?:(?:one|two|three|\d+)\s+)?(?:attachments?|(?:(?:current|provided|uploaded|attached|these)\s+)+'
                   r'(?:attachments?|files?|pdfs?|documents?|papers?|materials?))')
_ATTACHMENT_ONLY = re.compile(
    r'(?:只|仅|限定)(?:能|允许|限于|限|使用|依据|根据|基于|参考|读取|阅读|分析|看|读|用|\s)*'
    r'(?:(?:我|本次|当前|这些|这份|这篇|该|此|已|已经|所|上传|提供|的)\s*)*'
    r'(?:[一二两三四五六七八九十\d]+(?:份|篇|个)\s*)?(?:附件|PDF|文件|资料|材料|论文)'
    r'|\bonly\s+(?:(?:use|read|analy[sz]e|refer\s+to|rely\s+on|based\s+on|from)\s+)?'
    r'(?:the\s+)?' + _EN_ATTACHMENTS + r'\b'
    r'|\b(?:use|read|analy[sz]e|refer\s+to|rely\s+on)\s+(?:the\s+)?'
    + _EN_ATTACHMENTS + r'\s+only\b', re.I,
)
_ATTACHMENT_ONLY_RELEASE = re.compile(
    r'(?:解除|取消)(?:当前|本次)?(?:仅|只)?(?:附件|上传材料)(?:范围|材料)?限制'
    r'|\b(?:lift|remove|cancel)\s+(?:the\s+)?attachment[- ]only\s+(?:restriction|limit|scope)\b', re.I,
)
# 解除会扩大后续权限，仅接受句首或明确肯定前缀；修饰语/否定不明时保留限制。
_AFFIRMATIVE_RELEASE_PREFIX = re.compile(
    r'^(?:(?:请|现在|本次|这次|可以|允许)\s*)*$'
    r'|^(?:(?:please|now|you\s+may)\s+)*$', re.I,
)
_NEGATED_ONLY = re.compile(
    r'(?:不要|不能|不应|不必|无需|别|不再|不是|并非|解除|取消)(?:当前|本次)?\s*$'
    r'|\b(?:do\s+not|don[\'’]t|must\s+not|not|never|no\s+longer)\s+'
    r'(?:(?:use|read|analy[sz]e|refer\s+to|rely\s+on)\s+)?$', re.I,
)


def attachment_only(instructions: list[str]) -> bool:
    """判断当前用户是否明确限制仅用附件，不把附件存在当作排他授权。

    Args:
        instructions: 当前根问题和按时序排列的真实已回答澄清。

    Returns:
        有效的明确附件only限制；否定only或真实解除可取消之前限制。
    """
    restricted = False
    for instruction in instructions:
        events = []
        releases = list(_ATTACHMENT_ONLY_RELEASE.finditer(instruction))
        for match in releases:
            prefix = re.split(r'[。！？；;\n]', instruction[:match.start()])[-1].lstrip()
            # 不猜测不明解除意图；解除短语内嵌的“仅附件”也不能成为反向事件。
            if _AFFIRMATIVE_RELEASE_PREFIX.fullmatch(prefix):
                events.append((match.start(), False))
        for match in _ATTACHMENT_ONLY.finditer(instruction):
            if any(release.start() <= match.start() < release.end() for release in releases):
                continue
            # 只在同一短句内判断否定，避免把前句的“不要联网”挪给附件限制。
            prefix = re.split(r'[。！？；;\n]', instruction[:match.start()])[-1][-50:]
            events.append((match.start(), not bool(_NEGATED_ONLY.search(prefix))))
        if events:
            restricted = sorted(events)[-1][1]
    return restricted


def assert_external_access(db, task_id: str) -> str:
    """拒绝明确仅附件/禁止外搜；存在附件本身不构成禁止联网。"""
    root_id, instructions = current_instructions(db, task_id)
    denied = False
    for instruction in instructions:
        # 仅由后续用户明确解除限制，附件内容和角色目标均不参与解析。
        if re.search(r'(?:解除|取消)(?:联网|外搜|外部搜索|仅附件)[^。\n]{0,6}限制|允许(?:联网|外搜|外部搜索)|\b(?:allow|enable)\s+(?:external|web|internet)\s+(?:search|access)\b', instruction, re.I):
            denied = False
        if (_EXPLICIT_EXTERNAL_RESTRICTION.search(instruction)
                or re.search(r'\b(?:do not|don.t|never)\s+(?:search|browse|access)\s+(?:the\s+)?(?:web|internet|network)\b|\bonly\s+(?:from|use|read)\s+(?:this|the|provided|attached|uploaded)\s+(?:pdf|documents?|files?)\b|(?:禁止|不要|不允许|不得|不能)(?:进行|额外)?(?:搜索|检索|查找)(?:网络|网页|互联网)|(?:仅|只)(?:使用|依据|根据|基于|参考|读取|阅读|读|看)(?:这份|这篇|这个|该|提供的|上传的|此)(?:PDF|pdf|文件|资料|论文)', instruction, re.I)
                or re.search(r'\b(?:use|read|refer to|based on)?\s*only\s+(?:the\s+)?(?:attachments?|attached|uploaded|provided|local)\b|\b(?:do not|don.t|never)\s+browse\b|(?:禁止|不要|不允许|不得|不能)(?:额外|进行|使用|访问)*(?:联网|外部|网络)(?:搜索|检索|查找|访问)?', instruction, re.I)
                or re.search(r'(?:禁止|不要|不允许|不得|不能|无需)(?:联网|外搜|外部搜索|网络搜索)|(?:仅|只)(?:能|允许|使用|依据|根据|基于|参考|读取|阅读|读|看|用|分析|已上传|上传|的|\s)*(?:附件|上传的(?:文件|PDF)|本地(?:文件|资料))|\b(?:no|without|do\s+not|don.t|never)\s+(?:(?:use|perform|do)\s+)?(?:external|web|internet|online)\s*(?:search|access)?\b|\b(?:only\s+(?:use|read|based\s+on)\s+(?:the\s+)?(?:uploaded|attached|attachments?|local)|(?:attachments?|uploaded\s+files?)\s+only)\b', instruction, re.I)):
            denied = True
    if denied:
        raise PermissionError('current user instructions forbid external search/access; use authorized attachments only')
    return root_id


def canonical_paper_url(url: str) -> tuple[str, str, str | None] | None:
    """提取官方论文地址的稳定身份；任意网页、域名和摘要匹配均不证明身份。"""
    parsed = urlsplit(url)
    if parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None
    host, path = parsed.hostname.lower(), parsed.path
    if host in SOURCE_HOSTS['arxiv']:
        match = re.fullmatch(r'/(?:abs|pdf)/((?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7}))(v\d+)?(?:\.pdf)?/?', path)
        if match:
            return 'arxiv', match[1], match[2]
    if host in {'openreview.net', 'www.openreview.net'} and path in {'/forum', '/pdf'}:
        identifier = parse_qs(parsed.query).get('id', [''])[0]
        if re.fullmatch(r'[A-Za-z0-9_-]+', identifier):
            return 'openreview', identifier, None
    patterns = {
        'openaccess.thecvf.com': r'/content/[^?#]+/(?:html|papers)/[^/]+_paper\.(?:html|pdf)',
        'aclanthology.org': r'/(?:[12]\d{3}\.[A-Za-z0-9_-]+\.\d+|[A-Za-z]\d{2}-\d{4})(?:\.pdf)?/?',
        'proceedings.mlr.press': r'/v\d+/[^/]+(?:\.html|\.pdf|/[^/]+\.pdf)',
        'papers.neurips.cc': r'/paper(?:_files)?/paper/\d{4}/(?:hash|file)/[^/]+(?:\.html|\.pdf)',
        'proceedings.neurips.cc': r'/paper(?:_files)?/paper/\d{4}/(?:hash|file)/[^/]+(?:\.html|\.pdf)',
        'proceedings.iclr.cc': r'/paper_files/paper/\d{4}/(?:hash|file)/[^/]+(?:\.html|\.pdf)',
        'ecva.net': r'/papers/[^?#]+/(?:html|papers|pdf)/[^/]+(?:\.php|\.html|\.pdf)',
        'www.ecva.net': r'/papers/[^?#]+/(?:html|papers|pdf)/[^/]+(?:\.php|\.html|\.pdf)',
    }
    if host in patterns and re.fullmatch(patterns[host], path):
        # 精确官方论文路径才是候选；作者主页/新闻等不得取得原文权限。
        return host, path, None
    return None


def source_allowed(url: str, sources: list[str] | None) -> bool:
    """显式来源集合只允许已映射域名；未知来源不能变成全网授权。"""
    sources = normalize_web_sources(sources)
    if not sources:
        return True
    host = (urlsplit(url).hostname or '').lower()
    allowed = set().union(*(SOURCE_HOSTS[source] for source in sources))
    if host not in allowed:
        return False
    selected = {source.lower() for source in sources}
    path = urlsplit(url).path
    if host == 'openaccess.thecvf.com':
        venue = re.match(r'/content/(CVPR|ICCV)(?:\d{4}|_\d{4})/', path, re.I)
        return bool(venue and venue[1].lower() in selected)
    if host == 'aclanthology.org':
        venue = re.match(r'/\d{4}\.(acl|emnlp|naacl)[-.]', path, re.I)
        return bool(venue and venue[1].lower() in selected)
    return True


def same_paper_locator(original: str, alternate: str, version: str | None = None) -> bool:
    """仅认可可确定同一地址/同一arXiv版本；模糊标题相似度不能授权替换。"""
    if original == alternate:
        anchor = canonical_paper_url(original)
        if version and anchor and anchor[0] == 'arxiv':
            return anchor[2] == (version if version.startswith('v') else 'v' + version)
        return not version or bool(anchor and anchor[2] == version)
    first, second = canonical_paper_url(original), canonical_paper_url(alternate)
    if not first or not second or first[:2] != second[:2]:
        return False
    if first[0] == 'arxiv':
        required = version if not version or version.startswith('v') else 'v' + version
        return first[2] == second[2] and (not required or second[2] == required)
    return first[0] == 'openreview' and not version


def exact_material_required(db, task_id: str, candidate) -> bool:
    """用户显式页/版本要求禁止跨载体版本；普通已带版本题录也不静默替换。"""
    from research_loop.services.papers import PaperService
    _, instructions = current_instructions(db, task_id)
    if PaperService(db, None, None).user_scope(task_id, target_ref=candidate.paper_id) is not None:
        return True
    return any(re.search(r'(?:只|仅|必须|指定|严格|exact|only)[^。\n]{0,30}(?:版本|version|v\d+)|(?:arxiv\.org/(?:abs|pdf)/[^\s]*v\d+)|(?:版本|version)\s*[vV]?\d+', text, re.I)
               for text in instructions)


def alternate_metadata_matches(candidate, title: str, content: str) -> bool:
    """仅在官方候选详情题名精确、原DOI或作者完整匹配时确认同论文。

    同论文不等于同版本；本函数不赋予页码或原版替换权限。
    """
    normalized = lambda value: re.sub(r'\W+', '', value, flags=re.UNICODE).casefold()
    if normalized(candidate.title) != normalized(title):
        return False
    header = content[:3000].casefold()
    if candidate.doi:
        doi = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', candidate.doi.casefold())
        return bool(re.search(r'(?<![\w.])' + re.escape(doi) + r'(?![\w./])', header))
    if not candidate.authors:
        return False
    return all(re.search(r'(?<!\w)' + re.escape(author.casefold()) + r'(?!\w)', header)
               for author in candidate.authors)


def explicit_source_hosts(db, task_id: str) -> set[str] | None:
    """识别用户明确的仅某域名要求，不能将发表场所默认解释为下载域名限制。"""
    _, instructions = current_instructions(db, task_id)
    hosts = None
    for text in instructions:
        matches = re.findall(r'(?:只能|仅能|仅允许|只允许|仅从|只从|仅使用|只使用|只在|仅在|\bonly(?: use| fetch| access| download)?(?: from)?)\s*(?:https?://)?([a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,})(?:[/:]|\b)', text, re.I)
        if matches:
            hosts = {host.lower() for host in matches}
    return hosts


def matched_detail_title(candidate, url: str, title: str, content: str) -> str:
    """只去掉官方已知标题外壳，或用精确首标题加DOI/作者证明主文身份。"""
    normalized = lambda value: re.sub(r'\W+', '', value, flags=re.UNICODE).casefold()
    def strip_wrapper(value, locator):
        """仅剥离当前规范arXiv身份对应的前缀和固定站点后缀。"""
        anchor = canonical_paper_url(locator)
        if anchor and anchor[0] == 'arxiv':
            identifier = re.escape(anchor[1] + (anchor[2] or ''))
            value = re.sub(r'^\s*\[' + identifier + r'\]\s*', '', value, flags=re.I)
            value = re.sub(r'\s*(?:\||[-–—])\s*arxiv(?:\.org)?\s*$', '', value, flags=re.I)
        return value
    expected = strip_wrapper(candidate.title, candidate.paper_url or '')
    actual = strip_wrapper(title, url)
    if normalized(expected) == normalized(actual):
        return candidate.title
    heading = re.search(r'^#{1,2}\s+([^\n]+)', content[:3000], re.M)
    if heading and normalized(expected) == normalized(strip_wrapper(heading[1], url)):
        if alternate_metadata_matches(candidate, candidate.title, content):
            return candidate.title
    return ''


def same_pdf_for_detail(detail_url: str, pdf_url: str) -> bool:
    """绑定官方详情与主文PDF的确定性路径；同站其他论文也不能冒充。"""
    from research_loop.providers.paper_acquisition import resolve_pdf_url
    anchor = canonical_paper_url(detail_url)
    if not anchor:
        return False
    if anchor[0] in {'arxiv', 'openreview'}:
        return same_paper_locator(detail_url, pdf_url, anchor[2])
    first, second = urlsplit(detail_url), urlsplit(pdf_url)
    if first.hostname != second.hostname:
        return False
    resolved = resolve_pdf_url(None, detail_url)
    if resolved != detail_url or first.path.endswith('.pdf'):
        return urlsplit(resolved).path == second.path and urlsplit(resolved).query == second.query
    if first.hostname == 'proceedings.mlr.press':
        detail = re.fullmatch(r'/v(\d+)/([^/]+)\.html', first.path)
        if detail:
            return second.path in {f'/v{detail[1]}/{detail[2]}.pdf', f'/v{detail[1]}/{detail[2]}/{detail[2]}.pdf'}
    if first.hostname in {'papers.neurips.cc', 'proceedings.neurips.cc', 'proceedings.iclr.cc'}:
        detail = re.fullmatch(r'(/paper(?:_files)?/paper/\d{4})/hash/([^/]+)-Abstract(?:-Conference)?\.html', first.path)
        if detail:
            return second.path in {detail[1] + '/file/' + detail[2] + '-Paper.pdf',
                                   detail[1] + '/file/' + detail[2] + '-Paper-Conference.pdf'}
    return False


def final_pdf_locator(expected_url: str, final_url: str, restricted_hosts: set[str] | None = None) -> dict:
    """核对实际下载终点；明确异论文/异版本或越出用户域名时不得授予页证据。"""
    expected, final = canonical_paper_url(expected_url), canonical_paper_url(final_url)
    details = {'expected_url': expected_url, 'final_url': final_url,
               'expected_identity': list(expected) if expected else None,
               'final_identity': list(final) if final else None}
    if restricted_hosts and (urlsplit(final_url).hostname or '').lower() not in restricted_hosts:
        return {**details, 'status': 'outside_explicit_host_scope'}
    if expected and final:
        matches = expected == final or same_pdf_for_detail(expected_url, final_url)
        return {**details, 'status': 'same_official_locator' if matches else 'paper_or_version_mismatch'}
    return {**details, 'status': 'unconfirmed_final_locator'}
