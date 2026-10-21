"""论文地址适配：定位 PDF，不解析正文或搜索替代论文。

以下声明适用于本模块改编的第三方部分，不改变其他代码或依赖的许可：
Portions adapted from daily-paper-reader by ziwenhahaha.
Source: https://github.com/ziwenhahaha/daily-paper-reader

MIT License

Copyright (c) 2026 ziwenhahaha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re
from urllib.parse import parse_qs, urljoin, urlsplit


class PaperAcquisitionError(ValueError):
    """携带获取层错误码，避免把下载失败误报为论文身份非法。"""

    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 acquisition: dict | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.acquisition = acquisition or {}


def access_challenge_reason(url: str, content: str = "", *, content_type: str = "",
                            title: str = "") -> str | None:
    """识别访问验证页面，不把论文正文出现 captcha 一词当作阻断。

    Args:
        url: 实际响应地址，而非被代理访问的论文地址。
        content: HTML 或 Fetch/Jina 返回的可见页面内容。
        content_type: 实际响应的 Content-Type；旧诊断可没有正文。
        title: Fetch 返回的页面标题，仅作验证页结构信号。

    Returns:
        明确的验证页原因，证据不足时返回 None。
    """
    value = str(content or "").strip()
    if value.startswith("%PDF-"):
        return None
    parsed = urlsplit(str(url or ""))
    is_html = "html" in str(content_type).lower() or bool(re.match(r"(?:<!doctype\s+html|<html\b)", value, re.I))
    challenge_path = ((parsed.hostname in {"openreview.net", "www.openreview.net"} and parsed.path.rstrip("/") == "/challenge")
                      or bool(re.match(r"/cdn-cgi/challenge-platform(?:/|$)", parsed.path, re.I)))
    # 旧轨迹没有响应正文，但真实最终验证地址及 HTML 类型已经足以区分详情页。
    if is_html and challenge_path:
        return "redirected_challenge_endpoint"
    headings = [str(title or "")]
    headings.extend(re.findall(r"<(?:title|h1)\b[^>]*>(.*?)</(?:title|h1)>", value[:12000], re.I | re.S))
    jina_title = re.match(r"Title:\s*([^\n]+)", value, re.I)
    if jina_title:
        headings.append(jina_title.group(1))
    verification_heading = re.compile(
        r"(?:just a moment[.!…]*|security (?:verification|check)|verify (?:that )?you are human|"
        r"human verification|checking your browser|access denied|sign in|log in|人机验证|安全验证|访问被拒绝)", re.I)
    directive = re.compile(r"(?:verify (?:that )?you are human|checking your browser|"
                           r"enable javascript(?: and cookies)?|complete (?:the |this )?(?:captcha|security check)|"
                           r"sign in to (?:view|access)|人机验证|请完成验证|访问被拒绝|登录后访问)", re.I)
    for heading in headings:
        plain = " ".join(unescape(re.sub(r"<[^>]+>", "", heading)).split())
        if verification_heading.fullmatch(plain) and directive.search(value[:4000]):
            return "verification_page"
    # Fetch 可能只提供纯文本验证提示，必须匹配完整首行。论文标题的学术副标题
    # （如 Verify You Are Human: A Survey ...）不是让访问者执行验证的页面指令。
    visible = unescape(re.sub(r"<[^>]+>", " ", value[:1000])).strip()
    first_line = re.sub(r"^#{1,6}\s+", "", visible.splitlines()[0]).strip() if visible else ""
    if re.fullmatch(r"(?:verify (?:that )?you are human(?: to continue)?|checking your browser|"
                    r"enable javascript and cookies|sign in to (?:view|access)|人机验证|登录后访问)[.!。！…]*", first_line, re.I):
        return "verification_prompt"
    return None


def blocked_primary_hosts(primary_failure: dict | None) -> set[str]:
    """从真实原址诊断提取受阻域名，兼容旧验证页失败且不迁移 Jina 错误。

    Args:
        primary_failure: 原址失败包装，含 error_code 与 acquisition 下载诊断。

    Returns:
        不应通过 Fetch 代理再次尝试的原址域名。仅查询实际下载 attempts，
        不读取 jina_attempts；旧 PDF_LINK_UNAVAILABLE 也按最终验证页识别。
    """
    failure = primary_failure or {}
    acquisition = failure.get("acquisition") or {}
    hosts: set[str] = set()
    for attempt in [acquisition] + list(acquisition.get("attempts") or []):
        if attempt.get("provider") == "jina":
            continue
        final_url = attempt.get("final_url") or ""
        challenge = (attempt.get("failure_kind") == "access_challenge"
                     or (attempt is acquisition and failure.get("error_code") == "PDF_ACCESS_CHALLENGE")
                     or access_challenge_reason(final_url, content_type=attempt.get("content_type", "")))
        denied = attempt.get("status", attempt.get("http_status")) in {401, 403}
        if not (challenge or denied):
            continue
        urls = [final_url or attempt.get("requested_url") or attempt.get("requested_source")]
        if challenge:
            # 原址被重定向至验证页时，原域名和验证终点均不交给代理原样再试。
            urls.extend([attempt.get("requested_url"), attempt.get("requested_source")])
        for url in urls:
            host = urlsplit(str(url or "")).hostname
            if host and host != "r.jina.ai":
                hosts.add(host)
    return hosts


def resolve_pdf_url(pdf_url: str | None, paper_url: str | None) -> str:
    """优先使用显式 PDF，否则转换已知来源的详情地址。

    Args:
        pdf_url: 上游直接返回的 PDF 地址；不重新猜测或改变版本。
        paper_url: 同一候选的详情页地址。

    Returns:
        PDF 地址或尚需读取的详情页；两者缺失时返回空字符串。
    """
    if pdf_url and pdf_url.strip():
        return pdf_url.strip()
    link = (paper_url or "").strip()
    if not link:
        return ""
    parsed = urlsplit(link)
    host, path = (parsed.hostname or "").lower(), parsed.path
    if host in {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org'} and not parsed.username and not parsed.password:
        # 同一论文同一显式版本的确定性地址转换，不搜索替代论文或去掉vN。
        match = re.fullmatch(r'/(?:abs|pdf)/((?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?)(?:\.pdf)?/?', path)
        if match and parsed.scheme in {'http', 'https'}:
            return parsed._replace(path='/pdf/' + match.group(1), fragment='').geturl()
    if path.lower().endswith(".pdf") or (path == "/pdf" and parsed.query):
        return link
    if host in {"openreview.net", "www.openreview.net"} and path == "/forum":
        identifier = parse_qs(parsed.query).get("id", [""])[0]
        if identifier:
            from urllib.parse import urlencode
            return "https://openreview.net/pdf?" + urlencode({"id": identifier})
    if host == "openaccess.thecvf.com" and "/html/" in path:
        return link.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf")
    if host in {"ecva.net", "www.ecva.net"} and "/html/" in path:
        match = re.search(r"/html/(\d+)_", path)
        if match:
            base = link[:link.index("/html/")]
            return f"{base}/papers/{match.group(1).zfill(5)}.pdf"
        return link.replace("/html/", "/pdf/").replace("_paper.php", "_paper.pdf")
    if host == "aclanthology.org":
        return parsed._replace(path=path.rstrip("/") + ".pdf").geturl()
    return link


class _PaperLinks(HTMLParser):
    """只收集明确 PDF 元数据和主文链接，不把补充材料当正文。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._label: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "meta" and (attributes.get("name") or "").lower() == "citation_pdf_url":
            self.metadata.append(attributes.get("content") or "")
        if tag == "a":
            self._href, self._label = attributes.get("href"), []

    def handle_data(self, data):
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(" ".join(self._label).split()).lower()))
            self._href, self._label = None, []


def pdf_from_detail(html: str, page_url: str) -> str:
    """提取页面明确指向的主文 PDF，优先 citation_pdf_url。

    Args:
        html: 已获取的详情页 HTML。
        page_url: 最终页面地址，用于解析相对链接。

    Returns:
        唯一主文 PDF 地址；没有明确候选或存在歧义时返回空字符串。

    Notes:
        保留官方站点与 mlresearch GitHub 分发路径，额外允许页面
        自身域名以适配其余会议来源。禁止任意跨站链接或补充材料冒充主文。
    """
    parser = _PaperLinks()
    parser.feed(html)
    host = urlsplit(page_url).hostname
    allowed = {host, "proceedings.iclr.cc", "papers.neurips.cc", "proceedings.neurips.cc",
               "proceedings.mlr.press", "openaccess.thecvf.com", "www.ecva.net", "ecva.net",
               "aclanthology.org", "openreview.net"}

    def candidates(values):
        """验证来源和文件类型，保持首次出现顺序并去重。"""
        output = []
        for value in values:
            if not value:
                continue
            target = urljoin(page_url, value)
            parsed = urlsplit(target)
            mlresearch = parsed.hostname == "raw.githubusercontent.com" and parsed.path.startswith("/mlresearch/")
            is_pdf = (parsed.path.lower().endswith(".pdf")
                      or (parsed.hostname == "openreview.net" and parsed.path == "/pdf")
                      or (parsed.hostname in {'arxiv.org', 'www.arxiv.org', 'export.arxiv.org'}
                          and bool(re.fullmatch(r'/pdf/[\w.\-/]+', parsed.path))))
            if (parsed.scheme in {"http", "https"} and not parsed.username and not parsed.password
                    and (parsed.hostname in allowed or mlresearch) and is_pdf
                    and not re.search(r"(?:supplement|supp\b|appendix)", parsed.path, re.I)):
                if target not in output:
                    output.append(target)
        return output

    found = candidates(parser.metadata)
    if found:
        return found[0] if len(found) == 1 else ""
    labels = {"paper", "pdf", "download pdf", "[pdf]", "paper pdf", "full paper", "full text pdf"}
    found = candidates(href for href, label in parser.links if label in labels)
    return found[0] if len(found) == 1 else ""
