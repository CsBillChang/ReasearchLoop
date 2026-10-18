"""论文地址适配：定位 PDF，不解析正文或搜索替代论文。

涉及第三方适配的版权和许可声明见仓库根 NOTICE。"""
from __future__ import annotations

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
            is_pdf = parsed.path.lower().endswith(".pdf") or (parsed.hostname == "openreview.net" and parsed.path == "/pdf")
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
