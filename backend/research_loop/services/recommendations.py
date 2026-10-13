"""可解释的候选排序与简报渲染；仅使用元数据，不生成全文结论或被引量。"""
import hashlib
import json
import re
from urllib.parse import urlparse
from research_loop.storage.indexes import terms


def paper_identity(paper: dict) -> str:
    """按明确DOI或来源ID及版本去重；不以标题相似度合并不同论文。

    Args:
        paper: DPR规范候选，包含source、paper_id、可选doi/version。

    Returns:
        适于持久主键的SHA256；缺失版本仍明确标记unknown，不虚构版本。
    """
    doi = (paper.get('doi') or '').strip().lower().removeprefix('https://doi.org/')
    pid = str(paper.get('source_id') or paper['paper_id'])
    version = str(paper.get('version') or '')
    if paper['source'] == 'arxiv' and (match := re.search(r'v(\d+)$', pid)):
        version = version or match[1]
        pid = pid[:match.start()]
    identity = [doi or paper['source'] + ':' + pid, version or 'unknown']
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def rank_candidates(candidates: list[dict], config: dict, profile: dict, delivered: set,
                    user_read: set, deep_read: set) -> list[dict]:
    """结合查询、有效画像和阅读历史排序；相同已交付版本不重复占用五篇限额。"""
    explicit = profile.get('explicit_preferences', []) if config['use_profile'] else []
    slots = {x['slot'] for x in explicit}
    inferred = [x for x in profile.get('inferred_interests', []) if x['slot'] not in slots] if config['use_profile'] else []
    ranked = {}
    for paper in candidates:
        identity = paper_identity(paper)
        if identity in delivered:
            continue
        text_terms = set(terms(paper['title'] + ' ' + (paper.get('abstract') or '')))
        query_matches = [q for q in paper.get('_queries', []) if q in config['directions']]
        if not query_matches or paper['source'] not in config['sources']:
            continue
        reasons = ['DPR查询匹配：' + '、'.join(query_matches)]
        score = 10.0 * len(query_matches)
        used = []
        for entries, weight, label in ((explicit, 4, '明确偏好'), (inferred, 1, '可撤销的推断兴趣')):
            for entry in entries:
                if text_terms & set(terms(entry['content'])):
                    score += weight
                    used.append({'memory_id':entry['memory_id'], 'version':entry['version'], 'kind':entry['kind']})
                    reasons.append(label + '匹配：' + entry['content'])
        was_deep_read = paper['paper_id'] in deep_read
        was_read = identity in user_read
        if was_read:
            score -= 6
            reasons.append('你曾标为已读，排序降低')
        elif was_deep_read:
            score -= 3
            reasons.append('已生成过精读笔记，优先展示其他候选；不代表你已读')
        else:
            score += 2
            reasons.append('尚未交付过此来源身份/版本')
        item = {'identity':identity, 'paper':paper, 'score':score, 'reasons':reasons,
                'memory_sources':used, 'deep_read':was_deep_read, 'user_read':was_read,
                'abstract_source':'DPR上游摘要元数据；未读取全文'}
        if identity not in ranked or score > ranked[identity]['score']:
            ranked[identity] = item
    return sorted(ranked.values(), key=lambda x:(-x['score'], -(int(re.sub(r'\D','', x['paper'].get('published') or '')[:8] or 0)), x['identity']))[:config['limit']]


def safe_link(value: str | None) -> str | None:
    """只渲染HTTP(S)链接，避免把上游元数据用作可执行URL。"""
    if value and urlparse(value).scheme in {'http','https'}:
        return value.replace(' ', '%20').replace(')', '%29').replace('(', '%28')
    return None


def markdown_text(value: str) -> str:
    """将不可信上游字段当作普通文字，禁止嵌入指令式HTML/Markdown。"""
    import html
    return re.sub(r'([\\`*_\[\]{}#|!])', r'\\\1', html.escape(value)).replace('\n',' ')


def render_brief(config: dict, items: list[dict], coverage: list[dict], period: dict) -> str:
    """生成聊天可读简报；摘要来自上游，不声称已阅读全文或遍历来源全部论文。"""
    lines = ['## '+markdown_text(config['title']), f"检索日期窗口：{period['from']} 至 {period['to']}。候选 {len(items)}/{config['limit']} 篇。",
             '以下为上游摘要元数据与推荐理由，未下载全文、未生成精读笔记。']
    if not items:
        lines.append('本次没有可交付的新候选：可能没有命中、已交付过，或来源不可用；具体见覆盖记录。')
    for number, item in enumerate(items,1):
        p = item['paper']
        lines += [f"### {number}. {markdown_text(p['title'])}",
                  f"来源：{p['source']} · 日期：{p.get('published') or '未知'}（{p.get('date_precision','unknown')}） · 版本：{p.get('version') or '上游未注明'}",
                  '**摘要（上游元数据）**：'+markdown_text(p.get('abstract') or '上游未提供摘要，不能据此推断全文内容。'),
                  '**推荐理由**：'+markdown_text('；'.join(item['reasons']))]
        url = safe_link(p.get('paper_url')) or safe_link(p.get('pdf_url'))
        if url:
            lines.append('[论文来源链接]('+url+')')
    lines += ['### 来源覆盖与缺口', '仅说明这些关键词/日期窗口的有界检索；不保证上游收录完整或后补录全部可见。每日重叠一天去重。']
    labels = {'queried_success':'已检索，有候选','queried_empty':'已检索，返回为空','failed':'检索失败','not_configured':'来源未配置','partial_results':'窗口未完整执行','truncated':'达到候选上限，保留窗口缺口','budget_exhausted':'本次预算耗尽，未检索'}
    for row in coverage:
        lines.append(f"- {row['source']} / {markdown_text(row['query'])}：{labels.get(row['status'], row['status'])}；{row['from']}—{row['to']}。" + (' '+markdown_text(row.get('error','')) if row.get('error') else ''))
    return '\n\n'.join(lines)
