"""上下文摘录纯函数：保留原句和省略标识，不生成新事实。"""
def excerpt(text: str, query: str, cap: int = 800) -> str:
    """抽取头尾与相关句；只摘录原文，不产生新的摘要事实。"""
    from research_loop.storage.indexes import terms
    if len(text) <= cap:
        return text
    lines = text.splitlines()
    keywords = set(terms(query))
    relevant = sorted(enumerate(lines),key=lambda x:-len(keywords & set(terms(x[1]))))[:2]
    middle = '\n'.join(line[:cap//4] for _,line in sorted(relevant))
    return text[:cap//4]+'\n[摘录省略；可读回原记录]\n'+middle+'\n'+text[-cap//4:]

