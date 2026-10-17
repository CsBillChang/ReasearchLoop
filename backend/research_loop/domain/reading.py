"""从用户原始指令提取明确的物理页限制，不把 Agent 的阅读批次当成权限。

这里只处理可确定的全任务连续页范围。不同论文有不同页要求等歧义情况
应先澄清；未指定范围时返回 None，表示全文开放。模型生成的子目标不参与解析。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class UserPageRange:
    """用户明确指定的闭区间和对应原话。

    Attributes:
        start: 从 1 起算的物理起始页。
        end: 包含在内的物理结束页。
        instruction: 用于定位限制来源的用户原话。
    """

    start: int
    end: int
    instruction: str


_NUMBER = r"\d+|[一二三四五六七八九十百两]+"
# 附件标题属于用户指令中的对象，不是阅读权限本身。只识别同一短句内
# “附件 <名称> [的] 物理 <连续页码>”，不吞跨句限制，不按论文名设置分支。
# “根据”和“依据”沿用通用前缀语义；可选助词只改变句式，不放宽页码校验。
_NAMED_ATTACHMENT_PREFIX = (
    r"(?:只|仅|限定)(?:使用|读取|阅读|参考|基于|根据|依据|看|读)?\s*"
    r"附件\s*[^。！？\n；;,，页]{1,120}?(?:的\s*)?物理\s*"
)
_PREFIX = (
    rf"(?:{_NAMED_ATTACHMENT_PREFIX}|"
    r"(?:只|仅|限定)(?:允许|能|可以|限于|限|使用|阅读|读取|看|读|依据|根据|基于|参考|已上传|上传|附件|论文|原文|物理|(?:每|各|两|\d+)篇|的|\s)*)"
)
_PATTERN = re.compile(
    rf"(?P<quote>{_PREFIX}(?:(?P<first>前)(?P<count>{_NUMBER})页|"
    rf"第?\s*(?P<start>{_NUMBER})(?:\s*(?:到|至|[-–—~～])\s*第?\s*(?P<end>{_NUMBER}))?\s*页))"
    rf"|(?P<english>\bonly\s+(?:read\s+)?(?:physical\s+)?pages?\s+"
    r"(?P<en_start>\d+)(?:\s*(?:-|–|to)\s*(?P<en_end>\d+))?)",
    re.IGNORECASE,
)

# 只隔开明确的出处标注短句，不按所有逗号切断：例如“只读，前面几页”
# 仍是未确定的阅读限制。页码后接“对应的内容”等对象时也不能当输出要求。
_PAGE_REFERENCE_REQUEST = re.compile(
    r"(?:^|[，,；;。！？\n]|(?:并且|并|且|同时))\s*(?:请\s*)?"
    r"(?:(?:只|仅)(?:需|要|需要)?\s*)?"
    r"(?:注明|标注|标明|标记|写明|附上|给出|提供|列出)\s*"
    r"(?:(?:对应|具体|准确|引用|原文|来源|证据|出处|物理|所在|文献|论文|其|的)\s*)*"
    r"(?:页码|页号|页(?![码号]))"
    r"(?!\s*(?:对应|所对应|的|之前|之后|以内|范围|内|中))"
)


def _number(value: str) -> int:
    """将阿拉伯数字或常用中文页码转换为整数。"""
    if value.isdigit():
        return int(value)
    digits = dict(zip("一二三四五六七八九两", [1, 2, 3, 4, 5, 6, 7, 8, 9, 2]))
    result, digit = 0, 0
    for char in value:
        if char in digits:
            digit = digits[char]
        else:
            result += (digit or 1) * {"十": 10, "百": 100}[char]
            digit = 0
    return result + digit


def user_page_range(instructions: list[str]) -> UserPageRange | None:
    """从原始任务及已提交的澄清回答提取显式限制。

    Args:
        instructions: 按时间排序的用户文本，不接受 Agent 分派或工具资料。

    Returns:
        唯一可确定的物理页范围；没有范围要求时为 None。

    Raises:
        ValueError: 用户给出冲突、非连续或无法确定的限制，需先澄清。
    """
    result = None
    for text in instructions:
        if re.search(r"(?:取消|解除)[^。！？\n]{0,10}页[^。！？\n]{0,5}限制|(?:不限制|不限)(?:阅读)?页(?:数|码|范围)", text):
            result = None
        matches = [match for match in _PATTERN.finditer(text)
                   if not re.search(r"(?:不要|不能|不应|不必|别|不)\s*$", text[max(0, match.start() - 6):match.start()])]
        if not matches:
            # 有明确限制措辞却无法解析时拒绝猜测，防止把限制误当成全文授权。
            cleaned = re.sub(r"(?:不要|不能|不应|不必|别|不)\s*" + _PREFIX + r"[^。！？\n]{0,40}?页", "", text)
            # 标注出处只影响输出；检查副本插入句界，防止把前句的“只依据”
            # 连到“注明页码”。原始指令及上方明确范围解析均保持不变。
            cleaned = _PAGE_REFERENCE_REQUEST.sub("。", cleaned)
            if (re.search(_PREFIX + r"[^。！？\n]{0,40}页", cleaned)
                    or re.search(r"(?:不要|禁止|不允许)(?:读|看|阅读|读取).{0,20}页", cleaned)
                    or re.search(r"\bonly\b.{0,30}\b(?:pages?|pp\.)", cleaned, re.I)):
                raise ValueError("unclear user page range; clarify the physical page range before reading")
            continue
        ranges = []
        for match in matches:
            start = 1 if match['first'] else _number(match['start'] or match['en_start'])
            end = _number(match['count'] or match['end'] or match['en_end'] or str(start))
            if start < 1 or end < start:
                raise ValueError("invalid user page range; clarify before reading")
            # 不能把“只读第1页和第3页”静默解释成只读第1页。
            if re.match(r"\s*(?:和|及|、|,|，|and)\s*第?\d+\s*页", text[match.end():], re.I):
                raise ValueError("non-contiguous user page range; clarify before reading")
            ranges.append(UserPageRange(start, end, match[0]))
        if len({(r.start, r.end) for r in ranges}) != 1:
            raise ValueError("conflicting user page ranges; clarify before reading")
        result = ranges[-1]
    return result
