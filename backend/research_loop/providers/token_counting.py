"""共享离线 Token 估算；实际供应商 usage 才是账单真值，不发送网络请求。

官方规格与估算说明：api-docs.deepseek.com/quick_start/{pricing,token_usage}/。
1M 按保守十进制1000000解释；384K由官方max_tokens合同393216确认。
未验证旧版公开 tokenizer 与 Flash-1017 兼容，不自动下载或启用。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import base64
import hashlib
import json
import math
import struct

_ACTIVE = ContextVar('researchloop_token_counting', default=None)


def model_capabilities(model: str) -> dict:
    """返回已核对能力；未知或旧名称保持旧128K的保守默认。"""
    flash = model in {'deepseek-flash', 'deepseek-v4-flash', 'deepseek-v4-flash-vision-exp'}
    return {'context_window_tokens': 1000000 if flash or model == 'deepseek-v4-pro' else 128000,
            'max_output_tokens': 393216 if flash or model == 'deepseek-v4-pro' else 8192,
            'published_context': '1M' if flash or model == 'deepseek-v4-pro' else '128K legacy/unknown fallback',
            'profile': 'DeepSeek-V4.1-Flash-1017' if flash else ('DeepSeek-V4-Pro-0813' if model == 'deepseek-v4-pro' else 'legacy_or_unknown')}


def counting_policy(settings) -> dict:
    """冻结计量配置与本地 tokenizer 身份；只有显式兼容声明才启用。"""
    path = settings.token_counting_tokenizer_path
    result = {'version': 2, 'model': settings.deepseek_model, 'method': 'conservative_character_estimate_v2',
              'is_exact': False, 'accuracy': 'heuristic_not_a_guaranteed_upper_bound',
              'context_window_tokens': settings.context_window_tokens,
              'context_output_reserve': settings.context_output_reserve,
              'context_safety_tokens': settings.context_safety_tokens,
              'context_compress_ratio': settings.context_compress_ratio,
              'model_max_output_tokens': settings.model_max_output_tokens,
              'capability': model_capabilities(settings.deepseek_model)}
    if path:
        if settings.token_counting_tokenizer_model != settings.deepseek_model:
            raise ValueError('TOKENIZER_COMPATIBILITY: 必须明确声明本地tokenizer与当前模型兼容')
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if settings.token_counting_tokenizer_sha256 and settings.token_counting_tokenizer_sha256 != sha:
            raise ValueError('TOKENIZER_IDENTITY: 本地tokenizer hash不匹配')
        result.update(method='local_tokenizer_json_plus_format_estimate_v2', tokenizer_path=str(Path(path).resolve()),
                      tokenizer_sha256=sha, tokenizer_model=settings.token_counting_tokenizer_model,
                      accuracy='local_text_tokenization_provider_format_estimated')
        _tokenizer(result['tokenizer_path'], sha)
    return result


@lru_cache(maxsize=4)
def _tokenizer(path: str, sha: str):
    """只加载明确指定的本机 tokenizer.json；缺依赖或文件变化时停止。"""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError('TOKENIZER_IDENTITY: 冻结tokenizer内容改变')
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError('TOKENIZER_DEPENDENCY: 本地tokenizer需要已安装tokenizers') from exc
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


@contextmanager
def use_policy(policy: dict | None):
    """在当前角色调用内传播统一口径，线程之间互不污染。"""
    token = _ACTIVE.set(policy)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _dump(value) -> str:
    """序列化当前模型可见值，包含工具schema和JSON转义开销。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _text_tokens(text: str, policy: dict) -> int:
    """离线分词或留有余量的字符估算；不声称保证大于实际Token数。"""
    if policy.get('tokenizer_path'):
        path, sha = policy['tokenizer_path'], policy['tokenizer_sha256']
        # 缓存不能绕过冻结身份检查。
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('TOKENIZER_IDENTITY: 冻结tokenizer内容改变')
        return len(_tokenizer(path, sha).encode(text, add_special_tokens=False).ids)
    # 官方约0.3/英文字符、0.6/中文字符仅为平均数。使用更保守的0.5/1.5，
    # 其他Unicode按2/字符留余量；随机代码/罕见字符仍可能偏差，不能冒称上界。
    ascii_chars = sum(ord(c) < 128 for c in text)
    cjk = sum('\u3400' <= c <= '\u9fff' for c in text)
    return math.ceil(ascii_chars * .5 + cjk * 1.5 + (len(text)-ascii_chars-cjk) * 2)


def _project_images(value, model):
    """校验内联PNG并替换估算副本；传输限制始终使用字节而非Token。"""
    projected, images = deepcopy(value), 0
    for message in projected.get('messages', []):
        if not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if not isinstance(block, dict) or block.get('type') != 'image_url':
                continue
            if message.get('role') != 'user' or model != 'deepseek-flash':
                raise ValueError('image Token estimate requires deepseek-flash user images')
            image = block.get('image_url') or {}
            url, prefix = image.get('url', ''), 'data:image/png;base64,'
            if not url.startswith(prefix) or image.get('detail') != 'original':
                raise ValueError('only inline PNG detail=original is supported')
            if len(url) > 32*1024*1024*4//3 + len(prefix)+4:
                raise ValueError('image exceeds inline byte limit')
            try:
                raw = base64.b64decode(url[len(prefix):], validate=True)
                if len(raw)>32*1024*1024 or raw[:8]!=b'\x89PNG\r\n\x1a\n' or raw[12:16]!=b'IHDR':
                    raise ValueError('invalid PNG header')
                width,height=struct.unpack('>II',raw[16:24])
                if not (0<width<=8192 and 0<height<=8192):
                    raise ValueError('PNG dimensions exceed limit')
                import fitz
                decoded=fitz.Pixmap(raw)
                if (decoded.width,decoded.height)!=(width,height):
                    raise ValueError('PNG decode dimensions mismatch')
            except Exception as exc:
                raise ValueError('invalid inline PNG: '+type(exc).__name__) from exc
            image['url']='png-sha256:'+hashlib.sha256(raw).hexdigest()
            images+=1
    return projected,images


def estimate_details(value, policy: dict | None = None, *, output_tokens: int = 0, model: str | None = None) -> dict:
    """统一计入内容、schema、图像、格式与输出预留，单独报告传输字节。"""
    policy = policy or _ACTIVE.get() or {'method':'conservative_character_estimate_v2'}
    model = model or policy.get('model') or 'deepseek-flash'
    serialized = _dump(value)
    transport = len(serialized.encode('utf-8'))
    has_images = isinstance(value,dict) and any(
        isinstance(m.get('content'),list) and any(isinstance(b,dict) and b.get('type')=='image_url' for b in m['content'])
        for m in value.get('messages',[]))
    projected,images = _project_images(value, model) if has_images else (value,0)
    if images and transport > 48*1024*1024:
        raise ValueError('image request exceeds transport byte limit')
    text = _dump(projected)
    legacy = policy.get('version') == 1
    text_tokens = len(text.encode('utf-8')) if legacy else _text_tokens(text, policy)
    # Chat模板与工具转码并非公开精确合同，格式余量独立标注；局部值不重复加整包余量。
    messages = value.get('messages',[]) if isinstance(value,dict) else []
    tools = value.get('tools',[]) if isinstance(value,dict) else []
    overhead = (1024 if legacy else 1024+8*len(messages)+16*len(tools)) if isinstance(value,dict) and 'messages' in value else 0
    if policy.get('legacy_context'):
        overhead = 0
    inputs = text_tokens + overhead + images*1024
    return {'estimated_input_tokens':inputs, 'reserved_tokens':inputs+output_tokens,
            'output_reserve_tokens':output_tokens, 'text_tokens':text_tokens,
            'format_overhead_tokens':overhead, 'image_count':images,
            'image_tokens_upper_bound':images*1024, 'transport_bytes':transport,
            'method':policy.get('method','conservative_character_estimate_v2'),
            'is_exact':False, 'accuracy':policy.get('accuracy','heuristic_not_a_guaranteed_upper_bound'),
            'tokenizer_sha256':policy.get('tokenizer_sha256'),
            'notice':'离线输入/格式估算可能偏差；供应商实际usage是结算真值'}


def estimate(value, policy: dict | None = None) -> int:
    """返回共享输入Token估算，供所有上下文容量子计算使用。"""
    return estimate_details(value, policy)['estimated_input_tokens']


def active_policy() -> dict:
    """读取当前调用冻结口径，不暴露凭据或跨线程可变配置。"""
    return _ACTIVE.get() or {}


def validate_frozen_request(request: dict, policy: dict | None, actual_model: str | None = None,
                            *, allow_model_override: bool = False) -> dict | None:
    """发送前检查新根冻结的单请求容量；不预留额度、不改历史收据。

    Args:
        request: 实际发送的消息、工具及max_tokens；输出按本次真实上限计入。
        policy: 根任务冻结策略；None表示旧任务，不补套当前策略。
        actual_model: 适配器实际模型；默认读取请求或根策略。
        allow_model_override: 显式固定视觉组件可用其独立模型合同，普通角色不允许。

    Returns:
        统一估算与本次输入容量；旧根返回None。

    Raises:
        ValueError: 模型/本地分词器不兼容、输出超过能力或整份请求超容量。
    """
    if not policy:
        return None
    model = actual_model or request.get('model') or policy['model']
    if model != policy['model']:
        if not allow_model_override:
            raise ValueError('TOKEN_POLICY_MODEL_CHANGED: 当前模型与根任务冻结模型不同；未发送新请求')
        if policy.get('tokenizer_path'):
            raise ValueError('TOKENIZER_COMPATIBILITY: 固定组件模型与根任务本地tokenizer不兼容')
    output = request['max_tokens']
    if type(output) is not int or output < 0:
        raise ValueError('INVALID_OUTPUT_RESERVE: 本次输出上限必须为非负整数')
    maximum = min(policy['model_max_output_tokens'], model_capabilities(model)['max_output_tokens'])
    if output > maximum:
        raise ValueError('CONTEXT_OUTPUT_CAPACITY: 本次输出上限超过冻结模型能力；未发送模型')
    value = {'messages':request['messages'], 'tools':request.get('tools',[])}
    for key in ('response_format','tool_choice'):
        if key in request:
            value[key] = request[key]
    details = estimate_details(value, policy, output_tokens=output, model=model)
    capacity = policy['context_window_tokens']-output-policy['context_safety_tokens']
    if details['estimated_input_tokens'] > capacity:
        raise ValueError('CONTEXT_CAPACITY: 本次输入、实际输出与安全预留超过根任务冻结窗口；未发送模型')
    return {**details, 'input_capacity':capacity, 'context_window_tokens':policy['context_window_tokens']}
