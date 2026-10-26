"""Reflection 六稿试点的持久发送前预算门；不读取配置文件或凭据。

账本在 HTTPX 的物理 transport 边界预留，再由原适配器解析后的实际 usage
结算。放行按冻结的1M模型上下文合同取2^20输入Token上界，加实际max_tokens
输出预留；字节/分词估算只作诊断。该界依赖供应商合同，不声称分词精确。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from research_loop.providers.deepseek import DeepSeekProvider, RequestNotSent
from research_loop.providers.token_counting import estimate_details


class PilotBudgetStopped(RequestNotSent):
    """本批已停止或发送前余量不足，本次未进入底层 HTTP transport。"""


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def _decimal(value, name):
    if not isinstance(value, (str, Decimal)):
        raise ValueError(name + ' must be an explicit decimal string')
    number = Decimal(value)
    if not number.is_finite() or number <= 0:
        raise ValueError(name + ' must be finite and positive')
    return number


class PilotBudgetLedger:
    """SQLite 原子预留与首错持久停止；所有根和子任务共享此同一对象/文件。"""

    @property
    def limits(self):
        return json.loads(_json(self._limits))

    @limits.setter
    def limits(self, value):
        if hasattr(self,'_limits'):
            raise ValueError('PILOT_CONFIGURATION_IS_FROZEN')
        self._limits = json.loads(_json(value))

    @property
    def rates(self):
        return json.loads(_json(self._rates))

    @rates.setter
    def rates(self, value):
        if hasattr(self,'_rates'):
            raise ValueError('PILOT_CONFIGURATION_IS_FROZEN')
        self._rates = json.loads(_json(value))

    @property
    def bounds(self):
        return json.loads(_json(self._bounds))

    @bounds.setter
    def bounds(self, value):
        if hasattr(self,'_bounds'):
            raise ValueError('PILOT_CONFIGURATION_IS_FROZEN')
        self._bounds = json.loads(_json(value))

    def __init__(self, path: Path, *, limits: dict, rates: dict, bounds: dict | None = None):
        """绑定经根任务冻结的额度、费率与保守计量参数，恢复不自动释放在途预留。"""
        path = Path(path).absolute()
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError('PILOT_LEDGER_SYMLINK')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.limits = {'max_requests': _integer(limits['max_requests'], 'max_requests', 1),
                       'max_tokens': _integer(limits['max_tokens'], 'max_tokens', 1),
                       'max_cost_cny': str(_decimal(limits['max_cost_cny'], 'max_cost_cny'))}
        if self.limits['max_requests'] > 48 or self.limits['max_tokens'] > 2_000_000 or Decimal(self.limits['max_cost_cny']) > 16:
            raise ValueError('PILOT_LIMIT_EXCEEDS_APPROVED_SIX_DRAFT_BUDGET')
        self.rates = {key: str(_decimal(rates[key], key)) for key in (
            'input_per_million_cny', 'output_per_million_cny', 'cached_input_per_million_cny')}
        if Decimal(self.rates['input_per_million_cny']) < 2 or Decimal(self.rates['output_per_million_cny']) < 8:
            raise ValueError('PILOT_RESERVATION_RATE_BELOW_FROZEN_PEAK')
        if Decimal(self.rates['cached_input_per_million_cny']) > Decimal(self.rates['input_per_million_cny']):
            raise ValueError('cached input rate exceeds uncached rate')
        raw_bounds = bounds or {}
        self.bounds = {'allowed_models': raw_bounds.get('allowed_models', ['deepseek-flash']),
                       'allowed_max_output_tokens': raw_bounds.get('allowed_max_output_tokens', [65536, 8192]),
                       'contract_input_upper_tokens': raw_bounds.get('contract_input_upper_tokens', 1_048_576),
                       'wire_byte_multiplier': raw_bounds.get('wire_byte_multiplier', 2),
                       'provider_format_reserve_tokens': raw_bounds.get('provider_format_reserve_tokens', 8192),
                       'per_message_reserve_tokens': raw_bounds.get('per_message_reserve_tokens', 32),
                       'per_tool_reserve_tokens': raw_bounds.get('per_tool_reserve_tokens', 128)}
        if not self.bounds['allowed_models'] or not all(isinstance(m, str) and m for m in self.bounds['allowed_models']):
            raise ValueError('explicit allowed models required')
        if self.bounds['allowed_models'] != ['deepseek-flash'] or self.bounds['contract_input_upper_tokens'] != 1_048_576:
            raise ValueError('PILOT_MODEL_CONTEXT_CONTRACT_NOT_FROZEN')
        if not self.bounds['allowed_max_output_tokens']:
            raise ValueError('explicit original output limits required')
        for number in self.bounds['allowed_max_output_tokens']:
            _integer(number, 'allowed_max_output_tokens', 1)
        for name in ('wire_byte_multiplier', 'provider_format_reserve_tokens', 'per_message_reserve_tokens', 'per_tool_reserve_tokens'):
            _integer(self.bounds[name], name, 1)
        config = _json({'limits': self.limits, 'rates': self.rates, 'bounds': self.bounds})
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS pilot_meta (id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL, stop_reason TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS pilot_calls (call_id TEXT PRIMARY KEY, state TEXT NOT NULL, request_sha256 TEXT NOT NULL, '
                       'input_reserved INTEGER NOT NULL, output_reserved INTEGER NOT NULL, cost_reserved TEXT NOT NULL, details_json TEXT NOT NULL, '
                       'created_at REAL NOT NULL, dispatched_at REAL, settled_at REAL, http_status INTEGER, request_id TEXT, '
                       'usage_json TEXT, actual_tokens INTEGER, actual_cost_upper TEXT, cache_adjusted_cost_estimate TEXT, failure_code TEXT)')
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT config FROM pilot_meta WHERE id=1').fetchone()
            if row is None:
                db.execute('INSERT INTO pilot_meta(id,config) VALUES(1,?)', (config,))
            elif row['config'] != config:
                raise ValueError('PILOT_FROZEN_CONFIGURATION_CHANGED')
            if db.execute("SELECT 1 FROM pilot_calls WHERE state IN ('reserved','dispatched') LIMIT 1").fetchone():
                self._stop(db, 'recovered_unsettled_call')
            db.commit()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=30000')
        db.execute('PRAGMA synchronous=FULL')
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _stop(db, reason):
        db.execute('UPDATE pilot_meta SET stop_reason=COALESCE(stop_reason,?) WHERE id=1', (reason,))

    def _cost(self, prompt, completion, cached=0):
        return ((Decimal(prompt-cached)*Decimal(self.rates['input_per_million_cny']) +
                 Decimal(cached)*Decimal(self.rates['cached_input_per_million_cny']) +
                 Decimal(completion)*Decimal(self.rates['output_per_million_cny'])) / Decimal(1_000_000))

    def halt(self, reason: str):
        """首个停止原因保持不变，不提供自动清除或继续执行接口。"""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._stop(db, reason)
            db.commit()

    @property
    def global_stop(self):
        with self._connect() as db:
            return db.execute('SELECT stop_reason FROM pilot_meta WHERE id=1').fetchone()['stop_reason']

    def reserve(self, request_sha256: str, input_upper: int, max_output: int, details: dict | None = None) -> str:
        """原子占用请求、Token与金额余量；尚未发送的额度也计入在途总额。"""
        _integer(input_upper, 'input_upper')
        _integer(max_output, 'max_output', 1)
        cost = self._cost(input_upper, max_output)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            reason = db.execute('SELECT stop_reason FROM pilot_meta WHERE id=1').fetchone()['stop_reason']
            if reason:
                raise PilotBudgetStopped(reason)
            if db.execute("SELECT 1 FROM pilot_calls WHERE state IN ('reserved','dispatched') LIMIT 1").fetchone():
                self._stop(db, 'physical_request_in_flight')
                db.commit()
                raise PilotBudgetStopped('physical_request_in_flight')
            rows = db.execute("SELECT * FROM pilot_calls WHERE state!='not_sent'").fetchall()
            tokens = sum(r['actual_tokens'] if r['state']=='succeeded' else r['input_reserved']+r['output_reserved'] for r in rows)
            money = sum((Decimal(r['actual_cost_upper']) if r['state']=='succeeded' else Decimal(r['cost_reserved']) for r in rows), Decimal(0))
            reason = ('request_limit' if len(rows) >= self.limits['max_requests'] else
                      'token_reservation_limit' if tokens+input_upper+max_output > self.limits['max_tokens'] else
                      'money_reservation_limit' if money+cost > Decimal(self.limits['max_cost_cny']) else None)
            if reason:
                self._stop(db, reason)
                db.commit()
                raise PilotBudgetStopped(reason)
            identity = uuid4().hex
            db.execute('INSERT INTO pilot_calls(call_id,state,request_sha256,input_reserved,output_reserved,cost_reserved,details_json,created_at) '
                       "VALUES(?,'reserved',?,?,?,?,?,?)", (identity,request_sha256,input_upper,max_output,str(cost),_json(details or {}),time.time()))
            db.commit()
            return identity

    def dispatched(self, identity):
        """进入底层 transport 前持久标记可能已发送，崩溃恢复不猜测远端状态。"""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            reason = db.execute('SELECT stop_reason FROM pilot_meta WHERE id=1').fetchone()['stop_reason']
            if reason:
                raise PilotBudgetStopped(reason)
            changed = db.execute("UPDATE pilot_calls SET state='dispatched',dispatched_at=? WHERE call_id=? AND state='reserved'", (time.time(),identity)).rowcount
            if changed != 1:
                raise ValueError('PILOT_DISPATCH_IDENTITY')
            db.commit()

    def not_sent(self, identity):
        """仅明确RequestNotSent使用：释放本次预留，保留尝试记录。"""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM pilot_calls WHERE call_id=?', (identity,)).fetchone()
            if not row or row['state'] not in {'reserved','dispatched','not_sent'}:
                raise ValueError('PILOT_NOT_SENT_IDENTITY')
            db.execute("UPDATE pilot_calls SET state='not_sent',settled_at=?,failure_code='RequestNotSent' WHERE call_id=?", (time.time(),identity))
            db.commit()

    def fail(self, identity, *, reason, state='unknown', http_status=None, request_id=None, usage=None):
        """未知/拒绝保留全部预留，停止全批；不把缺失usage写成零。"""
        if state not in {'unknown','rejected'}:
            raise ValueError('invalid failure state')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM pilot_calls WHERE call_id=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('PILOT_FAILURE_IDENTITY')
            if row['state'] in {'unknown','rejected'}:
                return
            if row['state'] not in {'reserved','dispatched'}:
                raise ValueError('PILOT_FAILURE_AFTER_SETTLEMENT')
            db.execute('UPDATE pilot_calls SET state=?,settled_at=?,failure_code=?,http_status=?,request_id=?,usage_json=? WHERE call_id=?',
                       (state,time.time(),reason,http_status,request_id,_json(usage) if usage is not None else None,identity))
            self._stop(db, reason)
            db.commit()

    def settle(self, identity: str, usage: dict | None, *, request_id=None):
        """幂等结算实际usage；越过预留仍保存真值并持久停，不截断或填零。"""
        if usage is None or not isinstance(usage, dict):
            self.fail(identity, reason='usage_missing')
            return
        try:
            prompt = _integer(usage['prompt_tokens'], 'prompt_tokens')
            completion = _integer(usage['completion_tokens'], 'completion_tokens')
            total = _integer(usage['total_tokens'], 'total_tokens')
            if total != prompt+completion:
                raise ValueError('usage total mismatch')
            cached = usage.get('prompt_cache_hit_tokens')
            if cached is None:
                cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            if cached is not None:
                _integer(cached, 'cached_tokens')
                if cached > prompt:
                    raise ValueError('cache exceeds prompt')
        except (KeyError, TypeError, ValueError):
            self.fail(identity, reason='usage_invalid',usage=usage)
            raise
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM pilot_calls WHERE call_id=?', (identity,)).fetchone()
            if row is None:
                raise ValueError('PILOT_SETTLEMENT_IDENTITY')
            serialized = _json(usage)
            if row['state']=='succeeded':
                if row['usage_json'] != serialized or row['request_id'] != request_id:
                    raise ValueError('PILOT_SETTLEMENT_CHANGED')
                return
            if row['state']!='dispatched':
                raise ValueError('PILOT_SETTLEMENT_REQUIRES_DISPATCHED')
            upper = self._cost(prompt,completion)
            adjusted = str(self._cost(prompt,completion,cached)) if cached is not None else None
            db.execute("UPDATE pilot_calls SET state='succeeded',settled_at=?,request_id=?,usage_json=?,actual_tokens=?,actual_cost_upper=?,cache_adjusted_cost_estimate=? WHERE call_id=?",
                       (time.time(),request_id,serialized,total,str(upper),adjusted,identity))
            if prompt > row['input_reserved'] or completion > row['output_reserved']:
                self._stop(db, 'reservation_upper_bound_breached')
            db.commit()

    def snapshot(self):
        """返回无原文、无凭据的全批账本，可与业务物理收据逐条对照。"""
        with self._connect() as db:
            rows = [dict(r) for r in db.execute('SELECT * FROM pilot_calls ORDER BY created_at,call_id')]
        known = [r for r in rows if r['actual_tokens'] is not None]
        return {'schema_version':1, 'global_stop':self.global_stop,
            'limits':self.limits, 'rates':self.rates, 'bounds':self.bounds,
            'attempt_rows':len(rows), 'physical_requests':sum(r['dispatched_at'] is not None and r['state']!='not_sent' for r in rows),
            'known_total_tokens':sum(r['actual_tokens'] for r in known),
            'known_cost_upper_cny':str(sum((Decimal(r['actual_cost_upper']) for r in known),Decimal(0))),
            'unsettled_or_usage_missing':sum(r['state'] in {'reserved','dispatched','unknown','rejected'} for r in rows),
            'calls':rows, 'input_estimation':'frozen_official_1M_context_contract_rounded_up_to_2_power_20_plus_actual_max_output',
            'bound_dependency':'supplier_model_context_and_output_contract; not_exact_tokenizer_count',
            'actual_money':'rate_based_estimate_not_account_bill'}


_ACTIVE_CALL = ContextVar('reflection_pilot_physical_call', default=None)


def request_reservation(payload: dict, wire_bytes: int, ledger: PilotBudgetLedger):
    """基于实际完整JSON预留；不改max_tokens、thinking、tools或消息正文。"""
    model = payload.get('model')
    output = payload.get('max_tokens')
    if model not in ledger.bounds['allowed_models']:
        raise RequestNotSent('PILOT_MODEL_NOT_FROZEN')
    if type(output) is not int or output not in ledger.bounds['allowed_max_output_tokens']:
        raise RequestNotSent('PILOT_ORIGINAL_MAX_OUTPUT_REQUIRED')
    messages, tools = payload.get('messages'), payload.get('tools') or []
    if not isinstance(messages,list) or not isinstance(tools,list):
        raise RequestNotSent('PILOT_PAYLOAD_SHAPE')
    has_images = any(isinstance(m.get('content'),list) and any(isinstance(b,dict) and b.get('type')=='image_url' for b in m['content']) for m in messages)
    if has_images and (output != 8192 or payload.get('thinking') != {'type':'disabled'}):
        raise RequestNotSent('PILOT_VISION_OUTPUT_OR_THINKING_CHANGED')
    if not has_images and output != 65536:
        raise RequestNotSent('PILOT_ORDINARY_OUTPUT_CHANGED')
    details = estimate_details(payload, model=model)
    formatting = (ledger.bounds['provider_format_reserve_tokens'] + len(messages)*ledger.bounds['per_message_reserve_tokens'] +
                  len(tools)*ledger.bounds['per_tool_reserve_tokens'])
    diagnostic = max(details['estimated_input_tokens'], wire_bytes*ledger.bounds['wire_byte_multiplier'])+formatting
    upper = ledger.bounds['contract_input_upper_tokens']
    # 图像base64的传输字节不等于视觉Token，放行仍使用同模型总上下文合同。
    if not has_images and diagnostic > upper:
        raise RequestNotSent('PILOT_DIAGNOSTIC_INPUT_EXCEEDS_FROZEN_CONTEXT_BOUND')
    return upper, output, {'model':model, 'max_output_tokens':output, 'wire_bytes':wire_bytes,
        'message_count':len(messages), 'tool_count':len(tools), 'stream':bool(payload.get('stream')),
        'image_count':details['image_count'],
        'shared_estimated_input_tokens':details['estimated_input_tokens'], 'format_reserve_tokens':formatting,
        'diagnostic_input_tokens':diagnostic,
        'input_upper_assumption':'frozen_1M_model_context_contract_rounded_up_to_1048576',
        'provider_contract_input_upper_tokens':upper,
        'exact_tokenizer_count':False}


class _BudgetTransport(httpx.BaseTransport):
    """每次真正进入HTTP transport都先记账；下层transport明确禁止内部重试。"""

    def __init__(self, inner, ledger, endpoint):
        self.inner, self.ledger, self.endpoint = inner,ledger,endpoint

    def handle_request(self, request):
        scope = _ACTIVE_CALL.get()
        if not scope or scope['ledger'] is not self.ledger:
            raise RequestNotSent('PILOT_HTTP_OUTSIDE_BUDGETED_PROVIDER_CALL')
        if scope['calls']:
            self.ledger.halt('unexpected_multiple_http_requests_per_provider_call')
            raise RequestNotSent('PILOT_HTTP_RETRY_OR_REDIRECT_DENIED')
        if request.method!='POST' or str(request.url)!=self.endpoint:
            raise RequestNotSent('PILOT_ENDPOINT_NOT_FROZEN')
        raw = request.read()
        try:
            payload = json.loads(raw)
            upper, output, details = request_reservation(payload,len(raw),self.ledger)
        except RequestNotSent:
            raise
        except (ValueError,TypeError,AttributeError) as exc:
            raise RequestNotSent('PILOT_INVALID_REQUEST_PAYLOAD') from exc
        identity = self.ledger.reserve(hashlib.sha256(raw).hexdigest(),upper,output,details)
        scope['calls'].append(identity)
        self.ledger.dispatched(identity)
        try:
            response = self.inner.handle_request(request)
        except RequestNotSent:
            self.ledger.not_sent(identity)
            raise
        except BaseException as exc:
            self.ledger.fail(identity,reason='transport_unknown:'+type(exc).__name__)
            raise
        if response.status_code>=400:
            self.ledger.fail(identity,reason='provider_http_'+str(response.status_code),state='rejected',
                http_status=response.status_code,request_id=response.headers.get('x-request-id') or response.headers.get('request-id'))
        return response

    def close(self):
        self.inner.close()


class BudgetedProvider:
    """保留原适配器的参数、解析和流式行为，只加持久物理发送账本。"""

    def __init__(self, provider: DeepSeekProvider, ledger: PilotBudgetLedger):
        if type(provider) is not DeepSeekProvider:
            raise ValueError('PILOT_REQUIRES_FROZEN_DEEPSEEK_PROVIDER')
        parsed = urlsplit(provider.endpoint)
        if provider.endpoint != 'https://api.deepseek.com/chat/completions' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('PILOT_UNSAFE_ENDPOINT')
        inner = provider.client._transport
        if type(inner) is httpx.HTTPTransport:
            pool = getattr(inner,'_pool',None)
            if type(pool).__name__!='ConnectionPool' or getattr(pool,'_retries',None)!=0:
                raise ValueError('PILOT_TRANSPORT_RETRIES_OR_PROXY_NOT_QUALIFIED')
        elif type(inner) is not httpx.MockTransport:
            raise ValueError('PILOT_TRANSPORT_NOT_QUALIFIED')
        self._provider, self.ledger, self._old_client = provider,ledger,provider.client
        self._client = httpx.Client(transport=_BudgetTransport(inner,ledger,provider.endpoint),
            timeout=provider.client.timeout, follow_redirects=False, trust_env=False)
        provider.client = self._client
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._provider,name)

    def _invoke(self, method, *args, **kwargs):
        if self.ledger.global_stop:
            raise PilotBudgetStopped(self.ledger.global_stop)
        scope = {'ledger':self.ledger,'calls':[]}
        token = _ACTIVE_CALL.set(scope)
        try:
            raw = method(*args,**kwargs)
            if len(scope['calls'])!=1:
                raise RuntimeError('PILOT_PROVIDER_RETURNED_WITHOUT_ONE_PHYSICAL_REQUEST')
            self.ledger.settle(scope['calls'][0],raw.get('usage'),request_id=raw.get('id'))
            return raw
        except BaseException as exc:
            for identity in scope['calls']:
                rows = self.ledger.snapshot()['calls']
                state = next(r['state'] for r in rows if r['call_id']==identity)
                if state in {'reserved','dispatched'}:
                    if isinstance(exc,RequestNotSent):
                        self.ledger.not_sent(identity)
                    else:
                        self.ledger.fail(identity,reason='provider_unknown:'+type(exc).__name__)
            raise
        finally:
            _ACTIVE_CALL.reset(token)

    def send(self,*args,**kwargs):
        return self._invoke(self._provider.send,*args,**kwargs)

    def send_stream(self,*args,**kwargs):
        return self._invoke(self._provider.send_stream,*args,**kwargs)

    def complete(self,messages,tools=None,force_text=False,**kwargs):
        return self.parse_response(self.send(messages,tools,force_text,**kwargs))

    def close(self):
        if not self._closed:
            self._client.close()
            self._old_client.close()
            self._closed = True


def wrap_provider(provider: DeepSeekProvider, ledger: PilotBudgetLedger) -> BudgetedProvider:
    """供TaskExecutor/Reviewer统一工厂调用；同批必须传同一个持久ledger。"""
    return BudgetedProvider(provider,ledger)
