"""H1 实验专用持久预算门；不发送网络，也不改变业务预算实现。"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4


class BudgetHalt(RuntimeError):
    """预算、时间或证据不完整，当前批次不允许发送新请求。"""


class Ledger:
    """按整批和根任务族原子预留；任何已发送未知结果都保持占用。"""

    def __init__(self, path, *, allowed_cells=None, clock=None, start_time=None,
                 batch_tokens=12_000_000, batch_requests=192, batch_yuan=96,
                 batch_seconds=5400, root_tokens=4_000_000, root_requests=32,
                 root_seconds=900, max_output_tokens=65_536, carried_usage=None):
        """创建专用账本；已有不完整发送记录的账本重新打开时保持冻结。"""
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or time.time
        allowed_cells = allowed_cells or [(f'H1-Q{i}', variant) for i in range(1, 5) for variant in ('A', 'B')]
        try:
            amount = Decimal(str(batch_yuan))
        except InvalidOperation as exc:
            raise ValueError('money budget must be a finite positive decimal') from exc
        if not amount.is_finite() or amount <= 0:
            raise ValueError('money budget must be a finite positive decimal')
        nano_yuan = amount * Decimal(1_000_000_000)
        if nano_yuan != nano_yuan.to_integral_value():
            raise ValueError('money budget must be exactly representable in nano yuan')
        self.config = {'allowed_cells': sorted([list(cell) for cell in allowed_cells]),
            'batch_tokens': batch_tokens, 'batch_requests': batch_requests,
            'batch_nano_yuan': int(nano_yuan), 'batch_seconds': batch_seconds,
            'root_tokens': root_tokens, 'root_requests': root_requests,
            'root_seconds': root_seconds, 'max_output_tokens': max_output_tokens,
            'input_nano_per_token': 2000, 'cached_input_nano_per_token': 40,
            'output_nano_per_token': 8000}
        if any(self.config[key] <= 0 for key in ('batch_tokens', 'batch_requests', 'batch_nano_yuan',
                'batch_seconds', 'root_tokens', 'root_requests', 'root_seconds', 'max_output_tokens')):
            raise ValueError('budget limits must be positive')
        self.carried_usage = self._validate_carried_usage(carried_usage)
        for usage_key, limit_key in (('tokens', 'batch_tokens'), ('requests', 'batch_requests'),
                ('nano_yuan', 'batch_nano_yuan'), ('seconds', 'batch_seconds')):
            if self.carried_usage[usage_key] > self.config[limit_key]:
                raise ValueError('carried usage exceeds original batch allowance: ' + usage_key)
        if any(self.carried_usage.values()):
            # 零扣减保持旧配置兼容；非零扣减属于新账本不可变合同，遗漏时不能重开。
            self.config['carried_usage'] = self.carried_usage.copy()
        with self._transaction() as db:
            schema = '''CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS roots (root_id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                    variant TEXT NOT NULL, started_at REAL NOT NULL, UNIQUE(case_id, variant));
                CREATE TABLE IF NOT EXISTS reservations (reservation_id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL,
                    sent_at REAL, finished_at REAL, reserved_input INTEGER NOT NULL,
                    reserved_output INTEGER NOT NULL, reserved_tokens INTEGER NOT NULL,
                    reserved_nano INTEGER NOT NULL, actual_input INTEGER, actual_output INTEGER,
                    actual_tokens INTEGER, actual_nano INTEGER, usage_json TEXT,
                    request_json TEXT NOT NULL, response_json TEXT, reason TEXT);'''
            for statement in schema.split(';'):
                if statement.strip():
                    db.execute(statement)
            previous = self._meta(db, 'config')
            if previous is None:
                now = self.clock()
                self._put(db, 'config', self.config)
                self._put(db, 'started_at', now if start_time is None else start_time)
                self._put(db, 'last_clock', now)
                self._put(db, 'halt_reason', None)
            elif previous != self.config:
                raise ValueError('cannot change limits or cells of an existing ledger')
            elif db.execute("SELECT 1 FROM reservations WHERE status IN ('reserved','sent') LIMIT 1").fetchone():
                self._put(db, 'halt_reason', 'ledger reopened with unresolved reservation; no resend')

    @staticmethod
    def _validate_carried_usage(carried_usage):
        """历史只以冻结的整Token、请求、纳元和向上取整秒扣减，不重开旧账本。"""
        keys = {'tokens', 'requests', 'nano_yuan', 'seconds'}
        if carried_usage is None:
            return dict.fromkeys(sorted(keys), 0)
        if not isinstance(carried_usage, dict) or set(carried_usage) != keys:
            raise ValueError('carried usage must include tokens, requests, nano_yuan and seconds')
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in carried_usage.values()):
            raise ValueError('carried usage must contain non-negative integers')
        return {key: carried_usage[key] for key in sorted(keys)}

    @contextmanager
    def _transaction(self):
        """每次使用独立连接，线程与进程通过写锁串行化检查及记账。"""
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _meta(db, key):
        row = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row['value']) if row else None

    @staticmethod
    def _put(db, key, value):
        db.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)',
                   (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))

    def _time_check(self, db, root=None):
        """时间倒退直接拒绝，不用墙钟回拨扩大授权窗口。"""
        now = self.clock()
        if now < self._meta(db, 'last_clock'):
            raise BudgetHalt('clock moved backwards')
        self._put(db, 'last_clock', now)
        if (self.carried_usage['seconds'] + now - self._meta(db, 'started_at')
                >= self.config['batch_seconds']):
            raise BudgetHalt('batch time exhausted')
        if root is not None and now - root['started_at'] >= self.config['root_seconds']:
            raise BudgetHalt('root time exhausted')
        return now

    def _check_open(self, db):
        reason = self._meta(db, 'halt_reason')
        if reason:
            raise BudgetHalt('batch halted: ' + reason)

    @staticmethod
    def _reservation(db, reservation_id):
        row = db.execute('SELECT * FROM reservations WHERE reservation_id=?', (reservation_id,)).fetchone()
        if row is None:
            raise ValueError('unknown reservation')
        return row

    @staticmethod
    def _totals(db, root_id=None):
        where = "status != 'not_sent'"
        args = ()
        if root_id is not None:
            where += ' AND root_id=?'
            args = (root_id,)
        row = db.execute("SELECT COUNT(*) requests, COALESCE(SUM(CASE WHEN status='settled' "
            "THEN actual_tokens ELSE reserved_tokens END),0) tokens, "
            "COALESCE(SUM(CASE WHEN status='settled' THEN actual_nano ELSE reserved_nano END),0) nano "
            'FROM reservations WHERE ' + where, args).fetchone()
        return dict(row)

    def _batch_totals(self, db):
        """历史消耗仅叠加到批授权，不能侵占新根任务各自的额度。"""
        totals = self._totals(db)
        return {key: totals[key] + self.carried_usage[carry_key]
            for key, carry_key in (('requests', 'requests'), ('tokens', 'tokens'),
                                  ('nano', 'nano_yuan'))}

    def begin_root(self, root_id, case_id, variant):
        """一个已批准格只允许创建一个根任务身份，不授权重新运行。"""
        with self._transaction() as db:
            self._check_open(db)
            now = self._time_check(db)
            if [case_id, variant] not in self.config['allowed_cells']:
                raise ValueError('case and variant not in frozen matrix')
            try:
                db.execute('INSERT INTO roots VALUES (?,?,?,?)', (root_id, case_id, variant, now))
            except sqlite3.IntegrityError as exc:
                raise BudgetHalt('root or case/variant already started; no repeat') from exc
        return {'root_id': root_id, 'case_id': case_id, 'variant': variant, 'started_at': now}

    def reserve(self, root_id, max_output_tokens=65_536, input_upper_bound=1_048_576,
                request_metadata=None):
        """在发送前按供应商输入容量上界与本次最大输出预留。"""
        for value in (max_output_tokens, input_upper_bound):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('reservation bounds must be positive integers')
        if max_output_tokens > self.config['max_output_tokens']:
            raise BudgetHalt('output allowance exceeds frozen request maximum')
        tokens = input_upper_bound + max_output_tokens
        nano = input_upper_bound * self.config['input_nano_per_token'] + max_output_tokens * self.config['output_nano_per_token']
        with self._transaction() as db:
            self._check_open(db)
            root = db.execute('SELECT * FROM roots WHERE root_id=?', (root_id,)).fetchone()
            if root is None:
                raise ValueError('root has not been registered')
            now = self._time_check(db, root)
            total, family = self._batch_totals(db), self._totals(db, root_id)
            if total['requests'] + 1 > self.config['batch_requests'] or family['requests'] + 1 > self.config['root_requests']:
                raise BudgetHalt('request budget exhausted')
            if total['tokens'] + tokens > self.config['batch_tokens'] or family['tokens'] + tokens > self.config['root_tokens']:
                raise BudgetHalt('token reservation budget exhausted')
            if total['nano'] + nano > self.config['batch_nano_yuan']:
                raise BudgetHalt('money reservation budget exhausted')
            identity = uuid4().hex
            db.execute('''INSERT INTO reservations(reservation_id,root_id,status,created_at,
                reserved_input,reserved_output,reserved_tokens,reserved_nano,request_json)
                VALUES (?,?,'reserved',?,?,?,?,?,?)''',
                (identity, root_id, now, input_upper_bound, max_output_tokens, tokens, nano,
                 json.dumps(request_metadata or {}, ensure_ascii=False, sort_keys=True)))
        return {'reservation_id': identity, 'root_id': root_id, 'reserved_tokens': tokens,
                'reserved_nano_yuan': nano, 'reserved_input': input_upper_bound, 'reserved_output': max_output_tokens}

    def mark_sent(self, reservation_id):
        """紧邻底层发送前调用；后续所有不完整结果必须保留未知占用。"""
        with self._transaction() as db:
            self._check_open(db)
            row = self._reservation(db, reservation_id)
            if row['status'] != 'reserved':
                raise BudgetHalt('reservation is not unsent; no resend')
            root = db.execute('SELECT * FROM roots WHERE root_id=?', (row['root_id'],)).fetchone()
            now = self._time_check(db, root)
            db.execute("UPDATE reservations SET status='sent',sent_at=? WHERE reservation_id=?", (now, reservation_id))

    @staticmethod
    def _usage(usage):
        """校验完整用量；缓存和推理明细只校验，不与总输入输出重复相加。"""
        if not isinstance(usage, dict):
            raise ValueError('missing usage object')
        def integer(key):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('missing or invalid usage.' + key)
            return value
        prompt, completion, total = [integer(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
        if total != prompt + completion:
            raise ValueError('total_tokens differs from input plus output')
        details = usage.get('prompt_tokens_details') or {}
        cached = usage.get('prompt_cache_hit_tokens', details.get('cached_tokens', 0))
        if isinstance(cached, bool) or not isinstance(cached, int) or not 0 <= cached <= prompt:
            raise ValueError('invalid cache hit tokens')
        if 'cached_tokens' in details and details['cached_tokens'] != cached:
            raise ValueError('cache details disagree')
        if 'prompt_cache_miss_tokens' in usage and usage['prompt_cache_miss_tokens'] != prompt - cached:
            raise ValueError('cache miss tokens disagree')
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
        if isinstance(reasoning, bool) or not isinstance(reasoning, int) or not 0 <= reasoning <= completion:
            raise ValueError('invalid reasoning token subset')
        return prompt, completion, total, cached

    def settle(self, reservation_id, usage, response_metadata=None):
        """只在完整响应与完整用量可读时结算；失效用量冻结整批。"""
        error = None
        with self._transaction() as db:
            row = self._reservation(db, reservation_id)
            if row['status'] == 'settled':
                if json.loads(row['usage_json']) != usage:
                    raise BudgetHalt('settled receipt is immutable')
                return dict(row)
            if row['status'] != 'sent':
                raise BudgetHalt('only sent requests can settle')
            try:
                prompt, completion, total, cached = self._usage(usage)
                if prompt > row['reserved_input'] or completion > row['reserved_output']:
                    raise ValueError('actual usage exceeds reserved upper bound')
                nano = ((prompt - cached) * self.config['input_nano_per_token'] +
                        cached * self.config['cached_input_nano_per_token'] + completion * self.config['output_nano_per_token'])
            except (TypeError, AttributeError, ValueError) as exc:
                error = str(exc)
                self._unknown(db, row, error, observed_usage=usage)
            else:
                db.execute("""UPDATE reservations SET status='settled',finished_at=?,actual_input=?,
                    actual_output=?,actual_tokens=?,actual_nano=?,usage_json=?,response_json=?
                    WHERE reservation_id=?""", (self.clock(), prompt, completion, total, nano,
                    json.dumps(usage, ensure_ascii=False, sort_keys=True),
                    json.dumps(response_metadata or {}, ensure_ascii=False, sort_keys=True), reservation_id))
        if error:
            raise BudgetHalt('usage cannot be settled: ' + error)
        return {'reservation_id': reservation_id, 'actual_tokens': total,
                'actual_nano_yuan': nano, 'actual_input': prompt, 'actual_output': completion}

    def _unknown(self, db, row, reason, observed_usage=None):
        """未知结果维持完整预留；记录可见 usage，禁止自动恢复发送。"""
        db.execute("UPDATE reservations SET status='unknown',finished_at=?,reason=?,usage_json=? WHERE reservation_id=?",
            (self.clock(), str(reason), json.dumps(observed_usage, ensure_ascii=False, sort_keys=True), row['reservation_id']))
        self._put(db, 'halt_reason', str(reason))

    def unknown(self, reservation_id, reason):
        """用于断流、超时、错误响应或缺用量；这些都不是确定未发送。"""
        with self._transaction() as db:
            row = self._reservation(db, reservation_id)
            if row['status'] == 'unknown':
                return
            if row['status'] != 'sent':
                raise BudgetHalt('only sent requests may become unknown')
            self._unknown(db, row, reason)

    def not_sent(self, reservation_id, reason):
        """仅释放尚未 mark_sent 的确定未发送预留；保留原记录。"""
        with self._transaction() as db:
            row = self._reservation(db, reservation_id)
            if row['status'] == 'not_sent':
                return
            if row['status'] != 'reserved':
                raise BudgetHalt('cannot release a sent or settled request')
            db.execute("UPDATE reservations SET status='not_sent',finished_at=?,reason=? WHERE reservation_id=?",
                       (self.clock(), str(reason), reservation_id))

    def snapshot(self):
        """返回用量和仍占用的额度；estimated_cost 不是供应商账单。"""
        with self._transaction() as db:
            reservations = [dict(row) for row in db.execute('SELECT * FROM reservations ORDER BY created_at,reservation_id')]
            totals = self._batch_totals(db)
            known = [r for r in reservations if r['status'] == 'settled']
            return {'limits': self.config, 'started_at': self._meta(db, 'started_at'),
                'halt_reason': self._meta(db, 'halt_reason'), 'charged_or_reserved': totals,
                'carried_usage': self.carried_usage.copy(),
                'current_charged_or_reserved': self._totals(db),
                'batch_absolute_deadline': (self._meta(db, 'started_at')
                    + self.config['batch_seconds'] - self.carried_usage['seconds']),
                'actual_known_tokens': sum(r['actual_tokens'] for r in known),
                'actual_known_input': sum(r['actual_input'] for r in known),
                'actual_known_output': sum(r['actual_output'] for r in known),
                'actual_known_estimated_nano_yuan': sum(r['actual_nano'] for r in known),
                'physical_sent_requests': sum(r['status'] in ('sent', 'settled', 'unknown') for r in reservations),
                'roots': [dict(row) for row in db.execute('SELECT * FROM roots ORDER BY started_at,root_id')],
                'reservations': reservations}
