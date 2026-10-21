"""独立付费网页请求收据：发送前原子占位，成功复用，未知状态不重发。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from research_loop.providers.paper_acquisition import PaperAcquisitionError


SCHEMA = '''CREATE TABLE IF NOT EXISTS web_request_receipts (
 receipt_id CHAR(64) PRIMARY KEY,
 root_task_id VARCHAR(64) NOT NULL,
 operation VARCHAR(16) NOT NULL,
 request_json TEXT NOT NULL,
 status VARCHAR(16) NOT NULL,
 response_json TEXT,
 diagnostic_json TEXT NOT NULL
)'''


class WebRequestStore:
    """在任务数据库中保存请求级收据，不混入生成Token账本。"""

    def __init__(self, db):
        """仅在启用组件时建立独立表；不改写旧任务或原始收据。"""
        self.db = db
        db.execute(db._sql(SCHEMA))

    def run(self, root_task_id: str, operation: str, payload: dict[str, Any],
            send: Callable[[], dict[str, Any]], active: Callable[[], bool]) -> dict[str, Any]:
        """成功结果先持久化再交付；并发、崩溃及发送后未知均不盲重试。

        Args:
            root_task_id: 后端解析的根任务身份。
            operation: search或fetch。
            payload: 无认证头的供应商参数。
            send: 单次发送回调，不包含自动重试。
            active: 取消及租约检查；失权时拒绝发送。

        Returns:
            原始JSON结果和收据身份；不估算实际计费或填零Token。
        """
        request = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        receipt = hashlib.sha256((root_task_id + '\n' + operation + '\n' + request).encode()).hexdigest()
        existing = self.db.fetchone('SELECT * FROM web_request_receipts WHERE receipt_id=?', (receipt,))
        if existing:
            return self._reuse(existing)
        if not active():
            raise PermissionError('web request cancelled or execution lease lost')
        try:
            self.db.execute('INSERT INTO web_request_receipts(receipt_id,root_task_id,operation,request_json,status,diagnostic_json) VALUES(?,?,?,?,?,?)',
                            (receipt, root_task_id, operation, request, 'started', '{}'))
        except Exception:
            # 唯一键竞争只由赢家发送；其他数据库错误不能被当作成功占位。
            existing = self.db.fetchone('SELECT * FROM web_request_receipts WHERE receipt_id=?', (receipt,))
            if existing:
                return self._reuse(existing)
            raise
        try:
            if not active():
                raise PermissionError('web request cancelled or execution lease lost')
            response = send()
        except Exception as exc:
            diagnostic = {'error_code': getattr(exc, 'code', 'WEB_REQUEST_UNKNOWN'),
                          'error_type': type(exc).__name__,
                          'acquisition': getattr(exc, 'acquisition', {})}
            # HTTP错误是已确认失败；发送前失权是not_sent，传输中断才未知。
            status = ('not_sent' if isinstance(exc, PermissionError) else
                      'failed' if isinstance(exc, PaperAcquisitionError) and exc.code != 'WEB_REQUEST_UNKNOWN' else 'unknown')
            self.db.execute('UPDATE web_request_receipts SET status=?,diagnostic_json=? WHERE receipt_id=?',
                            (status, json.dumps(diagnostic), receipt))
            if isinstance(exc, (PaperAcquisitionError, PermissionError)):
                raise
            raise PaperAcquisitionError('WEB_REQUEST_UNKNOWN', '网页请求发送结果未知；保留收据且不自动重发',
                                        acquisition={'receipt_id': receipt, **diagnostic}) from exc
        self.db.execute('UPDATE web_request_receipts SET status=?,response_json=?,diagnostic_json=? WHERE receipt_id=?',
                        ('completed', json.dumps(response, ensure_ascii=False), '{}', receipt))
        return {'receipt_id': receipt, 'reused': False, 'response': response,
                'billing': 'independent_request; actual charge unknown; not generation tokens'}

    @staticmethod
    def _reuse(row: dict[str, Any]) -> dict[str, Any]:
        """只重放已落库完整结果；started/unknown/失败不得重新发送。"""
        if row['status'] != 'completed':
            raise PaperAcquisitionError('WEB_REQUEST_UNRESOLVED', '已有网页请求未确认完成；禁止自动重复付费发送',
                                        acquisition={'receipt_id': row['receipt_id'], 'status': row['status'],
                                                     'diagnostic': json.loads(row['diagnostic_json'])})
        return {'receipt_id': row['receipt_id'], 'reused': True, 'response': json.loads(row['response_json']),
                'billing': 'reused receipt; not generation tokens'}
