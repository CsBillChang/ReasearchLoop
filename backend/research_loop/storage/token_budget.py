"""根任务累计Token账本：先预留、后结算；收据是真值，恢复不重复扣款或发送。"""
from __future__ import annotations

from contextlib import contextmanager
import base64
from copy import deepcopy
import hashlib
import json
import struct
from research_loop.storage.memory import Transaction, digest, dump
from research_loop.storage.mysql import _now

SCHEMA = """
CREATE TABLE IF NOT EXISTS task_token_budgets (
 root_task_id VARCHAR(64) PRIMARY KEY, token_limit INTEGER NOT NULL,
 tail_tokens INTEGER NOT NULL, used_tokens INTEGER NOT NULL DEFAULT 0,
 reserved_tokens INTEGER NOT NULL DEFAULT 0, created_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS token_reservations (
 reservation_id VARCHAR(160) PRIMARY KEY, root_task_id VARCHAR(64) NOT NULL,
 task_id VARCHAR(64) NOT NULL, sequence_no INTEGER NOT NULL,
 request_hash CHAR(64) NOT NULL, reserved_tokens INTEGER NOT NULL,
 actual_tokens INTEGER, status VARCHAR(32) NOT NULL, created_at VARCHAR(64) NOT NULL,
 UNIQUE(task_id,sequence_no)
);
"""


def create_policy(tx: Transaction, root: str, settings) -> None:
    """在新根任务事务中冻结配置；恢复和子任务不得修改旧预算。"""
    tx.execute('INSERT INTO task_token_budgets(root_task_id,token_limit,tail_tokens,created_at) VALUES(?,?,?,?)',
               (root, settings.task_token_limit, settings.task_token_tail, _now()))


def estimate_request(request: dict) -> int:
    """返回保守输入/输出预留；旧文字结果逐值保持，图片按独立规则计量。"""
    return estimate_details(request)['reserved_tokens']


def estimate_details(request: dict) -> dict:
    """识别可信PNG图片块，隔离传输字节与Token计量。

    Args:
        request: 冻结请求；只有官方deepseek-flash的user内联PNG启用图片分支。

    Returns:
        预留值、方法、图片数与传输字节；不是供应商实际usage。

    Raises:
        ValueError: 图片损坏、来源/模型不支持或超出官方传输限制。
    """
    original = {'messages': request['messages'], 'tools': request.get('tools', [])}
    has_images = any(isinstance(m.get('content'), list) and any(
        isinstance(b, dict) and b.get('type') == 'image_url' for b in m['content']) for m in original['messages'])
    if not has_images:
        # 日常文字路线保持原公式和单次序列化，不因实验组件复制大型上下文。
        size = len(dump(original).encode('utf-8'))
        return {'reserved_tokens': size + 1024 + int(request['max_tokens']),
                'method': 'utf8_bytes_plus_1024_plus_output', 'image_count': 0,
                'image_tokens_upper_bound': 0, 'transport_bytes': size}
    projected = deepcopy(original)
    images = 0
    for message in projected['messages']:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'image_url':
                continue
            if message.get('role') != 'user' or request.get('model') != 'deepseek-flash':
                raise ValueError('image Token estimate requires deepseek-flash user images')
            image = block.get('image_url') or {}
            url = image.get('url', '')
            prefix = 'data:image/png;base64,'
            if not url.startswith(prefix) or image.get('detail') != 'original':
                raise ValueError('only inline PNG detail=original is supported')
            if len(url) > 32 * 1024 * 1024 * 4 // 3 + len(prefix) + 4:
                raise ValueError('image exceeds inline byte limit')
            try:
                data = base64.b64decode(url[len(prefix):], validate=True)
                if len(data) > 32 * 1024 * 1024 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
                    raise ValueError('invalid PNG header')
                width, height = struct.unpack('>II', data[16:24])
                if not (0 < width <= 8192 and 0 < height <= 8192):
                    raise ValueError('PNG dimensions exceed limit')
                import fitz
                decoded = fitz.Pixmap(data)
                if (decoded.width, decoded.height) != (width, height):
                    raise ValueError('PNG decode dimensions mismatch')
            except Exception as exc:
                raise ValueError('invalid inline PNG: ' + type(exc).__name__) from exc
            images += 1
            # 只改估算副本，原始载荷/hash/收据保留base64原件。
            image['url'] = 'png-sha256:' + hashlib.sha256(data).hexdigest()
    wire_bytes = len(dump({'messages': request['messages'], 'tools': request.get('tools', [])}).encode('utf-8'))
    if images and wire_bytes > 48 * 1024 * 1024:
        raise ValueError('image request exceeds transport byte limit')
    text_bytes = len(dump(projected).encode('utf-8'))
    # 2026-10-17官方Vision每图上限1024，按上限保守预留而非猜测压缩后尺寸。
    return {'reserved_tokens': text_bytes + 1024 + int(request['max_tokens']) + images * 1024,
            'method': 'utf8_bytes_plus_1024_plus_output' if not images else 'text_utf8_plus_image_upper_bound_1024_plus_output',
            'image_count': images, 'image_tokens_upper_bound': images * 1024, 'transport_bytes': wire_bytes}


def actual_usage(usage: dict) -> int | None:
    """仅接受非负整数的供应商用量；缺失、不完整和异常数字不补零。"""
    def valid(value):
        """布尔值不能作为整数用量，负值也不是合法计费。"""
        return type(value) is int and value >= 0
    total = usage.get('total_tokens')
    parts = [usage.get('prompt_tokens'), usage.get('completion_tokens')]
    if valid(total):
        # 字段自相矛盾时保留未知，避免少计供应商执行。
        return total if not all(map(valid, parts)) or sum(parts) == total else None
    return sum(parts) if all(map(valid, parts)) else None


class TokenBudgetStore:
    """以根预算行锁串行处理并行子任务预留；不持锁等待网络。"""

    def __init__(self, db):
        """复用数据库连接工厂，不隐式创建或回填策略。"""
        self.db = db

    @contextmanager
    def locked(self, root):
        """提供根任务事务；SQLite用写锁，MySQL只锁本根预算。"""
        with self.db.connection() as conn:
            tx = Transaction(self.db, conn)
            if self.db.kind == 'sqlite':
                tx.execute('BEGIN IMMEDIATE')
            row = tx.one('SELECT * FROM task_token_budgets WHERE root_task_id=?' +
                         (' FOR UPDATE' if self.db.kind == 'mysql' else ''), (root,))
            yield tx, row

    def summary(self, root: str) -> dict | None:
        """返回独立预算投影；没有新策略的旧任务返回None。"""
        # 响应已落盘但结算前退出时，即使任务随后终态，查询也能补记真实usage。
        # 只修复新账本派生计数，不改原收据、不重启任务，更不发送模型请求。
        for item in self.db.fetchall(
                "SELECT r.task_id,r.sequence_no FROM token_reservations r JOIN model_calls c "
                "ON c.task_id=r.task_id AND c.sequence_no=r.sequence_no "
                "WHERE r.root_task_id=? AND r.status='reserved' AND c.status<>'started'", (root,)):
            self.settle(item['task_id'], item['sequence_no'])
        row = self.db.fetchone('SELECT * FROM task_token_budgets WHERE root_task_id=?', (root,))
        if row:
            row['remaining_tokens'] = max(0, row['token_limit'] - row['used_tokens'] - row['reserved_tokens'])
            row['overrun_tokens'] = max(0, row['used_tokens'] - row['token_limit'])
            row['unknown_reservations'] = self.db.fetchone(
                "SELECT COUNT(*) AS n FROM token_reservations WHERE root_task_id=? AND status='unknown'", (root,))['n']
            row['estimate_method'] = 'utf8_bytes_plus_1024_plus_output'
        return row

    def reserve(self, root: str, task: str, sequence: int, request: dict, role: str) -> None:
        """发送前原子预留；同请求重入不重扣，额度不足明确拒绝。

        Args:
            root: 全部协作角色共享的根任务。
            task: 发请求的角色任务。
            sequence: 该任务持久请求序号。
            request: 实际冻结messages/tools/输出上限。
            role: 仅Supervisor可使用收尾预留。

        Raises:
            ValueError: 用量未知、身份冲突或剩余额度不足。
        """
        with self.locked(root) as (tx, budget):
            if not budget:
                return
            old = tx.one('SELECT * FROM token_reservations WHERE task_id=? AND sequence_no=?', (task, sequence))
            if old:
                if old['request_hash'] != digest(request):
                    raise ValueError('TOKEN_REQUEST_CONFLICT: 冻结请求身份不同')
                if old['status'] != 'reserved':
                    raise ValueError('TOKEN_RECEIPT_REPLAY_REQUIRED: 已结算请求不得重新发送')
                return
            if tx.one("SELECT reservation_id FROM token_reservations WHERE root_task_id=? AND status='unknown'", (root,)):
                raise ValueError('TOKEN_USAGE_UNKNOWN: 存在未结算用量，停止新增请求')
            amount = estimate_request(request)
            usable = budget['token_limit'] - (0 if role == 'supervisor' else budget['tail_tokens'])
            if budget['used_tokens'] + budget['reserved_tokens'] + amount > usable:
                raise ValueError('TASK_TOKEN_BUDGET: 累计Token预算不足，保留已有成果并结束')
            tx.execute('UPDATE task_token_budgets SET reserved_tokens=reserved_tokens+? WHERE root_task_id=?', (amount, root))
            tx.execute('INSERT INTO token_reservations(reservation_id,root_task_id,task_id,sequence_no,request_hash,reserved_tokens,status,created_at) VALUES(?,?,?,?,?,?,?,?)',
                       (task + ':' + str(sequence), root, task, sequence, digest(request), amount, 'reserved', _now()))

    def settle(self, task: str, sequence: int) -> None:
        """从已落盘收据幂等结算；响应后崩溃可回放补记，未知保持占用。"""
        reservation = self.db.fetchone('SELECT root_task_id FROM token_reservations WHERE task_id=? AND sequence_no=?', (task, sequence))
        if not reservation:
            return
        root = reservation['root_task_id']
        with self.locked(root) as (tx, budget):
            entry = tx.one('SELECT * FROM token_reservations WHERE task_id=? AND sequence_no=?', (task, sequence))
            if entry['status'] in {'settled', 'released'}:
                return
            receipt = tx.one('SELECT * FROM model_calls WHERE task_id=? AND sequence_no=?', (task, sequence))
            if not receipt or receipt['status'] == 'started':
                return
            usage = json.loads(receipt.get('usage_json') or '{}')
            amount = actual_usage(usage)
            released = receipt['status'] == 'not_sent' and receipt['sent_state'] == 'not_sent'
            if released or amount is not None:
                amount = 0 if released else amount
                tx.execute('UPDATE task_token_budgets SET reserved_tokens=reserved_tokens-?,used_tokens=used_tokens+? WHERE root_task_id=?',
                           (entry['reserved_tokens'], amount, root))
                tx.execute('UPDATE token_reservations SET status=?,actual_tokens=? WHERE reservation_id=?',
                           ('released' if released else 'settled', amount, entry['reservation_id']))
            else:
                tx.execute("UPDATE token_reservations SET status='unknown' WHERE reservation_id=?", (entry['reservation_id'],))
