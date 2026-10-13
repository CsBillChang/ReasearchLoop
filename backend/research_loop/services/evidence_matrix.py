"""把已读原文组织成不可变矩阵；复用工具行动存储，无额外模型或文件写入。"""
from __future__ import annotations

import hashlib
import json
from research_loop.domain.research import DIMENSIONS, EvidenceMatrixInput
from research_loop.services.verification import quote_spans
from research_loop.storage.memory import MemoryStore, check_owner


def digest(value) -> str:
    """计算稳定内容身份，不将时间、展示状态或自动评审推断混入版本。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def quote_span(text: str, quote: str) -> tuple[int, int]:
    """兼容空白和真实字母跨行断词，返回唯一的原文区间。"""
    matches = quote_spans(quote, text)
    if not matches:
        raise ValueError('quote not found on the cited page; copy a continuous source excerpt, preserving numbers and negation')
    if len(matches) != 1:
        raise ValueError(f'quote matches {len(matches)} positions on the cited page; extend the excerpt to locate it uniquely')
    return matches[0]


class EvidenceMatrixService:
    """任务级矩阵的结构校验、版本读回和行动事务提交。"""

    def __init__(self, db):
        """注入已有权威数据库，不自动迁移或修改记录。"""
        self.db = db

    def _root(self, task_id):
        """只允许根任务矩阵，避免跨家族或子任务冒充汇合产物。"""
        task = self.db.get_task(task_id)
        if not task or task['root_task_id'] != task_id:
            raise ValueError('matrix must belong to an existing root task')
        return task

    def _page(self, task_id, evidence_id, document_id):
        """仅查询当前任务家族已实际获授的同版本页，不以全库存在替代授权。"""
        row = self.db.fetchone('SELECT e.* FROM page_evidence e JOIN task_evidence te ON te.evidence_id=e.evidence_id '
            'JOIN tasks t ON t.task_id=te.task_id WHERE t.root_task_id=? AND e.evidence_id=? AND e.document_id=?',
            (task_id, evidence_id, document_id))
        if not row:
            raise ValueError('evidence is outside the task family or declared document: ' + evidence_id)
        return row

    def prepare(self, task_id: str, args: EvidenceMatrixInput) -> dict:
        """从真实页派生矩阵，不写数据库；提交仍需租约及前版事务校验。

        Args:
            task_id: Supervisor根任务。
            args: 五维单元格与引句，不能携带自称页码或来源hash。

        Returns:
            带内容hash的独立产物；status是Agent判断，validation仅表示引句/身份检查。

        Raises:
            ValueError: 身份、引句、前版或范围不合法。
        """
        self._root(task_id)
        rows, errors = [], []
        for row_index, source in enumerate(args.rows):
            doc = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (source.document_id,))
            if not doc:
                raise ValueError('unknown document')
            pages = {eid: self._page(task_id, eid, doc['document_id']) for eid in source.inspected_evidence_ids}
            paper = self.db.get_paper(doc['paper_id'])
            row = {k: doc[k] for k in ('document_id', 'paper_id', 'content_sha256', 'parser_name', 'parser_version', 'config_sha256')}
            row['title'] = (paper or {}).get('metadata', {}).get('title') or doc['paper_id']
            row['scope'] = [{'evidence_id': eid, 'page_number': page['page_number'] or None,
                             'text_hash': hashlib.sha256(page['text'].encode()).hexdigest(),
                             'gaps': json.loads(page['gaps_json'])} for eid, page in sorted(pages.items())]
            row['cells'] = []
            for cell_index, cell in enumerate(source.cells):
                value = cell.model_dump(exclude={'references'})
                value['references'] = []
                for ref_index, ref in enumerate(cell.references):
                    locator = (f'rows[{row_index}] document_id={source.document_id} '
                               f'cells[{cell_index}] dimension={cell.dimension} '
                               f'references[{ref_index}] evidence_id={ref.evidence_id}')
                    if ref.evidence_id not in pages:
                        errors.append(locator + ': reference must belong to inspected_evidence_ids')
                        continue
                    page = pages[ref.evidence_id]
                    try:
                        start, end = quote_span(page['text'], ref.quote)
                    except ValueError as exc:
                        errors.append(locator + ': ' + str(exc))
                        continue
                    value['references'].append({'evidence_id': ref.evidence_id, 'page_number': page['page_number'] or None,
                        'quote': page['text'][start:end], 'start': start, 'end': end,
                        'text_hash': hashlib.sha256(page['text'].encode()).hexdigest()})
                row['cells'].append(value)
            row['cells'].sort(key=lambda c: DIMENSIONS.index(c['dimension']))
            rows.append(row)
        if errors:
            # 一次返回全部具体位置；不保存半份矩阵，也不替模型补造引句。
            raise ValueError('matrix reference errors:\n' + '\n'.join(errors))
        matrix = {'task_id': task_id, 'question': args.question, 'rows': rows,
                  'validation': 'source_identity_and_quote_only', 'semantic_review': 'not_performed'}
        matrix['matrix_hash'] = digest(matrix)
        latest = self.latest(task_id)
        if latest and latest['matrix_hash'] == matrix['matrix_hash']:
            return matrix
        if args.base_matrix_hash != (latest['matrix_hash'] if latest else None):
            raise ValueError('matrix version changed; use the latest base_matrix_hash')
        return matrix

    def commit_action(self, action: dict, owner: tuple | None) -> None:
        """矩阵成功观察与租约/版本校验同事务提交；进程退出后可重放原行动。

        Args:
            action: 图节点完成校验后的成功行动。
            owner: (task_id, worker_id, lease_token)，来自Worker而非模型。
        """
        task_id = action['task_id']
        matrix = action['observation']['data']['evidence_matrix']
        with MemoryStore(self.db).transaction() as tx:
            if owner:
                check_owner(tx, owner)
            else:
                task = tx.one('SELECT status,cancel_requested FROM tasks WHERE task_id=?', (task_id,))
                if not task or task['status'] != 'running' or task['cancel_requested']:
                    raise ValueError('matrix commit requires an active task')
            prior = tx.one('SELECT * FROM actions WHERE task_id=? AND sequence_no=?', (task_id, action['sequence']))
            if prior and prior['status'] == 'succeeded':
                if json.loads(prior['observation_json'])['data']['evidence_matrix']['matrix_hash'] != matrix['matrix_hash']:
                    raise ValueError('successful matrix action cannot be replaced')
                return
            latest = tx.one("SELECT observation_json FROM actions WHERE task_id=? AND tool_name='record_evidence_matrix' AND status='succeeded' ORDER BY sequence_no DESC LIMIT 1", (task_id,))
            latest_hash = json.loads(latest['observation_json'])['data']['evidence_matrix']['matrix_hash'] if latest else None
            if latest_hash != matrix['matrix_hash'] and action['arguments'].get('base_matrix_hash') != latest_hash:
                raise ValueError('matrix version changed before commit')
            if not prior:
                raise ValueError('persist the action intent before matrix commit')
            tx.execute('UPDATE actions SET status=?,observation_json=? WHERE task_id=? AND sequence_no=?',
                ('succeeded', json.dumps(action['observation'], ensure_ascii=False), task_id, action['sequence']))

    def history(self, task_id: str, check_integrity: bool = True) -> list[dict]:
        """读回去重版本；原页变化时保留旧矩阵并标记stale，不静默生成新证据。"""
        self._root(task_id)
        result, seen = [], set()
        for action in reversed(self.db.list_actions(task_id)):
            if action['tool_name'] != 'record_evidence_matrix' or action['status'] != 'succeeded':
                continue
            matrix = action['observation']['data']['evidence_matrix']
            if matrix['matrix_hash'] in seen:
                continue
            seen.add(matrix['matrix_hash'])
            valid_hash = digest({k: v for k, v in matrix.items() if k != 'matrix_hash'}) == matrix['matrix_hash']
            matrix.update(action_id=action['action_id'], created_at=action['created_at'], integrity='intact')
            if check_integrity:
                try:
                    if not valid_hash:
                        raise ValueError('matrix payload hash mismatch')
                    for row in matrix['rows']:
                        doc = self.db.fetchone('SELECT * FROM documents WHERE document_id=?', (row['document_id'],))
                        if not doc or any(doc[k] != row[k] for k in ('paper_id','content_sha256','parser_name','parser_version','config_sha256')):
                            raise ValueError('source version changed')
                        for scope in row['scope']:
                            page = self._page(task_id, scope['evidence_id'], row['document_id'])
                            if hashlib.sha256(page['text'].encode()).hexdigest() != scope['text_hash'] or (page['page_number'] or None) != scope['page_number']:
                                raise ValueError('source page changed')
                except ValueError as exc:
                    matrix.update(integrity='stale', integrity_reason=str(exc))
            result.append(matrix)
        return result

    def latest(self, task_id: str) -> dict | None:
        """返回最近版本，包括其来源是否仍完整；不等于语义审校通过。"""
        return next(iter(self.history(task_id)), None)

    def delivery_ref(self, task_id: str, matrix_hash: str) -> dict:
        """显式交付只能引用当前且来源完整的矩阵，防止答复绑定过期产物。"""
        latest = self.latest(task_id)
        if not latest or latest['matrix_hash'] != matrix_hash or latest['integrity'] != 'intact':
            raise ValueError('delivery requires current intact evidence matrix')
        counts = {status: sum(cell['status'] == status for r in latest['rows'] for cell in r['cells'])
                  for status in ('evidenced', 'missing', 'uncertain', 'conflict')}
        return {'matrix_hash': matrix_hash, 'cell_counts': counts, 'semantic_review': 'not_performed'}
