"""固定任务集的运行边界；只扩展共同实验层，不改变业务提示词。"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from harness_ab_boundary import Boundary


def digest(path):
    """按文件字节核对冻结身份。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_final_suite(manifest):
    """通过共同语义门校验版本，拒绝拼写错误后退回旧合同。"""
    from harness_ab_pair_gate import validate_final_suite_policy
    return validate_final_suite_policy(manifest)


def validate_suite_manifest(manifest):
    """验证96格、材料副本和目录关系，尚未批准也能做离线预检。"""
    if not is_final_suite(manifest):
        return None
    root = Path(manifest['prepared_input_root']).resolve(strict=True)
    owned = Path(manifest['experiment_root']).resolve(strict=True)
    if (root != owned/'inputs'
            or Path(manifest['runtime_root']).resolve() != owned/'runtime'
            or Path(manifest['semantic_gate_root']).resolve() != owned/'semantic-gates'
            or any(Path(manifest['source_roots'][v]).resolve() != owned/'snapshots'/v for v in ('A','B'))):
        raise PermissionError('FINAL_OWNED_EXPERIMENT_PATHS_REQUIRED')
    snapshot_path = root/'material-snapshot.json'
    if snapshot_path.is_symlink() or digest(snapshot_path) != manifest['material_snapshot_sha256']:
        raise PermissionError('FINAL_MATERIAL_SNAPSHOT_CHANGED')
    cases_path = Path(manifest['cases_path'])
    plan_path = root/'execution-plan.json'
    if (cases_path.resolve() != root/'cases.json' or cases_path.is_symlink()
            or digest(cases_path) != manifest['final_cases_sha256']
            or digest(plan_path) != manifest['final_plan_sha256']):
        raise PermissionError('FINAL_INPUT_CHANGED')
    data = json.loads(cases_path.read_text()); plan = json.loads(plan_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    copies = snapshot.get('copies',[]); materials = snapshot.get('materials',[])
    if len(copies) != 108 or len(materials) != 54 or len({m['pdf_sha256'] for m in materials}) != 54:
        raise PermissionError('FINAL_54_PDF_AND_PARSED_COPIES_REQUIRED')
    for item in copies:
        path=Path(item['copy'])
        if (path.is_symlink() or not path.resolve().is_relative_to(root/'materials')
                or path.stat().st_nlink != 1 or digest(path) != item['sha256']):
            raise PermissionError('FINAL_MATERIAL_COPY_CHANGED')
    copy_hashes={str(Path(i['copy']).resolve()):i['sha256'] for i in copies}
    if len(copy_hashes) != 108:
        raise PermissionError('FINAL_DUPLICATE_COPY_PATH')
    for item in materials:
        for key,hkey in [('pdf_path','pdf_sha256'),('parsed_path','parsed_cache_sha256')]:
            if copy_hashes.get(str(Path(item[key]).resolve())) != item[hkey]:
                raise PermissionError('FINAL_SNAPSHOT_MATERIAL_PATH_CHANGED')
    cases = data['cases']
    order = manifest['execution_order']
    if (len(cases) != 48 or len({c['case_id'] for c in cases}) != 48
            or len(order) != 96 or len({tuple(c) for c in order}) != 96
            or sorted(order) != sorted(manifest['ledger_limits']['allowed_cells'])
            or any(c.get('suite_kind') != 'agent_final_v1' for c in cases)
            or any(order.count([c['case_id'],v]) != 1 for c in cases for v in ('A','B'))):
        raise PermissionError('FINAL_96_CELL_MATRIX_REQUIRED')
    # 不允许只改执行顺序或漏掉第二轮，却保留原题数标签。
    planned = plan.get('execution_order')
    if planned != order:
        raise PermissionError('FINAL_ORDER_CHANGED')
    roots = [Path(manifest[k]).resolve() for k in ('runtime_root','semantic_gate_root')]
    roots += [Path(p).resolve() for p in manifest['source_roots'].values()]
    for other in roots:
        if other == root or other.is_relative_to(root) or root.is_relative_to(other):
            raise PermissionError('FINAL_INPUT_MUST_BE_OUTSIDE_WRITABLE_RUNTIME_AND_SOURCE')
    url = urlparse(manifest['vector_endpoint'])
    if (url.scheme != 'http' or url.hostname != '127.0.0.1'
            or url.port != manifest['vector_port']
            or url.port in {33316,6339,6351,6357,6361,6401}
            or url.username or url.password or url.path not in {'','/'}):
        raise PermissionError('FINAL_DEDICATED_VECTOR_ENDPOINT_REQUIRED')
    if Path(manifest['material_project_root']).resolve() != root:
        raise PermissionError('FINAL_RUNTIME_MATERIAL_SOURCE_MUST_BE_COPY')
    for case in cases:
        for material in case.get('materials',[]):
            path = Path(material['pdf_path'])
            if (path.is_symlink() or not path.resolve().is_relative_to(root)
                    or path.stat().st_nlink != 1 or digest(path) != material['pdf_sha256']
                    or copy_hashes.get(str(path.resolve())) != material['pdf_sha256']):
                raise PermissionError('FINAL_PDF_COPY_REQUIRED')
    return {'cases':48,'cells':96,'input_root':str(root)}


def cell_directory(manifest,case_id,variant,mode):
    """在创建目录之前限制阶段名和真实路径，禁止链接和父目录跳转。"""
    runtime=Path(manifest['runtime_root']).resolve()
    stage=manifest.get('qualification_run','qualification') if mode=='qualify' else 'cells'
    if (not isinstance(stage,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',stage)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',case_id) or variant not in {'A','B'}):
        raise PermissionError('FINAL_CELL_COMPONENT_INVALID')
    cell=runtime/stage/(case_id+'-'+variant)
    if cell.resolve()!=cell or not cell.is_relative_to(runtime):
        raise PermissionError('FINAL_CELL_PATH_REDIRECTED')
    return cell


class CellBoundary(Boundary):
    """每格只写自己的目录，整批预算SQLite是唯一共享可写文件。"""

    def __init__(self, cell, runtime):
        cell=Path(cell);runtime=Path(runtime).resolve()
        if cell.resolve()!=cell or not cell.is_relative_to(runtime) or cell==runtime:
            raise PermissionError('FINAL_CELL_OUTSIDE_RUNTIME')
        super().__init__(cell)
        self.ledger = runtime/'budget.sqlite3'
        self.shared_files = {self.ledger, *(self.ledger.with_name(self.ledger.name+s)
            for s in ('-journal','-wal','-shm'))}

    def allowed_path(self, value, fd=-1):
        """解析链接后才判断共享账本，不能借同名路径指向其它格。"""
        if super().allowed_path(value,fd):
            return True
        path = Path(os.fsdecode(value))
        if not path.is_absolute() and fd not in {-1,None}:
            path = self.fd_root(fd)/path
        return path in self.shared_files and path.resolve() == path

    def audit(self,event,args):
        """数据库只允许本格主库和整批账本，禁止连其它格或部署库。"""
        if self.enabled and event == 'sqlite3.connect':
            value=args[0]
            if value != ':memory:' and Path(value).resolve() not in {self.output/'task.sqlite3',self.ledger}:
                self.reject('sqlite_not_current_cell_or_shared_ledger')
        return super().audit(event,args)
