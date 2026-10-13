"""AG7.4评测共用的隔离、冻结和指标函数；不改变产品提示词或成功判定。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from research_loop.config import Settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/ag7-4-evaluation-20261011'


def sha(path: Path) -> str:
    """计算原始文件hash，冻结后不允许静默更换材料。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive(path: Path, value) -> None:
    """独占写入产物；重复运行必须停止，不能覆盖旧成绩。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def settings() -> Settings:
    """从当前配置派生本机独立数据库/索引/输出，不返回或记录认证信息。"""
    current = Settings.from_env(ROOT)
    parsed = urlparse(current.database_url)
    if parsed.scheme != 'mysql' or parsed.hostname not in {'localhost', '127.0.0.1'}:
        raise ValueError('AG74_LOCAL_MYSQL_REQUIRED')
    return current.model_copy(update={
        'database_url': urlunparse(parsed._replace(path='/researchloop_test_ag74_20261011')),
        'data_dir': OUT/'data', 'output_dir': OUT/'notes',
        'retrieval_namespace': 'researchloop_test_ag74_20261011',
        'max_model_requests': 8, 'max_tool_actions': 20, 'reserve_model_requests': 2,
        'max_reflection_rounds': 1, 'request_timeout_seconds': 120,
        'discovery_backends': {},
    })


def ranked_metrics(ranked: list[str], relevant: list[str], *, answerable: bool,
                   corpus: set[str], k: int = 10) -> dict:
    """以冻结标注集计算Recall@k和候选排名MRR；缺证题不记为满分。

    Args:
        ranked: 有序且唯一的片段身份，最多冻结的候选数。
        relevant: 事前标注的相关片段，不代表穷举所有可能相关片段。
        answerable: False时不计算Recall/MRR，不用检索分数判断事实缺证。
        corpus: 本题允许的语料身份集合。
        k: 固定召回截点。

    Raises:
        ValueError: 重复/越界排名或标注与可回答状态矛盾。
    """
    if k != 10 or len(ranked) > 24 or len(set(ranked)) != len(ranked) or len(set(relevant)) != len(relevant):
        raise ValueError('INVALID_RANKING_OR_QRELS')
    if not set(ranked).issubset(corpus) or not set(relevant).issubset(corpus):
        raise ValueError('OUT_OF_SCOPE_ID')
    if bool(relevant) != answerable:
        raise ValueError('ANSWERABILITY_QRELS_MISMATCH')
    first = next((i for i, value in enumerate(ranked, 1) if value in relevant), None)
    return {'recall_at_10': len(set(ranked[:k]) & set(relevant))/len(relevant) if answerable else None,
            'reciprocal_rank_at_24': (1/first if first else 0.0) if answerable else None,
            'first_relevant_rank': first, 'returned': len(ranked), 'gold_count': len(relevant)}


def verify_freeze() -> dict:
    """核对材料、标注、源码与模型清单hash；历史批次不因代码更新重新执行。"""
    frozen = json.loads((OUT/'freeze.json').read_text())
    for rel, expected in frozen['files'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('AG74_FROZEN_FILE_CHANGED: '+rel)
    return frozen


def freeze() -> None:
    """在任何排名或生成结果之前封存输入、标注、源码、模型身份与预定口径。"""
    from datetime import datetime, timezone
    import platform
    import sys
    if (OUT/'retrieval/started.json').exists() or (OUT/'reflection/started.json').exists():
        raise ValueError('RESULTS_ALREADY_STARTED')
    spec=json.loads((ROOT/'backend/tests/acceptance/ag74_cases.json').read_text())
    corpus=json.loads((OUT/'corpus.json').read_text())
    cases=json.loads((OUT/'retrieval-cases.json').read_text())
    assert len(cases)==24 and sum(c['answerable'] for c in cases)==21
    assert len(corpus['documents'])==6 and len(corpus['chunks'])==282
    assert sum(len(d['pages']) for d in corpus['documents'])==63
    assert len(spec['reflection'])==4 and sum(len(c['claims']) for c in spec['reflection'])==8
    models=json.loads((ROOT/'data/ag71-preflight/models/manifest.json').read_text())
    files=list((ROOT/'backend/research_loop').rglob('*.py'))
    files+=list((ROOT/'backend/scripts').glob('*.py'))
    files+=[ROOT/'backend/tests/acceptance/ag74_cases.json',ROOT/'backend/tests/unit/test_evaluation.py',
            OUT/'corpus.json',OUT/'retrieval-cases.json',ROOT/'data/ag71-preflight/models/manifest.json',
            ROOT/'backend/uv.lock',ROOT/'backend/pyproject.toml']
    files+=[ROOT/d['material']['path'] for d in corpus['documents']]
    for model in models.values():
        for name,identity in model['files'].items():
            p=__import__('pathlib').Path(model['directory'])/name
            if sha(p)!=identity['sha256']:raise ValueError('LOCAL_MODEL_CHANGED')
    exclusive(OUT/'freeze.json',{'created_at':datetime.now(timezone.utc).isoformat(),
        'files':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'environment':{'python':sys.version,'platform':platform.platform(),'cpu':'Apple M2','ram_gib':8,
                       'inference_threads':2,'generative_model':settings().deepseek_model},
        'corpus':{'documents':6,'physical_pages':63,'chunks':282,'new_downloads':0},
        'retrieval':{'modes':['bm25','vector','rrf','rrf_rerank'],'questions':24,'answerable':21,'missing':3,
            'candidate_depth':24,'recall_cutoff':10,'rrf_k':60,'generative_cap':0,
            'qrels':'Agent事前选定且逐页绑定的原句片段；跨块全部计入；非穷举相关性/双人标注金标准',
            'metrics':'21题macro Recall@10、MRR@24；3缺证题单列不计分；每通道预检后单次热运行',
            'latency':'共享实测组件用时相加；median/P95 nearest rank；不代表生产端到端延迟',
            'resources':'首次索引build秒数及本进程峰值RSS，模型冷初始化单列，Qdrant RSS不包含',
            'tuning_cases':0,'report_cases':24,'retrieval_answer_generation':False},
        'reflection':{**spec['reflection_budget'],'cases':['F1','F2','F3','F4'],
            'gold_claims':{'supported':5,'contradicted':2,'insufficient':1},
            'baseline':'same frozen unchanged drafts; 0 requests; not a separate unrestricted agent run',
            'execution':'API创建隔离任务＋服务绑定受控草稿→真实TaskExecutor→既有按需Reflection；无会话防止Memory混杂',
            'metrics':'可支持主张/8、错误修复/2、缺证处理/1、误报/5、不必要改写/5、未解决数、任务终态/4、独立复核、requests/tokens/seconds',
            'completion':'完整=交付当前稿且正确修复/保留/披露；准确报告未解决的有界partial单列；未改错不算正确修复',
            'stop':'unknown、供应商失败或缺usage停止余下发送；质量负向保留分母、不重跑',
            'protocol_changes':0,'new_paper_searches':0,'full_reading_regeneration':0},
        'acceptance':'正确实验/核对/账本/旧资产保护是完成门；正向提升不是完成门；不占AG7.5最终12次额度'})
    print('AG7.4 frozen; no ranked results or generative calls yet')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','verify'])
    (freeze if parser.parse_args().mode=='freeze' else verify_freeze)()
