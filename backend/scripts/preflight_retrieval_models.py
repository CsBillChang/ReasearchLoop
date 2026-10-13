"""AG7.1本机模型组件预检；不连接数据库、论文源或生成模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path


PAIRS = [
    ("如何用语言识别训练中没有出现过的物体类别？", "Open vocabulary object detection aligns image regions with language embeddings of unseen category names."),
    ("怎样在跨对话时记住用户的研究偏好？", "Long term memory stores user preferences and retrieves related research from previous conversations."),
    ("如何确保进程重启不会重复保存笔记？", "Idempotent file writes use persistent receipts and content hashes to recover after process failure."),
    ("光合作用如何产生氧气？", "Photosynthesis uses light energy to convert carbon dioxide and water into sugars and releases oxygen."),
]


def run(role: str, manifest: Path) -> dict:
    """在单进程顺序测一个候选，报告资源与合成跨语言冒烟，非科研评测。"""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    start = time.perf_counter()
    queries, docs = zip(*PAIRS)
    if role == "embedding":
        from research_loop.providers.embeddings import LocalEmbedding
        model = LocalEmbedding(manifest)
        vectors = np.array(model.encode(list(docs)))
        q = np.array(model.encode(list(queries), query=True))
        rankings = (q @ vectors.T).argmax(axis=1).tolist()
        cold = time.perf_counter()-start
        t = time.perf_counter()
        for _ in range(4):
            model.encode([d * 10 for d in docs])
        warm = time.perf_counter()-t
        details = {"identity": model.identity, "dimension": vectors.shape[1],
                   "norms": np.linalg.norm(vectors, axis=1).tolist()}
    else:
        spec = json.loads(manifest.read_text())[role]
        folder = Path(spec["directory"])
        for name in (spec["weights"], "tokenizer.json"):
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == spec["files"][name]["sha256"]
        tokenizer = Tokenizer.from_file(str(folder/"tokenizer.json"))
        tokenizer.no_truncation(); tokenizer.no_padding()
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(folder/spec["weights"]), sess_options=options, providers=["CPUExecutionProvider"])

        def score(query, passages):
            """成对编码，不用截断掩盖输入容量。"""
            rows = [tokenizer.encode(query, p) for p in passages]
            n = max(len(r.ids) for r in rows)
            assert n <= 512
            ids = np.ones((len(rows), n), dtype=np.int64); mask = np.zeros_like(ids)
            for i, r in enumerate(rows):
                ids[i, :len(r.ids)] = r.ids; mask[i, :len(r.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if any(i.name == "token_type_ids" for i in session.get_inputs()):
                feed["token_type_ids"] = np.zeros_like(ids)
            return session.run(None, feed)[0].reshape(-1)

        rankings = [int(score(q, docs).argmax()) for q in queries]
        cold = time.perf_counter()-start
        t = time.perf_counter()
        for q in queries: score(q, [d*10 for d in docs])
        warm = time.perf_counter()-t
        details = {"repo": spec["repo"], "revision": spec["revision"], "precision": "int8-arm64",
                   "weights_sha256": spec["files"][spec["weights"]]["sha256"]}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    passed = rankings == [0, 1, 2, 3] and warm < 30 and peak < 2*1024**3 and cold < 90
    return {"role": role, "passed": passed, "synthetic_smoke_only": True,
            "rankings": rankings, "cold_load_and_smoke_seconds": cold,
            "warm_16_long_inputs_seconds": warm, "peak_rss_bytes_macos": peak,
            "cpu_threads": 2, "criteria": {"correct_synthetic_top1": 4, "warm_seconds_lt": 30,
            "peak_rss_bytes_lt": 2*1024**3, "cold_seconds_lt": 90}, **details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["embedding", "reranker"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.role, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
