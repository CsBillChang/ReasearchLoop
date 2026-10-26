"""本机Cross-encoder重排：只输出相关性logit，不生成答案或作事实判决。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading
import time


class LocalReranker:
    """锁定多语言mMiniLM ONNX模型，CPU小批量运行，不下载或执行远端代码。"""

    def __init__(self, manifest_path: str | Path, threads: int = 2, *, batch_size: int = 4,
                 sort_by_length: bool = False):
        """记录固定清单；首次实际评分才验证/加载模型，失败可明确退回RRF。"""
        if not 1 <= threads <= 4 or batch_size not in (1, 4, 8, 16) or not isinstance(sort_by_length, bool):
            raise ValueError("RERANK_CONFIG_UNSUPPORTED")
        self.manifest_path, self.threads = Path(manifest_path), threads
        self.batch_size, self.sort_by_length = batch_size, sort_by_length
        self.session = None
        self.identity = None
        self.lock = threading.RLock()
        self.last_stats = {}

    def _load(self) -> None:
        """核对权重、tokenizer、配置及版本，再分配本机推理内存。"""
        import onnxruntime as ort
        from tokenizers import Tokenizer
        spec = json.loads(self.manifest_path.read_text())["reranker"]
        if spec["repo"] != "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" or not 1 <= self.threads <= 4:
            raise ValueError("RERANK_CONFIG_UNSUPPORTED")
        folder = Path(spec["directory"])
        for name in (spec["weights"], "tokenizer.json", "config.json"):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != spec["files"][name]["sha256"]:
                raise ValueError("RERANK_MODEL_HASH_MISMATCH")
        self.tokenizer = Tokenizer.from_file(str(folder/"tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder/spec["weights"]), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.identity = {"model_id": spec["repo"], "revision": spec["revision"],
            "weights_sha256": spec["files"][spec["weights"]]["sha256"],
            "tokenizer_sha256": spec["files"]["tokenizer.json"]["sha256"],
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "max_pair_tokens": 512, "score_kind": "raw_relevance_logit", "batch_size": self.batch_size,
            "threads": self.threads, "sort_by_length": self.sort_by_length}

    def score(self, query: str, passages: list[str]) -> list[float]:
        """返回同序logit；任一输入过长时整批退回融合结果，不静默截掉证据。

        Args:
            query: 当前检索问题。
            passages: 已授权原文片段，最多64条。

        Raises:
            ValueError: 输入超容量或模型输出非有限值。
        """
        with self.lock:
            # stats仅代表本次成功评分；空输入、失败及中断均不能沿用上次成功。
            self.last_stats = {}
            if not passages:
                return []
            if len(passages) > 64:
                raise ValueError("RERANK_CANDIDATE_LIMIT")
            import numpy as np
            if self.session is None:
                self._load()
            started = time.perf_counter()
            rows = [self.tokenizer.encode(query, text) for text in passages]
            tokenized = time.perf_counter()
            if any(len(row.ids) > 512 for row in rows):
                raise ValueError("RERANK_INPUT_TOO_LONG")
            # 仅改变计算组批，不裁剪原文；评分后按原输入身份还原，不能错配引用。
            order = sorted(range(len(rows)), key=lambda i: len(rows[i].ids)) if self.sort_by_length else list(range(len(rows)))
            scores = [0.0] * len(rows)
            padded_tokens, inference_seconds = 0, 0.0
            batches = []
            for start in range(0, len(rows), self.batch_size):
                indices = order[start:start+self.batch_size]
                batch = [rows[i] for i in indices]
                size = max(len(row.ids) for row in batch)
                padded_tokens += size * len(batch)
                ids = np.ones((len(batch), size), dtype=np.int64)
                mask = np.zeros_like(ids)
                for i, row in enumerate(batch):
                    ids[i, :len(row.ids)], mask[i, :len(row.ids)] = row.ids, 1
                feed = {"input_ids": ids, "attention_mask": mask}
                if any(inp.name == "token_type_ids" for inp in self.session.get_inputs()):
                    feed["token_type_ids"] = np.zeros_like(ids)
                batches.append({"input_indices": list(indices),
                    "pair_lengths": [len(row.ids) for row in batch],
                    "feed_shapes": {name: list(value.shape) for name, value in feed.items()}})
                inference_start = time.perf_counter()
                output = self.session.run(None, feed)[0].reshape(-1)
                inference_seconds += time.perf_counter() - inference_start
                if len(output) != len(batch) or not np.isfinite(output).all():
                    raise ValueError("RERANK_OUTPUT_INVALID")
                for original, value in zip(indices, output):
                    scores[original] = float(value)
            self.last_stats = {"pairs": len(rows), "input_tokens": sum(len(r.ids) for r in rows),
                "padded_tokens": padded_tokens, "tokenize_seconds": tokenized-started,
                "inference_seconds": inference_seconds, "total_seconds": time.perf_counter()-started,
                "input_pair_lengths": [len(row.ids) for row in rows],
                "execution_order": order, "batches": batches,
                "threads": self.threads, "batch_size": self.batch_size,
                "sort_by_length": self.sort_by_length, "status": "succeeded"}
            return scores
