"""本机ONNX编码适配：只读固定文件，不下载模型、不执行远端模型代码。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


class LocalEmbedding:
    """多语言E5-small本机编码；依赖按需导入，不影响未启用索引的旧流程。"""

    def __init__(self, manifest_path: str | Path, threads: int = 2):
        """加载并验证冻结模型文件，暂不分配ONNX模型内存。

        Args:
            manifest_path: 含embedding条目的下载清单。
            threads: CPU线程上限；本机8GB配置默认两线程。

        Raises:
            ValueError: 文件hash、模型类型或线程配置不符。
        """
        from tokenizers import Tokenizer
        spec = json.loads(Path(manifest_path).read_text())["embedding"]
        if spec["repo"] != "intfloat/multilingual-e5-small" or not 1 <= threads <= 4:
            raise ValueError("EMBEDDING_CONFIG_UNSUPPORTED")
        self.directory = Path(spec["directory"])
        for name in (spec["weights"], "tokenizer.json", "config.json"):
            digest = hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            if digest != spec["files"][name]["sha256"]:
                raise ValueError("MODEL_FILE_HASH_MISMATCH: " + name)
        self.tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.threads, self.spec, self.session = threads, spec, None
        self.dimension = 384
        self.tokenizer_id = spec["files"]["tokenizer.json"]["sha256"]
        self.identity = {"model_id": spec["repo"], "revision": spec["revision"],
            "weights_sha256": spec["files"][spec["weights"]]["sha256"],
            "tokenizer_sha256": self.tokenizer_id, "dimension": self.dimension,
            "pooling": "masked_mean_l2", "max_tokens": 512,
            "prefixes": {"document": "passage: ", "query": "query: "},
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "tokenizers": importlib.metadata.version("tokenizers")}

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """返回未截断原文的字符区间，用于无内容丢失的分块。"""
        return self.tokenizer.encode(text, add_special_tokens=False).offsets

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """计算归一化向量；超出512-token输入直接拒绝，不静默截断。

        Args:
            texts: 已按当前分词器分块的原文或查询。
            query: True时使用E5 query前缀，默认passage前缀。

        Returns:
            与输入同序的384维有限浮点向量。
        """
        import numpy as np
        import onnxruntime as ort
        if not texts:
            return []
        rows = [self.tokenizer.encode(("query: " if query else "passage: ") + t) for t in texts]
        if any(len(row.ids) > 512 for row in rows):
            raise ValueError("EMBEDDING_INPUT_TOO_LONG")
        if self.session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(self.directory / self.spec["weights"]),
                sess_options=options, providers=["CPUExecutionProvider"])
        size = max(len(row.ids) for row in rows)
        ids = np.full((len(rows), size), 1, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row.ids)] = row.ids
            mask[i, :len(row.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        for inp in self.session.get_inputs():
            if inp.name == "token_type_ids":
                feed[inp.name] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        if pooled.shape != (len(texts), self.dimension) or not np.isfinite(pooled).all():
            raise ValueError("EMBEDDING_OUTPUT_INVALID")
        return pooled.tolist()
