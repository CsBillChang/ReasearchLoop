"""本机 MPS FP16 重排；沿用 CPU 的原文分词与组批合同。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import threading
from types import SimpleNamespace

from research_loop.providers.reranker import LocalReranker


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
_TORCH_CONFIGURATION_LOCK = threading.Lock()


def _verify_file(folder: Path, name: str, files: dict) -> str:
    """逐块核对实际文件，避免为大权重再分配整份内存。"""
    expected = files[name]["sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("RERANK_MODEL_HASH_INVALID")
    digest = hashlib.sha256()
    with (folder / name).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("RERANK_MODEL_HASH_MISMATCH")
    return expected


def _loading_report(info: dict, weights: Path) -> dict:
    """只容许旧 checkpoint 的确定性 position_ids 非参数缓冲区。"""
    fields = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if (not isinstance(info, dict) or set(info) != fields
            or any(not isinstance(info[key], (list, set)) for key in fields)
            or any(info[key] for key in fields - {"unexpected_keys"})):
        raise ValueError("RERANK_MODEL_LOADING_KEYS_CHANGED")
    ignored = list(info["unexpected_keys"])
    if (any(key != "roberta.embeddings.position_ids" for key in ignored)
            or len(ignored) != len(set(ignored))):
        raise ValueError("RERANK_MODEL_LOADING_KEYS_CHANGED")
    if ignored:
        with weights.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("RERANK_POSITION_BUFFER_INVALID")
            length = struct.unpack("<Q", prefix)[0]
            if not 1 <= length <= 16 * 1024 * 1024:
                raise ValueError("RERANK_POSITION_BUFFER_INVALID")
            try:
                header = json.loads(stream.read(length))
                buffer = header["roberta.embeddings.position_ids"]
                offsets = buffer["data_offsets"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("RERANK_POSITION_BUFFER_INVALID") from exc
            if (buffer.get("dtype") != "I64" or buffer.get("shape") != [1, 514]
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(value) is not int or value < 0 for value in offsets)
                    or offsets[1] - offsets[0] != 514 * 8):
                raise ValueError("RERANK_POSITION_BUFFER_INVALID")
            stream.seek(8 + length + offsets[0])
            values = stream.read(514 * 8)
            if len(values) != 514 * 8 or struct.unpack("<514q", values) != tuple(range(514)):
                raise ValueError("RERANK_POSITION_BUFFER_INVALID")
    return {"ignored_nonparameter_buffers": sorted(ignored),
            "ignored_buffer_values_verified": bool(ignored)}


def _configure_torch(torch) -> None:
    """相同配置不重设 inter-op，避免第二个实例触发 Torch 一次性设置错误。"""
    with _TORCH_CONFIGURATION_LOCK:
        try:
            if torch.get_num_threads() != 4:
                torch.set_num_threads(4)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise ValueError("RERANK_THREAD_CONFIG_UNAVAILABLE") from exc
        if torch.get_num_threads() != 4 or torch.get_num_interop_threads() != 1:
            raise ValueError("RERANK_THREAD_CONFIG_UNAVAILABLE")


def _mps_device(value) -> bool:
    """接受 Torch 的 mps 或 mps:0 设备名。"""
    return re.fullmatch(r"mps(?::[0-9]+)?", str(value)) is not None


class MpsSession:
    """为原 LocalReranker 提供同形状会话，检查实际设备、精度和完整输出。"""

    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self._verify_model()

    def _verify_model(self) -> None:
        """每批检查，避免载入后意外迁移或切回训练状态。"""
        parameters = list(self.model.parameters())
        if (not parameters or any(not _mps_device(p.device) for p in parameters)
                or any(p.dtype != self.torch.float16 for p in parameters if p.is_floating_point())
                or not any(p.is_floating_point() for p in parameters)
                or any(module.training for module in self.model.modules())):
            raise ValueError("RERANK_MPS_MODEL_DEVICE_OR_DTYPE_INVALID")

    def get_inputs(self):
        """XLM-R 的 token_type_ids 保持模型默认零值，与实验输入一致。"""
        return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask")]

    def run(self, output_names, feed):
        """实际传入整数 token，推理后同步并返回同序 float32 logits。"""
        import numpy as np
        if output_names is not None or set(feed) != {"input_ids", "attention_mask"}:
            raise ValueError("RERANK_MPS_INPUT_INVALID")
        ids, mask = feed["input_ids"], feed["attention_mask"]
        if (not isinstance(ids, np.ndarray) or not isinstance(mask, np.ndarray)
                or ids.dtype != np.int64 or mask.dtype != np.int64
                or ids.ndim != 2 or mask.shape != ids.shape
                or not 1 <= ids.shape[0] <= 4 or not 1 <= ids.shape[1] <= 512
                or np.any(ids < 0) or not np.isin(mask, [0, 1]).all()
                or not np.all(mask.sum(axis=1) > 0)):
            raise ValueError("RERANK_MPS_INPUT_INVALID")
        self._verify_model()
        tensors = {name: self.torch.as_tensor(value, dtype=self.torch.int64, device="mps")
                   for name, value in feed.items()}
        for name, tensor in tensors.items():
            if (not _mps_device(tensor.device) or tensor.dtype != self.torch.int64
                    or not np.array_equal(tensor.detach().cpu().numpy(), feed[name])):
                raise ValueError("RERANK_MPS_INPUT_TRANSFER_INVALID")
        self.torch.mps.synchronize()
        with self.torch.inference_mode():
            output = self.model(**tensors).logits
        self.torch.mps.synchronize()
        if not _mps_device(output.device) or output.dtype != self.torch.float16:
            raise ValueError("RERANK_MPS_OUTPUT_DEVICE_OR_DTYPE_INVALID")
        result = output.detach().float().cpu().numpy()
        if result.shape != (ids.shape[0], 1) or not np.isfinite(result).all():
            raise ValueError("RERANK_OUTPUT_INVALID")
        return [result]


class MpsReranker(LocalReranker):
    """固定 revision 的 MPS FP16 后端；首次评分才加载本机文件。"""

    def __init__(self, manifest_path: str | Path):
        """沿用原始查询、原文、batch4、原顺序及 512 token 拒绝上限。"""
        super().__init__(manifest_path, threads=4, batch_size=4, sort_by_length=False)

    def _load(self) -> None:
        """验证同版本权重和原 tokenizer，只从本机加载，不自动转 CPU。"""
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            raise ValueError("RERANK_MPS_CPU_FALLBACK_FORBIDDEN")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        try:
            manifest = json.loads(self.manifest_path.read_text())
            original, floating = manifest["reranker"], manifest["reranker_fp16"]
            for spec in (original, floating):
                if spec["repo"] != MODEL_ID or spec["revision"] != MODEL_REVISION:
                    raise ValueError("RERANK_CONFIG_UNSUPPORTED")
            if floating["weights"] != "model.safetensors":
                raise ValueError("RERANK_CONFIG_UNSUPPORTED")
            token_folder, float_folder = Path(original["directory"]), Path(floating["directory"])
            tokenizer_hash = _verify_file(token_folder, "tokenizer.json", original["files"])
            _verify_file(token_folder, "config.json", original["files"])
            weights_hash = _verify_file(float_folder, "model.safetensors", floating["files"])
            _verify_file(float_folder, "config.json", floating["files"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("RERANK_MANIFEST_INVALID") from exc
        except OSError as exc:
            raise ValueError("RERANK_MODEL_FILES_UNAVAILABLE") from exc
        try:
            import torch
            from tokenizers import Tokenizer
            from transformers import AutoModelForSequenceClassification
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise ValueError("RERANK_MPS_DEPENDENCIES_UNAVAILABLE") from exc
        if not torch.backends.mps.is_available():
            raise ValueError("RERANK_MPS_UNAVAILABLE")
        _configure_torch(torch)
        transformers_logging.disable_progress_bar()
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(float_folder), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, attn_implementation="eager", dtype=torch.float16,
            output_loading_info=True)
        loading_compatibility = _loading_report(loading_info, float_folder / "model.safetensors")
        if (type(model).__name__ != "XLMRobertaForSequenceClassification"
                or model.config.model_type != "xlm-roberta" or model.config.num_labels != 1
                or model.config.hidden_size != 384 or model.config.num_hidden_layers != 12):
            raise ValueError("RERANK_MODEL_ARCHITECTURE_INVALID")
        model.to("mps")
        model.eval()
        torch.mps.synchronize()
        tokenizer = Tokenizer.from_file(str(token_folder / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        session = MpsSession(model, torch)
        identity = {"backend": "pytorch_mps", "variant": "mps_fp16", "model_id": MODEL_ID,
            "revision": MODEL_REVISION, "weights_sha256": weights_hash,
            "tokenizer_sha256": tokenizer_hash, "device": "mps", "dtype": "torch.float16",
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "max_pair_tokens": 512, "score_kind": "raw_relevance_logit", "batch_size": 4,
            "threads": 4, "inter_op_threads": 1, "sort_by_length": False,
            "attention_implementation": "eager", "cpu_fallback_enabled": False,
            "loading_compatibility": loading_compatibility}
        # 所有检查完成后一起发布，失败不留下可继续使用的半初始化会话。
        self.tokenizer, self.identity, self.session = tokenizer, identity, session
