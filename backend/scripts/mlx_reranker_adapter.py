"""同版 XLM-R 重排器的隔离 MLX 适配，不接入业务默认。

按 Hugging Face Transformers v5.20.2 的 XLM-R 数学结构独立实现。
参考实现采用 Apache-2.0，见其 modeling_xlm_roberta.py 文件头；MLX 为 MIT。
本模块不复制 mlx-embeddings 或其他 GPL 实现，且仅在构造会话时导入 MLX。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
VARIANTS = ("mlx_fp32", "mlx_w8a32")
GROUP_SIZE = 64
BITS = 8
LEGACY_BUFFER = "roberta.embeddings.position_ids"
FIXED_CONFIG = {
    "model_type": "xlm-roberta", "hidden_size": 384,
    "intermediate_size": 1536, "num_attention_heads": 12,
    "num_hidden_layers": 12, "vocab_size": 250002,
    "max_position_embeddings": 514, "type_vocab_size": 1,
    "pad_token_id": 1, "hidden_act": "gelu", "layer_norm_eps": 1e-5,
    "position_embedding_type": "absolute",
}


class AdapterContractError(ValueError):
    """模型身份、输入、量化配置或GPU完成证明不符合冻结合同。"""


def validate_config(config: dict) -> None:
    """只接收已批准的单logit编码器，拒绝同名架构的静默变体。"""
    if any(config.get(k) != value for k, value in FIXED_CONFIG.items()):
        raise AdapterContractError("MLX_MODEL_CONFIG_CHANGED")
    if (config.get("architectures") != ["XLMRobertaForSequenceClassification"]
            or config.get("id2label") != {"0": "LABEL_0"}
            or config.get("is_decoder", False)
            or config.get("add_cross_attention", False)):
        raise AdapterContractError("MLX_MODEL_ARCHITECTURE_CHANGED")


def parameter_shapes() -> dict[str, tuple[int, ...]]:
    """列出全部201个学习参数；位置ID仅为单独验证的历史缓冲。"""
    shapes = {
        "roberta.embeddings.word_embeddings.weight": (250002, 384),
        "roberta.embeddings.position_embeddings.weight": (514, 384),
        "roberta.embeddings.token_type_embeddings.weight": (1, 384),
    }
    for suffix in ("weight", "bias"):
        shapes[f"roberta.embeddings.LayerNorm.{suffix}"] = (384,)
    for i in range(12):
        p = f"roberta.encoder.layer.{i}"
        for name, out_dim, in_dim in (
            ("attention.self.query", 384, 384),
            ("attention.self.key", 384, 384),
            ("attention.self.value", 384, 384),
            ("attention.output.dense", 384, 384),
            ("intermediate.dense", 1536, 384),
            ("output.dense", 384, 1536),
        ):
            shapes[f"{p}.{name}.weight"] = (out_dim, in_dim)
            shapes[f"{p}.{name}.bias"] = (out_dim,)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            for suffix in ("weight", "bias"):
                shapes[f"{p}.{name}.{suffix}"] = (384,)
    shapes.update({"classifier.dense.weight": (384, 384),
                   "classifier.dense.bias": (384,),
                   "classifier.out_proj.weight": (1, 384),
                   "classifier.out_proj.bias": (1,)})
    return shapes


def linear_shapes() -> dict[str, tuple[int, int]]:
    """固定74个线性层，含分类头；不量化Embedding或归一化。"""
    return {key[:-7]: shape for key, shape in parameter_shapes().items()
            if key.endswith(".weight") and len(shape) == 2
            and "embeddings." not in key}


def validate_parameter_metadata(metadata: dict, legacy_values=None) -> dict:
    """严格覆盖每个学习参数，禁止缺失或多余参数被静默初始化。"""
    shapes = parameter_shapes()
    expected = set(shapes)
    actual = set(metadata)
    has_legacy = LEGACY_BUFFER in actual
    if actual - {LEGACY_BUFFER} != expected:
        raise AdapterContractError("MLX_PARAMETER_NAMES_CHANGED")
    for name, shape in shapes.items():
        value = metadata[name]
        if tuple(value["shape"]) != shape or value["dtype"] != "float32":
            raise AdapterContractError("MLX_PARAMETER_SHAPE_OR_DTYPE_CHANGED")
    if has_legacy:
        value = metadata[LEGACY_BUFFER]
        if (tuple(value["shape"]) != (1, 514) or value["dtype"] != "int64"
                or legacy_values != [list(range(514))]):
            raise AdapterContractError("MLX_LEGACY_POSITION_BUFFER_CHANGED")
    return {"learning_parameter_tensors": len(shapes),
            "learning_parameter_elements": sum(math.prod(s) for s in shapes.values()),
            "legacy_buffer_verified_and_ignored": has_legacy,
            "missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}


def _asset(path: Path, expected: dict) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if (path.stat().st_size != expected["bytes"]
            or digest.hexdigest() != expected["sha256"]):
        raise AdapterContractError("MLX_MODEL_ASSET_HASH_CHANGED")


def _array_reference(array) -> dict:
    import numpy as np

    if (not isinstance(array, np.ndarray) or array.ndim != 2
            or array.dtype != np.dtype("<i8") or not array.flags.c_contiguous):
        raise AdapterContractError("MLX_INPUT_DTYPE_OR_LAYOUT_CHANGED")
    return {"dtype": "int64", "byte_order": "little", "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _parameter_readback_reference(array) -> dict:
    """只在初始化时散列实际量化数组，保留原dtype且不强制转换数值。"""
    import numpy as np

    value = np.ascontiguousarray(np.asarray(array))
    if value.dtype not in (np.dtype("<u4"), np.dtype("<f4")):
        raise AdapterContractError("MLX_QUANTIZED_READBACK_DTYPE_CHANGED")
    if value.dtype == np.float32 and not np.isfinite(value).all():
        raise AdapterContractError("MLX_QUANTIZED_READBACK_NONFINITE")
    return {"shape": list(value.shape), "dtype": value.dtype.name,
            "byte_order": "little", "bytes": value.nbytes,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def _metadata_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _gpu_stream(mx):
    """新建GPU流；统一内存数组没有PyTorch式驻留设备承诺。"""
    if not mx.metal.is_available():
        raise AdapterContractError("MLX_METAL_UNAVAILABLE")
    mx.set_default_device(mx.gpu)
    stream = mx.new_stream(mx.gpu)
    if stream.device.type != mx.gpu or mx.default_device().type != mx.gpu:
        raise AdapterContractError("MLX_GPU_STREAM_CHANGED")
    return stream


class MlxRerankerSession:
    """兼容原session.run的GPU会话，W8A32只压缩74个Linear权重。

    浮点激活、Embedding、LayerNorm和全部bias保持FP32。量化矩阵使用
    uint32打包存储，执行调用MLX quantized_matmul；不声称纯整数全图。
    """

    def __init__(self, spec: dict, floating: dict, variant: str = "mlx_fp32"):
        if variant not in VARIANTS:
            raise AdapterContractError("MLX_VARIANT_CHANGED")
        if (spec.get("repo") != MODEL_ID or floating.get("repo") != MODEL_ID
                or spec.get("revision") != MODEL_REVISION
                or floating.get("revision") != MODEL_REVISION
                or floating.get("weights") != "model.safetensors"
                or floating["files"]["config.json"]["sha256"]
                != spec["files"]["config.json"]["sha256"]):
            raise AdapterContractError("MLX_MODEL_IDENTITY_CHANGED")
        directory = Path(floating["directory"])
        for name in ("config.json", "model.safetensors"):
            _asset(directory / name, floating["files"][name])
        config = json.loads((directory / "config.json").read_text())
        validate_config(config)
        self.mx = importlib.import_module("mlx.core")
        self.variant = variant
        self.gpu_stream = _gpu_stream(self.mx)
        self.batch_proofs = []
        self._linear_call_count = self._quantized_call_count = 0
        self._called_linear_names = []
        with self.mx.stream(self.gpu_stream):
            self.weights = self.mx.load(str(directory / "model.safetensors"))
            metadata = {name: {"shape": list(value.shape),
                "dtype": "float32" if value.dtype == self.mx.float32
                else "int64" if value.dtype == self.mx.int64 else str(value.dtype)}
                for name, value in self.weights.items()}
            legacy = self.weights.get(LEGACY_BUFFER)
            legacy_values = None
            if legacy is not None:
                self.mx.eval(legacy)
                legacy_values = legacy.tolist()
            loading = validate_parameter_metadata(metadata, legacy_values)
            self.weights.pop(LEGACY_BUFFER, None)
            # 核实际已加载参数而非仅信模型文件SHA；归约完成后再量化。
            finite_checks = [self.mx.isfinite(value).all() for value in self.weights.values()]
            self.mx.eval(*finite_checks)
            if not all(value.item() for value in finite_checks):
                raise AdapterContractError("MLX_LOADED_PARAMETER_NONFINITE")
            loading["finite_learning_parameter_tensors_verified"] = len(finite_checks)
            self.quantized = {}
            if variant == "mlx_w8a32":
                for name in linear_shapes():
                    weight = self.weights.pop(name + ".weight")
                    packed, scales, biases = self.mx.quantize(
                        weight, group_size=GROUP_SIZE, bits=BITS, mode="affine")
                    self.quantized[name] = (packed, scales, biases)
                self.mx.eval(*[value for values in self.quantized.values() for value in values])
            self.mx.eval(*self.weights.values())
        self.mx.synchronize(self.gpu_stream)
        self._quantization_metadata()
        quantized_parameters = [{"name": name,
            "weight": _parameter_readback_reference(values[0]),
            "scales": _parameter_readback_reference(values[1]),
            "quantization_biases": _parameter_readback_reference(values[2])}
            for name, values in sorted(self.quantized.items())]
        self.quantized_parameters_sha256 = _metadata_digest(quantized_parameters)
        shapes = linear_shapes()
        self.identity = {
            "backend": "mlx", "variant": variant, "device": "gpu",
            "dtype": "float32", "activation_dtype": "float32",
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "weights_sha256": floating["files"]["model.safetensors"]["sha256"],
            "tokenizer_sha256": spec["files"]["tokenizer.json"]["sha256"],
            "quantization": "affine_w8a32" if self.quantized else "none",
            "bits": BITS if self.quantized else None,
            "group_size": GROUP_SIZE if self.quantized else None,
            "quantized_linear_count": len(self.quantized),
            "quantized_weight_elements": sum(math.prod(s) for s in shapes.values()) if self.quantized else 0,
            "linear_layer_count": len(shapes),
            "non_linear_parameter_dtype": "float32", "bias_dtype": "float32",
            "weight_packing": "uint32" if self.quantized else None,
            "quantized_parameters": quantized_parameters,
            "quantized_parameters_sha256": self.quantized_parameters_sha256,
            "loading_info": loading, "input_names": ["input_ids", "attention_mask"],
            "gpu_stream": str(self.gpu_stream),
            "gpu_proof_limit": "Explicit MLX GPU stream and synchronized evaluated quantized_matmul path; not a complete Metal kernel trace.",
        }

    def get_inputs(self):
        """保留原重排器对两个int64输入名称的探测。"""
        return [SimpleNamespace(name=name, type="tensor(int64)")
                for name in ("input_ids", "attention_mask")]

    def _quantization_metadata(self) -> list[dict]:
        if not self.quantized:
            if self.variant != "mlx_fp32":
                raise AdapterContractError("MLX_QUANTIZED_LAYERS_MISSING")
            return []
        shapes = linear_shapes()
        if self.variant != "mlx_w8a32" or set(self.quantized) != set(shapes):
            raise AdapterContractError("MLX_QUANTIZED_LAYER_SET_CHANGED")
        records = []
        for name, (out_dim, in_dim) in shapes.items():
            packed, scales, biases = self.quantized[name]
            if (packed.dtype != self.mx.uint32 or tuple(packed.shape) != (out_dim, in_dim // 4)
                    or scales.dtype != self.mx.float32 or biases.dtype != self.mx.float32
                    or tuple(scales.shape) != (out_dim, in_dim // GROUP_SIZE)
                    or tuple(biases.shape) != tuple(scales.shape)):
                raise AdapterContractError("MLX_QUANTIZED_STORAGE_CHANGED")
            records.append({"name": name, "original_shape": [out_dim, in_dim],
                "packed_shape": list(packed.shape), "packed_dtype": "uint32",
                "scales_shape": list(scales.shape), "scales_dtype": "float32",
                "biases_shape": list(biases.shape), "biases_dtype": "float32",
                "bits": BITS, "group_size": GROUP_SIZE, "mode": "affine"})
        return records

    def _linear(self, x, name):
        self._linear_call_count += 1
        self._called_linear_names.append(name)
        if name in self.quantized:
            self._quantized_call_count += 1
            packed, scales, biases = self.quantized[name]
            value = self.mx.quantized_matmul(x, packed, scales=scales,
                biases=biases, transpose=True, group_size=GROUP_SIZE, bits=BITS,
                mode="affine", stream=self.gpu_stream)
        else:
            value = self.mx.matmul(x, self.weights[name + ".weight"].T)
        return value + self.weights[name + ".bias"]

    def _norm(self, x, name):
        return self.mx.fast.layer_norm(x, self.weights[name + ".weight"],
            self.weights[name + ".bias"], 1e-5)

    def _forward(self, ids, attention_mask):
        mx = self.mx
        non_padding = (ids != 1).astype(mx.int32)
        positions = mx.cumsum(non_padding, axis=1) * non_padding + 1
        prefix = "roberta.embeddings"
        x = self.weights[prefix + ".word_embeddings.weight"][ids]
        x = x + self.weights[prefix + ".token_type_embeddings.weight"][mx.zeros_like(ids)]
        x = x + self.weights[prefix + ".position_embeddings.weight"][positions]
        x = self._norm(x, prefix + ".LayerNorm")
        # 与HF eager保持双向key padding mask，禁用dropout/causal捷径。
        mask = mx.where(attention_mask[:, None, None, :] != 0,
                        mx.array(0, dtype=mx.float32),
                        mx.array(-3.4028234663852886e38, dtype=mx.float32))
        batch, length = ids.shape
        for i in range(12):
            p = f"roberta.encoder.layer.{i}"
            q, k, v = [self._linear(x, p + ".attention.self." + name)
                .reshape(batch, length, 12, 32).transpose(0, 2, 1, 3)
                for name in ("query", "key", "value")]
            scores = mx.matmul(q, k.swapaxes(-1, -2)) * (32 ** -0.5) + mask
            probabilities = mx.softmax(scores, axis=-1)
            context = mx.matmul(probabilities, v).transpose(0, 2, 1, 3)
            context = context.reshape(batch, length, 384)
            x = self._norm(self._linear(context, p + ".attention.output.dense") + x,
                           p + ".attention.output.LayerNorm")
            hidden = self._linear(x, p + ".intermediate.dense")
            hidden = hidden * (1 + mx.erf(hidden / math.sqrt(2))) / 2
            x = self._norm(self._linear(hidden, p + ".output.dense") + x,
                           p + ".output.LayerNorm")
        head = mx.tanh(self._linear(x[:, 0, :], "classifier.dense"))
        return self._linear(head, "classifier.out_proj")

    def run(self, output_names, feed):
        """每批核输入回读、74个真实算子路径及GPU完成后才返回logit。"""
        import numpy as np

        # 必须在输入检查前清零；否则拒绝输入时会沿用上一批的计数。
        self._linear_call_count = self._quantized_call_count = 0
        self._called_linear_names = []
        proof = {"input_tensor_references": {}, "transferred_tensor_references": {},
            "gpu_stream_verified": False, "gpu_eval_complete": False,
            "sync_before": False, "sync_after": False,
            "quantization_verified": False, "output_dtype": None,
            "activation_dtype": "float32", "quantized_linear_count": len(self.quantized),
            "linear_call_count": 0, "quantized_matmul_calls": 0,
            "linear_call_names": [],
            "bits": BITS if self.quantized else None,
            "group_size": GROUP_SIZE if self.quantized else None,
            "weight_packing": "uint32" if self.quantized else None,
            "quantized_parameters_sha256": self.quantized_parameters_sha256,
            "quantized_layer_names": sorted(self.quantized), "error": None}
        self.batch_proofs.append(proof)
        try:
            if output_names is not None or set(feed) != {"input_ids", "attention_mask"}:
                raise AdapterContractError("MLX_INPUT_NAMES_CHANGED")
            for name, value in feed.items():
                proof["input_tensor_references"][name] = _array_reference(value)
            ids, mask = feed["input_ids"], feed["attention_mask"]
            if (ids.shape != mask.shape or not 1 <= ids.shape[0] <= 4
                    or not 1 <= ids.shape[1] <= 512
                    or np.any(ids < 0) or np.any(ids >= 250002)
                    or not np.isin(mask, (0, 1)).all()):
                raise AdapterContractError("MLX_INPUT_VALUES_OR_SHAPE_CHANGED")
            if (self.gpu_stream.device.type != self.mx.gpu
                    or self.mx.default_device().type != self.mx.gpu):
                raise AdapterContractError("MLX_GPU_STREAM_CHANGED")
            proof["gpu_stream_verified"] = True
            self._quantization_metadata()
            proof["quantization_verified"] = True
            self._linear_call_count = self._quantized_call_count = 0
            self._called_linear_names = []
            with self.mx.stream(self.gpu_stream):
                tensors = {name: self.mx.array(value, dtype=self.mx.int64)
                           for name, value in feed.items()}
                self.mx.eval(*tensors.values())
                for name, value in tensors.items():
                    actual = _array_reference(np.ascontiguousarray(np.asarray(value)))
                    proof["transferred_tensor_references"][name] = actual
                    if actual != proof["input_tensor_references"][name]:
                        raise AdapterContractError("MLX_TRANSFERRED_INPUT_CHANGED")
                self.mx.synchronize(self.gpu_stream)
                proof["sync_before"] = True
                output = self._forward(tensors["input_ids"], tensors["attention_mask"])
                self.mx.eval(output)
                self.mx.synchronize(self.gpu_stream)
                proof["sync_after"] = proof["gpu_eval_complete"] = True
                proof["linear_call_count"] = self._linear_call_count
                proof["quantized_matmul_calls"] = self._quantized_call_count
                proof["linear_call_names"] = list(self._called_linear_names)
                if (self._linear_call_count != 74
                        or set(self._called_linear_names) != set(linear_shapes())
                        or self._quantized_call_count != (74 if self.quantized else 0)
                        or output.dtype != self.mx.float32):
                    raise AdapterContractError("MLX_EXECUTED_PATH_CHANGED")
                proof["output_dtype"] = "float32"
                result = np.asarray(output)
            if result.shape != (len(ids), 1) or result.dtype != np.float32 or not np.isfinite(result).all():
                raise AdapterContractError("MLX_OUTPUT_INVALID")
            proof["output_shape"] = list(result.shape)
            return [np.ascontiguousarray(result)]
        except BaseException as exc:
            proof["error"] = {"type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            # 记录已进入的算子调用，即使eval失败也不伪装为0；完成性另看GPU标记。
            proof["linear_call_count"] = self._linear_call_count
            proof["quantized_matmul_calls"] = self._quantized_call_count
            proof["linear_call_names"] = list(self._called_linear_names)
