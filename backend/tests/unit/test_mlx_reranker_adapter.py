"""用合成元数据与替身核MLX边界，不导入MLX或执行学习模型。"""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import mlx_reranker_adapter as adapter


def config():
    return {**adapter.FIXED_CONFIG,
        "architectures": ["XLMRobertaForSequenceClassification"],
        "id2label": {"0": "LABEL_0"}}


def metadata():
    return {k: {"shape": list(v), "dtype": "float32"}
            for k, v in adapter.parameter_shapes().items()}


def feed():
    return {"input_ids": np.array([[0, 5, 2, 1]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int64)}


class TensorShape:
    """仅保存shape/dtype，避免创建任何真实大权重。"""

    def __init__(self, shape, dtype=np.float32):
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)

    def __array__(self, dtype=None, copy=None):
        return np.zeros(self.shape, dtype=dtype or self.dtype)


class FakeMx:
    gpu = "gpu"
    float32 = np.dtype("float32")
    int64 = np.dtype("int64")
    uint32 = np.dtype("uint32")

    def __init__(self):
        self.device_type = self.gpu
        self.stream_value = SimpleNamespace(device=SimpleNamespace(type=self.gpu))
        self.metal = SimpleNamespace(is_available=lambda: True)
        self.events = []
        self.corrupt_transfer = False
        self.quantize_calls = []
        self.quantized_calls = []

    def set_default_device(self, device):
        self.device_type = device

    def default_device(self):
        return SimpleNamespace(type=self.device_type)

    def new_stream(self, device):
        return self.stream_value

    def stream(self, stream):
        return nullcontext()

    def eval(self, *arrays):
        self.events.append("eval")

    def synchronize(self, stream):
        self.events.append("synchronize")

    def isfinite(self, value):
        return np.array(True)

    def array(self, value, dtype):
        result = np.array(value, dtype=dtype)
        if self.corrupt_transfer:
            result[0, 0] += 1
        return result

    def load(self, path):
        return {name: TensorShape(shape) for name, shape in adapter.parameter_shapes().items()}

    def quantize(self, weight, *, group_size, bits, mode):
        self.quantize_calls.append((group_size, bits, mode))
        out_dim, in_dim = weight.shape
        return (TensorShape((out_dim, in_dim // 4), np.uint32),
                TensorShape((out_dim, in_dim // group_size)),
                TensorShape((out_dim, in_dim // group_size)))

    def quantized_matmul(self, x, weight, **kwargs):
        self.quantized_calls.append(kwargs)
        return np.zeros((*x.shape[:-1], weight.shape[0]), dtype=np.float32)


def fake_session(variant="mlx_fp32"):
    value = adapter.MlxRerankerSession.__new__(adapter.MlxRerankerSession)
    value.mx, value.variant = FakeMx(), variant
    value.gpu_stream = value.mx.stream_value
    value.batch_proofs = []
    value.quantized_parameters_sha256 = "f" * 64
    value.quantized, value.weights = {}, {}
    if variant == "mlx_w8a32":
        for name, (out_dim, in_dim) in adapter.linear_shapes().items():
            value.quantized[name] = (TensorShape((out_dim, in_dim // 4), np.uint32),
                TensorShape((out_dim, in_dim // 64)), TensorShape((out_dim, in_dim // 64)))

    def score(ids, mask):
        # 仅模拟运行回执，不执行编码器或任何学习模型。
        value._linear_call_count = 74
        value._quantized_call_count = 74 if value.quantized else 0
        value._called_linear_names = list(adapter.linear_shapes())
        return np.zeros((len(ids), 1), dtype=np.float32)

    value._forward = score
    return value


class ModelMetadataTests(unittest.TestCase):
    def test_exact_config_and_201_parameters(self):
        adapter.validate_config(config())
        result = adapter.validate_parameter_metadata(metadata())
        self.assertEqual(result["learning_parameter_tensors"], 201)
        self.assertEqual(result["learning_parameter_elements"], 117641089)
        shapes = adapter.linear_shapes()
        self.assertEqual(len(shapes), 74)
        self.assertIn("classifier.out_proj", shapes)
        self.assertTrue(all(in_dim % 64 == 0 for _, in_dim in shapes.values()))
        self.assertFalse(any("embeddings" in name for name in shapes))

    def test_different_activation_head_or_decoder_rejected(self):
        for key, value in (("hidden_act", "gelu_new"), ("hidden_size", 768),
                           ("id2label", {"0": "A", "1": "B"}),
                           ("architectures", ["XLMRobertaModel"]), ("is_decoder", True)):
            changed = config(); changed[key] = value
            with self.subTest(key=key), self.assertRaises(adapter.AdapterContractError):
                adapter.validate_config(changed)

    def test_missing_extra_shape_or_precision_rejected(self):
        for kind in ("missing", "extra", "shape", "precision"):
            value = metadata(); key = "classifier.out_proj.weight"
            if kind == "missing": value.pop(key)
            elif kind == "extra": value["roberta.pooler.dense.weight"] = value[key]
            elif kind == "shape": value[key]["shape"] = [2, 384]
            else: value[key]["dtype"] = "float16"
            with self.subTest(kind=kind), self.assertRaises(adapter.AdapterContractError):
                adapter.validate_parameter_metadata(value)

    def test_only_exact_legacy_buffer_may_be_ignored(self):
        value = metadata();value[adapter.LEGACY_BUFFER] = {"shape": [1, 514], "dtype": "int64"}
        self.assertTrue(adapter.validate_parameter_metadata(value, [list(range(514))])[
            "legacy_buffer_verified_and_ignored"])
        for legacy in (None, [list(range(513))], [[0] * 514]):
            with self.assertRaises(adapter.AdapterContractError):
                adapter.validate_parameter_metadata(value, legacy)

    def test_constructor_rejects_unknown_variant_before_framework_import(self):
        with patch.object(adapter.importlib, "import_module") as imported:
            with self.assertRaises(adapter.AdapterContractError):
                adapter.MlxRerankerSession({}, {}, "mlx_fp16")
        imported.assert_not_called()

    def test_constructor_rejects_other_model_before_framework_import(self):
        with patch.object(adapter.importlib, "import_module") as imported:
            with self.assertRaises(adapter.AdapterContractError):
                adapter.MlxRerankerSession({"repo": "other"}, {})
        imported.assert_not_called()

    def test_synthetic_constructor_quantizes_only_74_linear_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); (p / "config.json").write_text(json.dumps(config()))
            spec = {"repo": adapter.MODEL_ID, "revision": adapter.MODEL_REVISION,
                "files": {"config.json": {"sha256": "c"}, "tokenizer.json": {"sha256": "t"}}}
            floating = {"repo": adapter.MODEL_ID, "revision": adapter.MODEL_REVISION,
                "weights": "model.safetensors", "directory": folder,
                "files": {"config.json": {"sha256": "c"}, "model.safetensors": {"sha256": "w"}}}
            mx = FakeMx()
            with patch.object(adapter, "_asset") as checked, patch.object(
                    adapter.importlib, "import_module", return_value=mx) as imported:
                session = adapter.MlxRerankerSession(spec, floating, "mlx_w8a32")
            imported.assert_called_once_with("mlx.core")
            self.assertEqual(checked.call_count, 2)
            self.assertEqual(len(mx.quantize_calls), 74)
            self.assertEqual(set(mx.quantize_calls), {(64, 8, "affine")})
            self.assertEqual(session.identity["quantized_linear_count"], 74)
            self.assertEqual(session.identity["loading_info"]["learning_parameter_tensors"], 201)
            self.assertEqual(session.identity["loading_info"]["finite_learning_parameter_tensors_verified"], 201)
            identities = session.identity["quantized_parameters"]
            self.assertEqual(len(identities), 74)
            self.assertEqual(session.identity["quantized_parameters_sha256"], adapter._metadata_digest(identities))
            first = identities[0]
            self.assertEqual(first["weight"]["dtype"], "uint32")
            self.assertEqual(first["scales"]["dtype"], "float32")
            self.assertEqual(len(first["weight"]["sha256"]), 64)
            self.assertIn("roberta.embeddings.word_embeddings.weight", session.weights)
            self.assertIn("classifier.out_proj.bias", session.weights)
            self.assertNotIn("classifier.out_proj.weight", session.weights)


class SessionBoundaryTests(unittest.TestCase):
    def test_fp32_round_trip_completed_before_return(self):
        session = fake_session(); result = session.run(None, feed())[0]
        self.assertEqual(result.shape, (1, 1));self.assertEqual(result.dtype, np.float32)
        proof = session.batch_proofs[-1]
        self.assertEqual(proof["input_tensor_references"], proof["transferred_tensor_references"])
        for key in ("gpu_stream_verified", "gpu_eval_complete", "sync_before", "sync_after",
                    "quantization_verified"):
            self.assertIs(proof[key], True)
        self.assertEqual(proof["linear_call_count"], 74)
        self.assertEqual(proof["quantized_matmul_calls"], 0)
        self.assertEqual(session.mx.events[-1], "synchronize")

    def test_w8_proves_packed_storage_and_all_calls(self):
        session = fake_session("mlx_w8a32");session.run(None, feed())
        proof = session.batch_proofs[-1]
        self.assertEqual(proof["quantized_linear_count"], 74)
        self.assertEqual(proof["quantized_matmul_calls"], 74)
        self.assertEqual(len(set(proof["linear_call_names"])), 74)
        self.assertEqual((proof["bits"], proof["group_size"], proof["weight_packing"]), (8, 64, "uint32"))

    def test_actual_quantized_call_uses_explicit_gpu_stream_and_bias(self):
        session = fake_session("mlx_w8a32")
        session._linear_call_count = session._quantized_call_count = 0
        session._called_linear_names = []
        session.weights["classifier.out_proj.bias"] = np.array([2], dtype=np.float32)
        result = session._linear(np.zeros((1, 384), dtype=np.float32), "classifier.out_proj")
        np.testing.assert_equal(result, [[2]])
        call = session.mx.quantized_calls[0]
        self.assertEqual((call["bits"], call["group_size"], call["mode"]), (8, 64, "affine"))
        self.assertIs(call["stream"], session.gpu_stream)

    def test_changed_input_dtype_layout_names_and_values_rejected(self):
        for kind in ("dtype", "layout", "names", "mask", "token", "shape", "length", "batch"):
            value = feed()
            if kind == "dtype": value["input_ids"] = value["input_ids"].astype(np.int32)
            elif kind == "layout": value["input_ids"] = value["input_ids"][:, ::-1]
            elif kind == "names": value["token_type_ids"] = value["input_ids"]
            elif kind == "mask": value["attention_mask"][0, 0] = 2
            elif kind == "token": value["input_ids"][0, 0] = 250002
            elif kind == "shape": value["attention_mask"] = value["attention_mask"][:, :2].copy()
            elif kind == "length": value = {k: np.ones((1, 513), dtype=np.int64) for k in value}
            else: value = {k: np.ones((5, 2), dtype=np.int64) for k in value}
            session = fake_session()
            with self.subTest(kind=kind), self.assertRaises(adapter.AdapterContractError):
                session.run(None, value)
            self.assertFalse(session.batch_proofs[-1]["gpu_eval_complete"])

    def test_transfer_mutation_rejected(self):
        session = fake_session();session.mx.corrupt_transfer = True
        with self.assertRaisesRegex(adapter.AdapterContractError, "TRANSFERRED"):
            session.run(None, feed())
        self.assertFalse(session.batch_proofs[-1]["sync_before"])

    def test_gpu_stream_switch_rejected(self):
        session = fake_session();session.gpu_stream.device.type = "cpu"
        with self.assertRaisesRegex(adapter.AdapterContractError, "STREAM"):
            session.run(None, feed())

    def test_quantized_layer_set_or_storage_changes_rejected(self):
        for kind in ("missing", "dtype", "shape"):
            session = fake_session("mlx_w8a32"); name = next(iter(session.quantized))
            if kind == "missing": session.quantized.pop(name)
            else:
                packed, scale, bias = session.quantized[name]
                if kind == "dtype": packed = TensorShape(packed.shape, np.int8)
                else: scale = TensorShape((1, 1))
                session.quantized[name] = packed, scale, bias
            with self.subTest(kind=kind), self.assertRaises(adapter.AdapterContractError):
                session.run(None, feed())

    def test_nonfinite_wrong_shape_or_precision_outputs_rejected(self):
        for value in (np.array([[np.nan]], np.float32), np.zeros((1, 2), np.float32),
                      np.zeros((1, 1), np.float16)):
            session = fake_session();original = session._forward
            def score(ids, mask):
                original(ids, mask)
                return value
            session._forward = score
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(adapter.AdapterContractError):
                session.run(None, feed())
            self.assertIsNotNone(session.batch_proofs[-1]["error"])

    def test_missing_or_duplicate_executed_linear_path_rejected(self):
        session = fake_session();original = session._forward
        def score(ids, mask):
            output = original(ids, mask)
            session._called_linear_names[-1] = session._called_linear_names[0]
            return output
        session._forward = score
        with self.assertRaisesRegex(adapter.AdapterContractError, "PATH"):
            session.run(None, feed())

    def test_partial_forward_failure_keeps_actual_attempt_counts(self):
        session = fake_session("mlx_w8a32")
        def stopped(ids, mask):
            session._linear_call_count = session._quantized_call_count = 7
            session._called_linear_names = list(adapter.linear_shapes())[:7]
            raise RuntimeError("synthetic interruption")
        session._forward = stopped
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            session.run(None, feed())
        proof = session.batch_proofs[-1]
        self.assertEqual((proof["linear_call_count"], proof["quantized_matmul_calls"]), (7, 7))
        self.assertEqual(len(proof["linear_call_names"]), 7)
        self.assertFalse(proof["gpu_eval_complete"])

    def test_eval_failure_after_forward_keeps_counts(self):
        session = fake_session("mlx_w8a32");original = session.mx.eval
        def interrupted(*values):
            original(*values)
            if len(session.mx.events) > 2:
                raise RuntimeError("synthetic eval failure")
        session.mx.eval = interrupted
        with self.assertRaisesRegex(RuntimeError, "eval failure"):
            session.run(None, feed())
        proof = session.batch_proofs[-1]
        self.assertEqual((proof["linear_call_count"], proof["quantized_matmul_calls"]), (74, 74))
        self.assertFalse(proof["sync_after"])

    def test_rejected_next_feed_does_not_reuse_previous_counts(self):
        session = fake_session("mlx_w8a32");session.run(None, feed())
        bad = feed();bad["input_ids"] = bad["input_ids"].astype(np.int32)
        with self.assertRaises(adapter.AdapterContractError):
            session.run(None, bad)
        proof = session.batch_proofs[-1]
        self.assertEqual((proof["linear_call_count"], proof["quantized_matmul_calls"]), (0, 0))

    def test_parameter_hash_binds_bytes_shape_and_dtype(self):
        original = np.array([[1, 2]], dtype=np.uint32)
        changed = np.array([[1, 3]], dtype=np.uint32)
        first = adapter._parameter_readback_reference(original)
        second = adapter._parameter_readback_reference(changed)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual((first["shape"], first["dtype"], first["bytes"]), ([1, 2], "uint32", 8))
        with self.assertRaises(adapter.AdapterContractError):
            adapter._parameter_readback_reference(np.array([[np.nan]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
