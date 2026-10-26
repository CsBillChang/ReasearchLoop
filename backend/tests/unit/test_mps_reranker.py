"""以合成张量和本机假文件验证 MPS provider，不加载真实模型。"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research_loop.providers import mps_reranker as provider


class Tensor:
    def __init__(self, value, dtype="torch.int64", device="mps:0"):
        self.value, self.dtype, self.device = np.array(value), dtype, device

    def detach(self): return self
    def cpu(self): return Tensor(self.value, self.dtype, "cpu")
    def float(self): return Tensor(self.value.astype(np.float32), "torch.float32", self.device)
    def numpy(self): return self.value


class Parameter:
    def __init__(self, dtype="torch.float16", device="mps:0"):
        self.dtype, self.device = dtype, device

    def is_floating_point(self): return True


class Torch(ModuleType):
    int64, float16 = "torch.int64", "torch.float16"

    def __init__(self):
        super().__init__("torch")
        self.syncs, self.intra, self.inter, self.inter_sets = 0, 4, 1, 0
        self.available, self.corrupt, self.transfer_device = True, False, "mps:0"
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: self.available))
        self.mps = SimpleNamespace(synchronize=self.synchronize)

    def synchronize(self): self.syncs += 1
    def inference_mode(self): return nullcontext()
    def get_num_threads(self): return self.intra
    def get_num_interop_threads(self): return self.inter
    def set_num_threads(self, value): self.intra = value

    def set_num_interop_threads(self, value):
        if self.inter_sets:
            raise RuntimeError("synthetic one-time setting")
        self.inter, self.inter_sets = value, self.inter_sets + 1

    def as_tensor(self, value, *, dtype, device):
        return Tensor(value + 1 if self.corrupt else value, dtype, self.transfer_device)


class XLMRobertaForSequenceClassification:
    def __init__(self):
        self.training, self.parameter, self.calls = False, Parameter(), 0
        self.output_device, self.output_dtype, self.output_shape, self.nan = "mps:0", "torch.float16", None, False
        self.config = SimpleNamespace(model_type="xlm-roberta", num_labels=1, hidden_size=384,
                                      num_hidden_layers=12)

    def parameters(self): return iter([self.parameter])
    def modules(self): return iter([self])
    def to(self, device): self.parameter.device = device; return self
    def eval(self): self.training = False; return self

    def __call__(self, **feed):
        self.calls += 1
        # 第一 token 携带合成片段身份，便于发现组批或还原顺序错误。
        values = feed["input_ids"].value[:, :1].astype(np.float32)
        if self.nan: values[:] = np.nan
        if self.output_shape: values = np.zeros(self.output_shape)
        return SimpleNamespace(logits=Tensor(values, self.output_dtype, self.output_device))


class Tokenizer:
    def __init__(self):
        self.encoded, self.no_truncation_calls, self.no_padding_calls = [], 0, 0

    def no_truncation(self): self.no_truncation_calls += 1
    def no_padding(self): self.no_padding_calls += 1

    def encode(self, query, passage):
        self.encoded.append((query, passage))
        index = int(passage.rsplit(" ", 1)[1])
        return SimpleNamespace(ids=[index + 10] * (index + 2))


class MpsRerankerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original, self.floating = self.root / "original", self.root / "floating"
        self.original.mkdir(); self.floating.mkdir()
        for folder, names in ((self.original, ("config.json", "tokenizer.json")),
                              (self.floating, ("config.json", "model.safetensors"))):
            for name in names: (folder / name).write_text("synthetic " + name)
        self.manifest = {"reranker": self.spec(self.original), "reranker_fp16": self.spec(self.floating)}
        self.path = self.root / "manifest.json"
        self.write_manifest()
        self.torch, self.model, self.tokenizer = Torch(), XLMRobertaForSequenceClassification(), Tokenizer()
        self.loading = {"missing_keys": set(), "unexpected_keys": set(),
                        "mismatched_keys": set(), "error_msgs": []}
        self.events, self.loader_args = [], []
        transformers, utils, tokenizers = ModuleType("transformers"), ModuleType("transformers.utils"), ModuleType("tokenizers")
        transformers.__path__ = []
        utils.logging = SimpleNamespace(disable_progress_bar=lambda: self.events.append("progress_disabled"))
        transformers.utils = utils

        def load(*args, **kwargs):
            self.events.append("load")
            self.loader_args.append((args, kwargs))
            return self.model, self.loading

        transformers.AutoModelForSequenceClassification = SimpleNamespace(from_pretrained=load)
        tokenizers.Tokenizer = SimpleNamespace(from_file=lambda path: self.tokenizer)
        self.modules = {"torch": self.torch, "tokenizers": tokenizers,
                        "transformers": transformers, "transformers.utils": utils}

    def spec(self, folder):
        return {"repo": provider.MODEL_ID, "revision": provider.MODEL_REVISION,
                "directory": str(folder), "weights": "model.safetensors",
                "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in folder.iterdir()}}

    def write_manifest(self): self.path.write_text(json.dumps(self.manifest))

    def load(self):
        engine = provider.MpsReranker(self.path)
        with patch.dict(sys.modules, self.modules), patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}), \
                patch.object(provider.importlib.metadata, "version", return_value="synthetic"):
            engine._load()
        return engine

    def feed(self, count=2):
        ids = np.arange(count * 3, dtype=np.int64).reshape(count, 3)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    def test_construct_and_empty_score_do_not_load_torch_or_files(self):
        with patch.object(provider.MpsReranker, "_load", side_effect=AssertionError("must stay lazy")):
            engine = provider.MpsReranker("does-not-exist")
            self.assertEqual(engine.score("original query", []), [])
        self.assertIsNone(engine.session)
        self.assertEqual((engine.threads, engine.batch_size, engine.sort_by_length), (4, 4, False))

    def test_load_is_local_safetensors_eager_fp16_and_keeps_identity(self):
        engine = self.load()
        self.assertEqual(self.events, ["progress_disabled", "load"])
        args, kwargs = self.loader_args[0]
        self.assertEqual(args, (str(self.floating),))
        self.assertEqual(kwargs, {"local_files_only": True, "trust_remote_code": False,
            "use_safetensors": True, "attn_implementation": "eager", "dtype": "torch.float16",
            "output_loading_info": True})
        self.assertEqual(engine.identity["revision"], provider.MODEL_REVISION)
        self.assertEqual(engine.identity["weights_sha256"], self.manifest["reranker_fp16"]["files"]["model.safetensors"]["sha256"])
        self.assertFalse(engine.identity["cpu_fallback_enabled"])
        self.assertEqual((self.tokenizer.no_truncation_calls, self.tokenizer.no_padding_calls), (1, 1))

    def test_hash_mismatch_stops_before_model_loading(self):
        (self.original / "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"): self.load()
        self.assertEqual(self.loader_args, [])

    def test_wrong_revision_or_weights_file_is_rejected(self):
        original = deepcopy(self.manifest)
        for section, field, value in (("reranker", "revision", "other"),
                                      ("reranker_fp16", "revision", "other"),
                                      ("reranker_fp16", "weights", "../unsafe.bin")):
            self.manifest = deepcopy(original); self.manifest[section][field] = value; self.write_manifest()
            with self.subTest(section=section, field=field), self.assertRaisesRegex(ValueError, "CONFIG_UNSUPPORTED"):
                self.load()
        self.assertEqual(self.loader_args, [])

    def test_missing_manifest_and_missing_files_have_stable_errors(self):
        self.manifest.pop("reranker_fp16"); self.write_manifest()
        with self.assertRaisesRegex(ValueError, "MANIFEST_INVALID"): self.load()
        self.manifest["reranker_fp16"] = self.spec(self.floating); self.write_manifest()
        (self.floating / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "FILES_UNAVAILABLE"): self.load()

    def test_mps_unavailable_never_loads_or_runs_model(self):
        self.torch.available = False
        with self.assertRaisesRegex(ValueError, "MPS_UNAVAILABLE"): self.load()
        self.assertEqual(self.loader_args, [])
        self.assertEqual(self.model.calls, 0)

    def test_explicit_cpu_fallback_is_rejected(self):
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(ValueError, "FALLBACK_FORBIDDEN"):
                provider.MpsReranker(self.path)._load()
        self.assertEqual(self.loader_args, [])

    def test_repeated_instances_do_not_reset_interop_threads(self):
        self.torch.intra, self.torch.inter = 2, 4
        self.load(); self.load()
        self.assertEqual((self.torch.intra, self.torch.inter, self.torch.inter_sets), (4, 1, 1))

    def test_busy_torch_thread_configuration_is_explicit_failure(self):
        self.torch.inter, self.torch.inter_sets = 4, 1
        with self.assertRaisesRegex(ValueError, "THREAD_CONFIG_UNAVAILABLE"): self.load()
        self.assertEqual(self.loader_args, [])

    def test_missing_or_unexpected_parameters_are_rejected(self):
        for key, values in (("missing_keys", {"classifier.weight"}),
                            ("unexpected_keys", {"extra.weight"}),
                            ("mismatched_keys", {("classifier.weight", (1, 384), (2, 384))}),
                            ("error_msgs", ["failed"])):
            info = {name: [] for name in self.loading}; info[key] = values
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "LOADING_KEYS_CHANGED"):
                provider._loading_report(info, self.floating / "model.safetensors")

    def test_missing_report_field_or_duplicate_buffer_is_rejected(self):
        for info in ({"unexpected_keys": []}, {**self.loading, "unknown": []},
                     {**self.loading, "unexpected_keys": ["roberta.embeddings.position_ids"] * 2}):
            with self.assertRaisesRegex(ValueError, "LOADING_KEYS_CHANGED"):
                provider._loading_report(info, self.floating / "model.safetensors")

    def test_position_buffer_requires_exact_0_to_513_values(self):
        weights = self.floating / "model.safetensors"
        header = json.dumps({"roberta.embeddings.position_ids": {
            "dtype": "I64", "shape": [1, 514], "data_offsets": [0, 4112]}}).encode()
        info = {**self.loading, "unexpected_keys": {"roberta.embeddings.position_ids"}}
        weights.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<514q", *range(514)))
        self.assertTrue(provider._loading_report(info, weights)["ignored_buffer_values_verified"])
        weights.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<514q", *([1] * 514)))
        with self.assertRaisesRegex(ValueError, "POSITION_BUFFER_INVALID"):
            provider._loading_report(info, weights)

    def test_wrong_model_architecture_is_rejected_before_session_publish(self):
        self.model.config.num_labels = 2
        with self.assertRaisesRegex(ValueError, "ARCHITECTURE_INVALID"): self.load()
        self.assertEqual(self.model.calls, 0)

    def test_cpu_parameters_wrong_dtype_or_training_are_rejected(self):
        for field, value in (("device", "cpu"), ("dtype", "torch.float32")):
            model = XLMRobertaForSequenceClassification(); setattr(model.parameter, field, value)
            with self.assertRaisesRegex(ValueError, "MODEL_DEVICE_OR_DTYPE_INVALID"):
                provider.MpsSession(model, self.torch)
        self.model.training = True
        with self.assertRaisesRegex(ValueError, "MODEL_DEVICE_OR_DTYPE_INVALID"):
            provider.MpsSession(self.model, self.torch)

    def test_each_batch_rechecks_model_device(self):
        session = provider.MpsSession(self.model, self.torch)
        self.model.parameter.device = "cpu"
        with self.assertRaisesRegex(ValueError, "MODEL_DEVICE_OR_DTYPE_INVALID"):
            session.run(None, self.feed())
        self.assertEqual(self.model.calls, 0)

    def test_actual_integer_transfer_and_synchronized_readback(self):
        session = provider.MpsSession(self.model, self.torch)
        values = session.run(None, self.feed())[0]
        np.testing.assert_array_equal(values, [[0], [3]])
        self.assertEqual((self.model.calls, self.torch.syncs), (1, 2))

    def test_corrupt_or_cpu_input_transfer_stops_before_model(self):
        session = provider.MpsSession(self.model, self.torch)
        self.torch.corrupt = True
        with self.assertRaisesRegex(ValueError, "INPUT_TRANSFER_INVALID"): session.run(None, self.feed())
        self.torch.corrupt, self.torch.transfer_device = False, "cpu"
        with self.assertRaisesRegex(ValueError, "INPUT_TRANSFER_INVALID"): session.run(None, self.feed())
        self.assertEqual(self.model.calls, 0)

    def test_non_integer_bad_shapes_masks_or_names_are_rejected(self):
        session = provider.MpsSession(self.model, self.torch)
        for feed in ({**self.feed(), "token_type_ids": np.zeros((2, 3), dtype=np.int64)},
                     {**self.feed(), "input_ids": np.ones((2, 3), dtype=np.float32)},
                     {**self.feed(), "attention_mask": np.ones((1, 3), dtype=np.int64)},
                     {**self.feed(), "attention_mask": np.full((2, 3), 2, dtype=np.int64)},
                     {**self.feed(), "attention_mask": np.zeros((2, 3), dtype=np.int64)}, self.feed(5)):
            with self.assertRaisesRegex(ValueError, "INPUT_INVALID"): session.run(None, feed)
        self.assertEqual(self.model.calls, 0)

    def test_wrong_output_device_dtype_shape_or_nan_is_rejected(self):
        for field, value in (("output_device", "cpu"), ("output_dtype", "torch.float32"),
                             ("output_shape", (2, 2)), ("nan", True)):
            model = XLMRobertaForSequenceClassification(); setattr(model, field, value)
            session = provider.MpsSession(model, self.torch)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "OUTPUT"):
                session.run(None, self.feed())

    def test_original_query_passages_batch4_and_order_are_preserved(self):
        engine = self.load()
        passages = ["original passage " + str(i) for i in (3, 0, 4, 1, 2)]
        self.assertEqual(engine.score("原问题不改写", passages), [13, 10, 14, 11, 12])
        self.assertEqual(self.tokenizer.encoded, [("原问题不改写", text) for text in passages])
        self.assertEqual((self.model.calls, engine.last_stats["pairs"]), (2, 5))

    def test_oversized_inputs_are_rejected_without_inference_or_truncation(self):
        engine = self.load()
        with patch.object(self.tokenizer, "encode", return_value=SimpleNamespace(ids=list(range(513)))):
            with self.assertRaisesRegex(ValueError, "INPUT_TOO_LONG"):
                engine.score("original", ["original passage 0"])
        with self.assertRaisesRegex(ValueError, "CANDIDATE_LIMIT"):
            engine.score("original", ["original passage 0"] * 65)
        self.assertEqual(self.model.calls, 0)


if __name__ == "__main__":
    unittest.main()
