"""本机模型接口合同；显式提供冻结模型时执行，不自动下载依赖或模型。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from research_loop.domain.retrieval import ChunkConfig, split_page
from research_loop.providers.embeddings import LocalEmbedding
from research_loop.providers.reranker import LocalReranker
from research_loop.storage.vector_index import valid_vector


class LocalModelTests(unittest.TestCase):
    """真实tokenizer/ONNX的容量、定位和hash合同；不是语义评测。"""

    @classmethod
    def setUpClass(cls):
        """读取显式路径一次，默认离线套件不要求大模型文件。"""
        path = os.getenv('AG71_MODEL_MANIFEST')
        if not path:
            raise unittest.SkipTest('explicit AG71_MODEL_MANIFEST required; no automatic downloads')
        cls.path = Path(path)
        cls.model = LocalEmbedding(cls.path)

    def test_real_normalized_dimension(self):
        """中英文输入均输出实际384维单位向量。"""
        output = self.model.encode(['研究用户的兴趣记忆。', 'Open vocabulary detection aligns language and image regions.'])
        self.assertEqual(len(output), 2)
        self.assertTrue(all(valid_vector(v, 384) for v in output))

    def test_capacity_refuses_instead_of_silent_truncation(self):
        """超容量必须拒绝，不让末尾信息静默消失。"""
        with self.assertRaisesRegex(ValueError, 'INPUT_TOO_LONG'):
            self.model.encode(['token ' * 1000])

    def test_unicode_long_paragraph_uses_exact_python_character_offsets(self):
        """长中英混合与符号分块仍能逐字还原，模型可实际编码各块。"""
        text = ('中文记忆 μ=0.07 ±4.0 αβ e\u0301 🧪 research region-language. ' * 150)
        page = {'text': text, 'evidence_id': 'synthetic-unicode', 'page_number': None}
        chunks = split_page(page, {'document_id': 'synthetic', 'content_sha256': 'test'},
            ChunkConfig(), self.model.tokenizer_id, self.model.offsets)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(c.text for c in chunks), text)
        self.assertTrue(all(len(self.model.tokenizer.encode('passage: '+c.text).ids) <= 512 for c in chunks))
        self.assertTrue(all(valid_vector(v, 384) for v in self.model.encode([c.text for c in chunks[:4]])))

    def test_model_hash_mismatch_rejected_before_inference(self):
        """错误下载清单不能启用模型，也不会尝试自动重新下载。"""
        spec = json.loads(self.path.read_text())
        weight = spec['embedding']['weights']
        spec['embedding']['files'][weight]['sha256'] = '0'*64
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'wrong.json'
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'MODEL_FILE_HASH_MISMATCH'):
                LocalEmbedding(path)

    def test_real_reranker_multilingual_pair_and_raw_scores(self):
        """真实中文问题/英文片段冒烟；分数是logit，不写成准确率或概率。"""
        model=LocalReranker(self.path)
        scores=model.score('如何识别没见过的物体类别？',[
            'Open vocabulary object detection aligns image regions with language embeddings of unseen category names.',
            'Photosynthesis produces oxygen from water using sunlight.'])
        self.assertGreater(scores[0],scores[1])
        self.assertEqual(model.identity['score_kind'],'raw_relevance_logit')

    def test_real_reranker_capacity_and_integrity(self):
        """长度拒绝和hash拒绝不会静默截断或重新下载权重。"""
        model=LocalReranker(self.path)
        with self.assertRaisesRegex(ValueError,'RERANK_INPUT_TOO_LONG'):
            model.score('term '*600,['source'])
        spec=json.loads(self.path.read_text()); key=spec['reranker']['weights']
        spec['reranker']['files'][key]['sha256']='0'*64
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'wrong.json'; path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'RERANK_MODEL_HASH_MISMATCH'):
                LocalReranker(path).score('query',['source'])
