"""54篇受控题录的查询可达性与权限回归；仅使用合成材料和隔离SQLite。"""
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from agent_final_catalog import ranked_indices
from prompt_bundle_cases import install_catalog, seed_history, upload_materials, validate_preconditions
from research_loop.services.papers import PaperService
from research_loop.storage.mysql import Database
from research_loop.tools.papers import SearchPapersArgs


class _Discovery:
    """替换只在本测试ExitStack内存在，退出后默认拒绝发现。"""

    def search(self, *args, **kwargs):
        raise PermissionError('external discovery forbidden')


class AgentFinalCatalogTests(unittest.TestCase):
    """保留旧20条合同，新版在全部54条上排序后截断返回。"""

    def setUp(self):
        """建立不含答案标签的合成54条目录，不连接生产或模型。"""
        self.materials, self.bundles, self.catalog = [], {}, []
        for index in range(54):
            content_hash = sha256(f'isolated-synthetic-pdf-{index}'.encode()).hexdigest()
            self.materials.append({'material_id': f'P{index + 1:02}', 'pdf_path': f'/synthetic/never-opened/{index}.pdf',
                                   'pdf_sha256': content_hash, 'physical_pages': 1})
            self.bundles[content_hash] = {'document_ref': {'paper_id': f'paper-{index:02}',
                'metadata': {'source': 'arxiv' if index % 2 == 0 else 'cvpr'}}}
            self.catalog.append({'material_index': index, 'title': f'Frozen neutral study {index}',
                'abstract': f'This synthetic study addresses distincttopic{index} without a relevance label.',
                'authors': ['Synthetic Author'], 'source_url': f'https://example.invalid/paper/{index}'})
        self.case = {'suite_kind': 'agent_final_v1', 'materials': self.materials,
                     'preconditions': {'catalog': self.catalog, 'save_policy': 'deny'}}

    def test_54_requires_explicit_version_and_55_remains_rejected(self):
        """旧批次无法因新适配自动扩大目录，未知suite标签仍走旧限额。"""
        self.assertEqual(len(validate_preconditions(self.case)['catalog']), 54)
        for marker in (None, 'prompt_bundle_v1', 'agent_final_typo'):
            old = deepcopy(self.case)
            old['suite_kind'] = marker
            with self.assertRaisesRegex(ValueError, 'TOO_LARGE'):
                validate_preconditions(old)
        oversized = deepcopy(self.case)
        oversized['preconditions']['catalog'].append(deepcopy(self.catalog[0]))
        with self.assertRaisesRegex(ValueError, 'TOO_LARGE'):
            validate_preconditions(oversized)

    def test_each_tail_item_is_reachable_with_original_tool_limit(self):
        """全部54篇均参与排序；尾部论文无需扩展业务工具limit即可返回。"""
        with ExitStack() as stack:
            installed = install_catalog(stack, _Discovery, self.case, self.bundles)
            self.assertEqual(installed['candidate_count'], 54)
            self.assertFalse(installed['original_evidence_grants_created'])
            provider = _Discovery()
            for index in range(54):
                result = provider.search(f'distincttopic{index}', limit=1)
                self.assertEqual([c.paper_id for c in result.candidates], [f'paper-{index:02}'])
                self.assertTrue(result.truncated)
        self.assertEqual(SearchPapersArgs.model_json_schema()['properties']['limit']['maximum'], 20)

    def test_query_changes_order_and_repeated_calls_are_identical(self):
        """重查保持稳定，换主题确实改变结果，结果不受调用历史影响。"""
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.case, self.bundles)
            provider = _Discovery()
            first = provider.search('distincttopic53', limit=5)
            different = provider.search('distincttopic3', limit=5)
            repeated = provider.search('distincttopic53', limit=5)
            self.assertEqual(first.model_dump(), repeated.model_dump())
            self.assertNotEqual(first.candidates[0].paper_id, different.candidates[0].paper_id)
            self.assertIn('lexical_bm25', first.query_scope)
            self.assertIn('not live search', first.source_results[0].message)

    def test_equal_scores_keep_frozen_order_and_results_are_copies(self):
        """无词项命中时说明性中性顺序稳定，修改返回结果不污染池。"""
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.case, self.bundles)
            provider = _Discovery()
            first = provider.search('unmatchedword', limit=54)
            self.assertEqual([c.paper_id for c in first.candidates], [f'paper-{i:02}' for i in range(54)])
            self.assertFalse(first.truncated)
            first.candidates[0].title = 'mutated'
            self.assertEqual(provider.search('unmatchedword', limit=1).candidates[0].title,
                             self.catalog[0]['title'])

    def test_source_filter_applies_before_ranking(self):
        """源条件约束完整池，不能因强匹配而把被排除来源重新加回。"""
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.case, self.bundles)
            provider = _Discovery()
            result = provider.search('distincttopic53', sources=['ARXIV'], limit=54)
            self.assertEqual(len(result.candidates), 27)
            self.assertTrue(all(c.source == 'arxiv' for c in result.candidates))
            self.assertNotIn('paper-53', [c.paper_id for c in result.candidates])
            self.assertFalse(provider.search('distincttopic53', sources=['missing']).candidates)
            for invalid in ('arxiv', [None], ['']):
                with self.assertRaisesRegex(ValueError, 'SOURCES_INVALID'):
                    provider.search('topic', sources=invalid)

    def test_unknown_dates_never_become_verified_publication_dates(self):
        """历史日期字段不赋予核验资格；有日期条件时明确返回缺证。"""
        self.catalog[53]['published_at'] = '2026-01-01'
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.case, self.bundles)
            provider = _Discovery()
            ordinary = provider.search('distincttopic53', limit=1)
            self.assertEqual(ordinary.candidates[0].metadata['material_id'], 'P54')
            self.assertEqual(ordinary.candidates[0].metadata['catalog_material_index'], 53)
            self.assertIsNone(ordinary.candidates[0].published)
            self.assertEqual(ordinary.candidates[0].date_precision, 'unknown')
            dated = provider.search('distincttopic53', date_from='2025-01-01', date_to='2027-01-01')
            self.assertFalse(dated.candidates)
            self.assertEqual(dated.source_results[0].status, 'partial_results')
            self.assertIn('publication dates unverified', dated.source_results[0].message)

    def test_gold_and_ranking_fields_are_rejected_before_install(self):
        """新目录仍使用原白名单，答案、人工相关性和权重不能混入种子。"""
        for key in ('gold', 'score', 'rank', 'expected_answer', 'relevant', 'ranking_weight'):
            changed = deepcopy(self.case)
            changed['preconditions']['catalog'][0][key] = 'forbidden'
            with ExitStack() as stack, self.assertRaisesRegex(ValueError, 'CATALOG_INVALID'):
                install_catalog(stack, _Discovery, changed, self.bundles)
        changed = deepcopy(self.case)
        changed['preconditions']['answers'] = ['forbidden']
        with self.assertRaisesRegex(ValueError, 'PRECONDITION_INVALID'):
            validate_preconditions(changed)

    def test_catalog_does_not_upload_54_attachments_or_pregrant_materials(self):
        """原PaperService仅授予实际返回候选，不把54篇变成必须读完的附件。"""
        client = Mock()
        self.assertEqual(upload_materials(client, self.case, self.bundles), [])
        client.post.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            db = Database('sqlite:///' + str(Path(directory)/'synthetic.sqlite3'))
            db.ensure_schema()
            db.create_task('root', None, 'input', 'find a paper', 'deny', [])
            self.assertEqual(seed_history(db, 'no-conversation-needed', self.case['preconditions'],
                                         suite_kind='agent_final_v1')['message_count'], 0)
            with ExitStack() as stack:
                install_catalog(stack, _Discovery, self.case, self.bundles)
                self.assertFalse(db.task_has_paper('root', 'paper-53'))
                self.assertFalse(db.task_has_paper('root', 'paper-00'))
                found = PaperService(db, _Discovery(), None).search('distincttopic53', limit=1, task_id='root')
                self.assertEqual(found.candidates[0].paper_id, 'paper-53')
                self.assertTrue(db.task_has_paper('root', 'paper-53'))
                self.assertFalse(db.task_has_paper('root', 'paper-00'))
                self.assertEqual(db.list_task_evidence('root'), [])
                self.assertEqual(db.get_task('root')['attachments_json'], '[]')

    def test_legacy_20_stays_fixed_order(self):
        """新排序不会改变旧冻结实验的测量条件。"""
        old = deepcopy(self.case)
        old.pop('suite_kind')
        old['materials'] = old['materials'][:20]
        old['preconditions']['catalog'] = old['preconditions']['catalog'][:20]
        with ExitStack() as stack:
            installed = install_catalog(stack, _Discovery, old, self.bundles)
            self.assertNotIn('ranking', installed)
            result = _Discovery().search('distincttopic19', limit=1)
            self.assertEqual(result.candidates[0].paper_id, 'paper-00')
            self.assertEqual(result.query_scope, 'frozen_local_candidate_pool_unranked_not_web_search')

    def test_catalog_requires_complete_source_binding_and_restores_provider(self):
        """绑定不完整在执行前拒绝，退出上下文后不遗留替换。"""
        incomplete = dict(self.bundles)
        incomplete.pop(self.materials[-1]['pdf_sha256'])
        with ExitStack() as stack, self.assertRaisesRegex(ValueError, 'SOURCE_MISSING'):
            install_catalog(stack, _Discovery, self.case, incomplete)
        with self.assertRaises(PermissionError):
            _Discovery().search('distincttopic53')

    def test_pure_ranker_normalization_cjk_and_no_hidden_field_interface(self):
        """英文大小写/全角和中文词项可匹配，函数只接受题名摘要二元组。"""
        rows = [('Alpha', 'plain'), ('ＬｅｎｓＶＬＭ', '按需补读压缩文本')]
        self.assertEqual(ranked_indices(rows, 'lensvlm'), [1, 0])
        self.assertEqual(ranked_indices(rows, '补读'), [1, 0])
        self.assertEqual(ranked_indices(rows, 'missing'), [0, 1])
        for malformed in ([{'title': 'x', 'answer': 'y'}], [('x', 'y', 'gold')], [('', None)]):
            with self.assertRaises(ValueError):
                ranked_indices(malformed, 'topic')


if __name__ == '__main__':
    unittest.main()
