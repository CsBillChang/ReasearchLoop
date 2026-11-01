"""实验来源说明进入真实注册schema；不改变业务来源、参数、权限及检索结果。"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from research_loop.tools.base import ToolSpec
from research_loop.tools.papers import specs, SearchPapersArgs
from research_loop.tools.registry import ToolRegistry
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from prompt_bundle_cases import install_catalog
from unit import test_prompt_bundle_cases as fixtures

_Discovery = fixtures._Discovery


class CatalogRuntimeNoticeTests(unittest.TestCase):
    """用真实工具注册表覆盖角色、隐藏范围、动态材料数和恢复。"""

    setUp = fixtures.PromptBundleCasesTests.setUp

    def catalog_case(self):
        case = deepcopy(self.case)
        case['preconditions']['catalog'] = deepcopy(self.catalog)
        return case

    def schemas(self, role='supervisor', allowed=None):
        return ToolRegistry(specs(), allowed_names=allowed).schemas(role)

    def description(self, schemas):
        return next(item['function']['description'] for item in schemas if item['function']['name'] == 'search_papers')

    def test_visible_role_schemas_change_description_only_and_restore(self):
        before = {role: self.schemas(role) for role in ('supervisor', 'researcher', 'reviewer')}
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.catalog_case(), self.bundles)
            for role, original in before.items():
                actual = self.schemas(role)
                text = self.description(actual)
                self.assertIn('共2条', text)
                self.assertIn('省略sources', text)
                self.assertNotIn('共54条', text)
                normalized = deepcopy(actual)
                next(item['function'] for item in normalized if item['function']['name'] == 'search_papers')['description'] = self.description(original)
                self.assertEqual(normalized, original)
        self.assertEqual({role: self.schemas(role) for role in before}, before)

    def test_hidden_search_tool_does_not_expose_catalog(self):
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.catalog_case(), self.bundles)
            self.assertNotIn('本地候选题录池', json.dumps(self.schemas(allowed={'read_paper'}), ensure_ascii=False))

    def test_no_catalog_leaves_schema_unchanged(self):
        before = self.schemas()
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.case, self.bundles)
            self.assertEqual(self.schemas(), before)

    def test_nested_pools_never_accumulate_notices(self):
        original = ToolSpec.schema
        with ExitStack() as outer:
            install_catalog(outer, _Discovery, self.catalog_case(), self.bundles)
            with ExitStack() as inner:
                case = self.catalog_case()
                case['preconditions']['catalog'] = case['preconditions']['catalog'][:1]
                install_catalog(inner, _Discovery, case, self.bundles)
                text = self.description(self.schemas())
                self.assertIn('共1条', text)
                self.assertNotIn('共2条', text)
            self.assertIn('共2条', self.description(self.schemas()))
        self.assertIs(ToolSpec.schema, original)

    def test_source_filter_and_limit_keep_original_contract(self):
        self.bundles[self.materials[1]['pdf_sha256']]['document_ref']['metadata']['source'] = 'local'
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.catalog_case(), self.bundles)
            self.assertEqual(len(_Discovery().search('topic').candidates), 2)
            self.assertEqual([x.paper_id for x in _Discovery().search('topic', sources=['local']).candidates], ['paper-1'])
            self.assertTrue(_Discovery().search('topic', limit=1).truncated)
            self.assertFalse(_Discovery().search('topic', sources=['controlled_local_catalog']).candidates)
            with self.assertRaises(ValidationError):
                SearchPapersArgs(query='topic', limit=54)

    def test_material_id_notice_is_conditional_not_fabricated(self):
        case = self.catalog_case()
        case['suite_kind'] = 'agent_final_v1'
        case['materials'][0]['material_id'] = 'P-test'
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, case, self.bundles)
            self.assertIn('metadata.material_id', self.description(self.schemas()))
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.catalog_case(), self.bundles)
            self.assertNotIn('metadata.material_id', self.description(self.schemas()))

    def test_bad_material_identity_does_not_patch_schema(self):
        original = ToolSpec.schema
        case = self.catalog_case()
        case['preconditions']['catalog'][0]['material_index'] = 999
        with ExitStack() as stack, self.assertRaises(ValueError):
            install_catalog(stack, _Discovery, case, self.bundles)
        self.assertIs(ToolSpec.schema, original)

    def test_synthetic_labels_and_answers_do_not_enter_notice(self):
        case = self.catalog_case()
        case['gold_answer'] = 'PRIVATE_SCORE_SENTINEL'
        case['preconditions']['catalog'][0]['title'] = 'PRIVATE_TITLE_SENTINEL'
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, case, self.bundles)
            text = self.description(self.schemas())
            self.assertNotIn('PRIVATE_SCORE_SENTINEL', text)
            self.assertNotIn('PRIVATE_TITLE_SENTINEL', text)

    def test_loaded_identity_stays_cold_while_outbound_description_is_complete(self):
        from prompt_bundle_ab import loaded_identity, verify_loaded
        from research_loop.agents import supervisor
        source = Path(supervisor.__file__).resolve().parents[3]
        cold = loaded_identity(source)
        manifest = {'source_roots': {'B': str(source)}, 'loaded_prompt_contract': {'B': cold}}
        verify_loaded(manifest, 'B')
        raw_spec = next(spec for spec in specs() if spec.name == 'search_papers')
        original = raw_spec.schema()
        with ExitStack() as stack:
            install_catalog(stack, _Discovery, self.catalog_case(), self.bundles)
            self.assertIn('共2条', self.description(self.schemas()))
            verify_loaded(manifest, 'B')
            self.assertEqual(raw_spec.schema(), original)
        verify_loaded(manifest, 'B')


if __name__ == '__main__':
    unittest.main()
