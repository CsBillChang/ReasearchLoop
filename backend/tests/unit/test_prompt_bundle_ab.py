"""联合包的冻结、授权、评分及真实HTTP输入装配；仅合成材料与SQLite。"""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import prompt_bundle_ab as bundle
import harness_ab_pair_gate as gate
from run_read_paper_ab import create_bundle_input_task, validate_manifest


class BundleSourceTests(unittest.TestCase):
    """只允许完整批准文件差异；源码与已加载对象分别验证。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = {v: str(self.root / v) for v in ('A', 'B')}
        for variant in self.roots:
            root = Path(self.roots[variant]); (root / 'backend/scripts').mkdir(parents=True)
            for name in bundle.BUSINESS_FILES:
                if variant == 'A' and name == bundle.NEW_PARTS:
                    continue
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('VALUE = ' + repr(variant) + '\n')
            (root / 'backend/scripts/shared.py').write_text('SAME = True\n')
        self.manifest = {'experiment_kind': bundle.KIND, 'source_roots': self.roots,
            'settings': {}, 'variant_settings': {'A': {}, 'B': {}},
            'prompt_bundle_source_contract': bundle.build_contract(self.roots)}

    def test_exact_runtime_file_delta_passes(self):
        self.assertEqual(bundle.validate_sources(self.manifest), self.manifest['prompt_bundle_source_contract'])

    def test_unapproved_business_or_common_change_is_rejected(self):
        p = Path(self.roots['B']) / 'backend/scripts/shared.py'; p.write_text('SAME = False\n')
        with self.assertRaises(PermissionError): bundle.validate_sources(self.manifest)

    def test_postfreeze_edit_of_approved_file_is_rejected(self):
        p = Path(self.roots['B']) / bundle.NEW_PARTS; p.write_text('VALUE = "changed after freeze"\n')
        with self.assertRaises(PermissionError): bundle.validate_sources(self.manifest)

    def test_symlink_source_is_rejected(self):
        (Path(self.roots['B']) / 'backend/scripts/link.py').symlink_to(Path(self.roots['A']) / 'backend/scripts/shared.py')
        with self.assertRaises(PermissionError): bundle.validate_sources(self.manifest)

    def test_old_business_switch_or_legacy_contract_is_rejected(self):
        for update in ({'settings': {'context_deduplicate_read_paper': False}},
                       {'variant_settings': {'A': {'context_deduplicate_read_paper': False}, 'B': {}}},
                       {'supervisor_prompt_contract': {}}, {'handoff_source_contract': {}}):
            with self.subTest(update=update), self.assertRaises(PermissionError):
                bundle.validate_sources({**self.manifest, **update})

    def test_runtime_class_and_role_mutation_fail_identity(self):
        source = str(Path(__file__).resolve().parents[3])
        identity = bundle.loaded_identity(source)
        manifest = {'source_roots': {'B': source}, 'loaded_prompt_contract': {'B': identity}}
        self.assertEqual(bundle.verify_loaded(manifest, 'B'), identity)
        self.assertEqual(len(identity['registry_sha256']), 12)
        from research_loop.agents import supervisor, researcher
        from research_loop.agents.memory import ContextService
        for target, attribute, value in ((supervisor, 'SYSTEM_PROMPT', 'unexpected prompt'),
                (ContextService, 'prepare_role', lambda *args: None),
                (researcher, 'finish_spec', lambda: (_ for _ in ()).throw(PermissionError('tampered registry')))):
            with self.subTest(attribute=attribute), patch.object(target, attribute, value), self.assertRaises(PermissionError):
                bundle.verify_loaded(manifest, 'B')
        self.assertEqual(bundle.verify_loaded(manifest, 'B'), identity)


class CodeFingerprintTests(unittest.TestCase):
    """检查指纹的稳定性及对真实代码变化的敏感性。"""

    def test_reference_sharing_does_not_change_identity(self):
        code = compile('result = None', 'synthetic.py', 'exec')
        shared = bytes(bytearray(b'synthetic value'))
        separate = bytes(bytearray(shared))
        self.assertIsNot(shared, separate)
        left = code.replace(co_consts=((shared, shared),))
        right = code.replace(co_consts=((shared, separate),))
        self.assertEqual(bundle.code_digest(left), bundle.code_digest(right))

    def test_constant_types_and_float_bits_are_not_collapsed(self):
        code = compile('result = None', 'synthetic.py', 'exec')
        constants = [None, Ellipsis, True, 1, 1.0, 0.0, -0.0, 1j, 'x', b'x',
                     ('x',), frozenset({'x'}), slice(1, 4, 2)]
        hashes = [bundle.code_digest(code.replace(co_consts=(value,))) for value in constants]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_unordered_constants_are_stable(self):
        values = ['text', 2, b'bytes', (3, 4)]
        self.assertEqual(bundle.normalize_code(frozenset(values)),
                         bundle.normalize_code(frozenset(reversed(values))))

    def test_nested_code_and_exception_table_changes_are_detected(self):
        left = compile('def outer():\n def inner(): return 1\n return inner\n', 'synthetic.py', 'exec')
        right = compile('def outer():\n def inner(): return 2\n return inner\n', 'synthetic.py', 'exec')
        self.assertNotEqual(bundle.code_digest(left), bundle.code_digest(right))
        function = compile('try: x = 1\nexcept: x = 2\n', 'synthetic.py', 'exec')
        self.assertTrue(function.co_exceptiontable)
        self.assertNotEqual(bundle.code_digest(function),
                            bundle.code_digest(function.replace(co_exceptiontable=b'')))

    def test_executable_fields_and_legacy_identity_are_not_ignored(self):
        left = compile('def calc(a, b): return a + b', 'synthetic.py', 'exec')
        right = compile('def calc(a, b): return a - b', 'synthetic.py', 'exec')
        self.assertNotEqual(bundle.code_digest(left), bundle.code_digest(right))
        function = next(value for value in left.co_consts if hasattr(value, 'co_code'))
        for change in ({'co_argcount': 1}, {'co_flags': function.co_flags ^ 16},
                       {'co_varnames': ('changed', 'b')}, {'co_names': ('changed',)},
                       {'co_freevars': ('changed',)}, {'co_cellvars': ('changed',)}):
            with self.subTest(field=list(change)):
                self.assertNotEqual(bundle.code_digest(function),
                                    bundle.code_digest(function.replace(**change)))
        source = str(Path(__file__).resolve().parents[3])
        identity = bundle.loaded_identity(source)
        legacy = {key: value for key, value in identity.items()
                  if key not in {'code_fingerprint_format', 'python_runtime'}}
        with self.assertRaisesRegex(PermissionError, 'LOADED_IDENTITY_MISMATCH'):
            bundle.verify_loaded({'source_roots': {'B': source},
                                  'loaded_prompt_contract': {'B': legacy}}, 'B')

    def test_unknown_constant_fails_closed(self):
        code = compile('result = None', 'synthetic.py', 'exec')
        with self.assertRaises(TypeError):
            bundle.code_digest(code.replace(co_consts=(object(),)))

    def test_cold_preflight_runtime_import_and_repeated_identity_match(self):
        source = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {v: str(root / v) for v in ('A', 'B')}
            # 合成A/B只供源码预检；实际角色从当前被测源码加载。
            for variant, target in roots.items():
                (Path(target) / 'backend/scripts').mkdir(parents=True)
                for name in bundle.BUSINESS_FILES:
                    if variant == 'A' and name == bundle.NEW_PARTS:
                        continue
                    original = source / name
                    content = original.read_text() if original.exists() else 'SYNTHETIC = True\n'
                    path = Path(target) / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('# synthetic ' + variant + '\n' + content)
            manifest = {'experiment_kind': bundle.KIND, 'source_roots': roots,
                'settings': {}, 'variant_settings': {'A': {}, 'B': {}},
                'prompt_bundle_source_contract': bundle.build_contract(roots)}
            path = root / 'manifest.json'; path.write_text(json.dumps(manifest))
            script = r"""
import json,socket,sys
from pathlib import Path
source,manifest,mode=sys.argv[1:]
sys.dont_write_bytecode=True
sys.path[:0]=[str(Path(source)/'backend/scripts')]
socket.has_ipv6=False
# 在进程内禁止网络、数据库、子进程及真实配置读取。
def audit(event,args):
    if event in {'socket.connect','socket.bind','socket.getaddrinfo','socket.sendto',
                 'socket.sendmsg','sqlite3.connect','subprocess.Popen','os.system','os.posix_spawn'}:
        raise AssertionError('forbidden cold probe event: '+event)
    if event=='open' and isinstance(args[0],(str,bytes)) and Path(args[0]).name=='.env':
        raise AssertionError('dotenv read forbidden')
sys.addaudithook(audit)
import run_read_paper_ab as runner
runner.select_variant_source({'source_roots':{'B':source}},'B')
import prompt_bundle_ab as bundle
if 'ast' in mode:
    bundle.validate_sources(json.loads(Path(manifest).read_text()))
if 'runtime' in mode:
    import httpx,harness_ab_resources,harness_ab_transport,harness_ab_network
identity=bundle.loaded_identity(source)
frozen={'experiment_kind':bundle.KIND,'source_roots':{'B':source},
        'loaded_prompt_contract':{'B':identity}}
for _ in range(5):
    assert runner.verify_loaded_supervisor(frozen,'B')==identity
print(json.dumps(identity,sort_keys=True))
"""
            results = []
            for mode in ('cold', 'ast', 'runtime', 'ast-runtime'):
                result = subprocess.run([sys.executable, '-I', '-B', '-c', script,
                    str(source), str(path), mode], capture_output=True, text=True, timeout=60,
                    env={'PATH': os.environ.get('PATH', ''), 'HOME': str(root), 'TMPDIR': str(root),
                         'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
                self.assertEqual(result.returncode, 0, mode + ': ' + result.stderr)
                results.append(json.loads(result.stdout))
            self.assertTrue(all(value == results[0] for value in results))

    def test_runtime_function_schema_roles_handler_and_mixed_source_rejected(self):
        from dataclasses import replace
        from research_loop.agents import graph
        from research_loop.tools.registry import ToolRegistry
        import research_loop.domain.evidence as evidence
        source = str(Path(__file__).resolve().parents[3])
        identity = bundle.loaded_identity(source)
        manifest = {'source_roots': {'B': source}, 'loaded_prompt_contract': {'B': identity}}
        original = ToolRegistry.get
        changes = ({'description': 'synthetic changed description'}, {'roles': ('unexpected',)},
                   {'handler': lambda *args, **kwargs: None})
        for change in changes:
            def changed(registry, name, change=change):
                return replace(original(registry, name), **change)
            with self.subTest(change=list(change)), patch.object(ToolRegistry, 'get', changed):
                with self.assertRaises(PermissionError): bundle.verify_loaded(manifest, 'B')
        with patch.object(graph, 'build_graph', lambda *args: None):
            with self.assertRaises(PermissionError): bundle.verify_loaded(manifest, 'B')
        with patch.object(evidence, '__file__', '/synthetic/wrong-source.py'):
            with self.assertRaises(PermissionError): bundle.verify_loaded(manifest, 'B')
        self.assertEqual(bundle.verify_loaded(manifest, 'B'), identity)


class BundleActivationTests(unittest.TestCase):
    """新预算不得借历史授权或可由模型修改的文件放行。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {'experiment_kind': bundle.KIND, 'batch_id': 'new-synthetic',
            'runtime_root': str(self.root / 'runtime'), 'source_roots': {'A': str(self.root / 'A'), 'B': str(self.root / 'B')},
            'authorization_path': str(self.root / 'authorization.json'),
            'live_qualification_released': True, 'real_execution_released': True,
            'ledger_limits': {'batch_tokens': 100, 'batch_seconds': 150, 'carried_usage': {'seconds': 50}}, 'execution_order': [['S1', 'A'], ['S1', 'B']]}
        self.write_auth()

    def write_auth(self, **changes):
        m = self.manifest
        self.auth = {'approved': True, 'experiment_kind': bundle.KIND, 'batch_id': m['batch_id'],
            'manifest_sha256': gate.manifest_identity(m), 'ledger_limits': m['ledger_limits'],
            'execution_order': m['execution_order'], 'allowed_modes': ['qualify', 'execute'], 'expires_at': 200.0,
            'round_started_at': 50.0, 'round_absolute_deadline': 150.0, **changes}
        Path(m['authorization_path']).parent.mkdir(parents=True, exist_ok=True)
        Path(m['authorization_path']).write_text(json.dumps(self.auth))

    def test_proposal_refuses_before_runtime_or_budget_creation(self):
        for mode in ('qualify', 'execute'):
            with self.assertRaisesRegex(PermissionError, 'NEW_AUTHORIZATION'):
                validate_manifest({**self.manifest, 'live_qualification_released': False,
                    'real_execution_released': False}, 'S1', 'A', mode)
        self.assertFalse(Path(self.manifest['runtime_root']).exists())

    def test_current_exact_authorization_is_recognized(self):
        self.assertEqual(bundle.validate_activation(self.manifest, 'execute', clock=lambda: 100), self.auth)

    def test_stale_wrong_mode_nonfinite_and_changed_caps_are_rejected(self):
        for changes in ({'experiment_kind': 'supervisor_goal_v1'}, {'batch_id': 'old'},
                        {'manifest_sha256': '0'*64}, {'ledger_limits': {}}, {'allowed_modes': ['qualify']},
                        {'expires_at': 100.0}, {'expires_at': float('nan')}, {'expires_at': float('inf')},
                        {'expires_at': True}, {'approved': False}):
            self.write_auth(**changes)
            with self.subTest(changes=changes), self.assertRaises(PermissionError):
                bundle.validate_activation(self.manifest, 'execute', clock=lambda: 100)

    def test_round_clock_binds_qualification_and_payload_cannot_restart_it(self):
        self.assertEqual(bundle.validate_outer_start(self.manifest, 50.0, clock=lambda: 100), 150.0)
        with self.assertRaisesRegex(PermissionError, 'OUTER_START'):
            bundle.validate_outer_start(self.manifest, 90.0, clock=lambda: 100)
        for changes in ({'round_started_at': None}, {'round_started_at': float('nan')},
                {'round_absolute_deadline': 151.0}, {'round_started_at': 101.0},
                {'round_absolute_deadline': float('inf')}, {'round_started_at': True}):
            self.write_auth(**changes)
            with self.subTest(changes=changes), self.assertRaises(PermissionError):
                bundle.validate_activation(self.manifest, 'qualify', clock=lambda: 100)
        self.write_auth()
        with self.assertRaisesRegex(PermissionError, 'ROUND_CLOCK'):
            bundle.validate_activation(self.manifest, 'execute', clock=lambda: 150)

    def test_authorization_cannot_be_in_runtime_or_source(self):
        for directory in ('runtime', 'A', 'B'):
            self.manifest['authorization_path'] = str(self.root / directory / 'authorization.json')
            self.write_auth()
            with self.assertRaisesRegex(PermissionError, 'OUTSIDE_MODEL_AND_SOURCE'):
                bundle.validate_activation(self.manifest, 'execute', clock=lambda: 100)


class BundlePairTests(unittest.TestCase):
    """零材料Memory、多源与预置条件都绑定审核；不猜证据所属论文。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, count):
        materials = []
        for i in range(count):
            path = self.root / f'{i}.pdf'; path.write_bytes(f'%PDF-synthetic-{i}'.encode())
            materials.append({'pdf_path': str(path), 'pdf_sha256': gate.sha256(path.read_bytes()), 'physical_pages': 1})
        case = {'case_id': 'S1', 'prompt': 'Synthetic current request', 'materials': materials,
            'material': materials[0] if materials else None,
            'preconditions': {'history_messages': [{'role': 'user', 'content': 'Synthetic previous request'},
                {'role': 'assistant', 'content': 'Synthetic previous answer'}] if not materials else []},
            'private_scoring_label': 'DO_NOT_INCLUDE'}
        cases = self.root / 'cases.json'; cases.write_text(json.dumps({'cases': [case]}))
        manifest = {'experiment_kind': bundle.KIND, 'cases_path': str(cases),
            'runtime_root': str(self.root / 'runtime'), 'semantic_gate_root': str(self.root / 'control')}
        rows = []
        for variant in ('A', 'B'):
            rid = 'synthetic-' + variant; cell = self.root / 'runtime/cells' / ('S1-' + variant); cell.mkdir(parents=True)
            for name, value in [('result.json', {'status': 'executed', 'root_id': rid}),
                ('task-view.json', {'task_id': rid, 'status': 'completed', 'prompt': case['prompt'],
                    'result': {'answer_markdown': 'Synthetic answer'}}), ('model-calls.json', []),
                ('actions.json', [{'observation': {'evidence_id': 'e-synthetic', 'text': 'Unbound source', 'physical_page': 1}}])]:
                (cell / name).write_text(json.dumps(value))
            rows.append({'case_id': 'S1', 'variant': variant, 'root_id': rid, 'issues': []})
        return manifest, rows, materials

    def review(self, manifest, challenge, materials):
        return {'version': 1, 'challenge_nonce': challenge['nonce'], 'challenge_sha256': gate.sha256(gate.canonical(challenge)),
            'reviewed_at': 101, 'reviewer': {'kind': 'human', 'identity': 'synthetic reviewer', 'blinded': True,
                'full_authorized_source_checked': True, 'paid_model_calls': 0},
            'reviews': [{'anonymous_id': value, 'answer_sha256': challenge['identity']['answers'][variant]['answer_sha256'],
                'source_sha256s': [m['pdf_sha256'] for m in materials], 'whole_answer_quality': 'full',
                'core_complete': True, 'critical_core_error': False, 'requested_scope_complete': True,
                'evidence_notes': 'Synthetic complete check'} for variant, value in challenge['anonymous_ids'].items()]}

    def test_zero_material_memory_pair_passes_without_pdf_or_model_receipts(self):
        manifest, rows, materials = self.fixture(0)
        challenge = gate.prepare_pair_gate(manifest, 'S1', rows, 'manifest', clock=lambda: 100)
        packet = gate.read_json(self.root / 'control/S1/review-packet.json')
        self.assertEqual(packet['materials'], [])
        self.assertEqual(len(packet['input_preconditions']['history_messages']), 2)
        self.assertNotIn('DO_NOT_INCLUDE', json.dumps(packet))
        result = gate.evaluate_pair_review(manifest, 'S1', rows, 'manifest', challenge,
            self.review(manifest, challenge, materials), clock=lambda: 102)
        self.assertEqual(result['status'], 'passed')

    def test_both_partial_usable_are_stopped_for_joint_candidate(self):
        manifest, rows, materials = self.fixture(0)
        challenge = gate.prepare_pair_gate(manifest, 'S1', rows, 'manifest', clock=lambda: 100)
        review = self.review(manifest, challenge, materials)
        for observation in review['reviews']:
            observation['whole_answer_quality'] = 'partial_usable'
        result = gate.evaluate_pair_review(manifest, 'S1', rows, 'manifest', challenge, review, clock=lambda: 102)
        self.assertEqual(result['status'], 'stopped')
        self.assertIn('both_answers_below_full_quality', result['stop_reasons'])

    def test_two_material_pair_binds_both_and_never_guesses_unbound_source(self):
        manifest, rows, materials = self.fixture(2)
        challenge = gate.prepare_pair_gate(manifest, 'S1', rows, 'manifest', clock=lambda: 100)
        packet = gate.read_json(self.root / 'control/S1/review-packet.json')
        self.assertEqual(packet['materials'], materials)
        for answer in packet['answers']:
            evidence = answer['evidence_locators']['evidence'][0]
            self.assertIsNone(evidence['source_sha256'])
            self.assertEqual(evidence['source_binding'], 'unresolved_no_source_guess')
        review = self.review(manifest, challenge, materials)
        self.assertEqual(gate.evaluate_pair_review(manifest, 'S1', rows, 'manifest', challenge, review, clock=lambda: 102)['status'], 'passed')
        review['reviews'][0]['source_sha256s'] = review['reviews'][0]['source_sha256s'][:1]
        with self.assertRaisesRegex(PermissionError, 'MATERIAL_MISMATCH'):
            gate.evaluate_pair_review(manifest, 'S1', rows, 'manifest', challenge, review, clock=lambda: 102)
        Path(materials[1]['pdf_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(PermissionError, 'MATERIAL_CHANGED'):
            gate.pair_identity(manifest, 'S1', rows)


class BundleBackendInputTests(unittest.TestCase):
    """实际共同入口→HTTP API→SQLite/授权路径，不仅检查helper返回值。"""

    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='prompt-bundle-api-'))
        self.root = Path(directory)
        from research_loop import main
        from research_loop.api import dependencies
        from research_loop.config import Settings
        from research_loop.storage.mysql import Database
        from fastapi.testclient import TestClient
        self.settings = Settings(project_root=self.root, data_dir=self.root/'data', output_dir=self.root/'output',
            database_url='sqlite:///' + str(self.root/'task.db'))
        self.db = Database(self.settings.database_url); self.db.ensure_schema()
        overrides = dict(main.app.dependency_overrides)
        main.app.dependency_overrides[dependencies.get_runtime_settings] = lambda: self.settings
        main.app.dependency_overrides[dependencies.get_database] = lambda: self.db
        self.stack.callback(lambda: (main.app.dependency_overrides.clear(), main.app.dependency_overrides.update(overrides)))
        self.stack.enter_context(patch.object(main, 'get_settings', return_value=self.settings))
        self.client = self.stack.enter_context(TestClient(main.app))
        self.materials, self.bundles = [], {}
        for index in range(2):
            import fitz
            path = self.root/f'{index}.pdf'
            pdf = fitz.open(); page = pdf.new_page(); page.insert_text((72, 72), f'Synthetic {index}'); pdf.save(path); pdf.close()
            digest = gate.sha256(path.read_bytes())
            self.materials.append({'pdf_path': str(path), 'pdf_sha256': digest, 'physical_pages': 1})
            self.bundles[digest] = {'document_ref': {'paper_id': f'paper-{index}', 'metadata': {'source': 'arxiv'}}}

    def case(self, name, materials, **pre):
        return {'case_id': name, 'prompt': 'Synthetic user draft and question, not a scoring label',
            'materials': materials, 'material': materials[0] if materials else None, 'preconditions': pre,
            'grading': {'hidden': 'do not expose'}}

    def test_memory_input_is_seeded_in_current_conversation_without_fake_old_task(self):
        from research_loop.domain.memory import HistoryRead
        from research_loop.services.memory import MemoryService
        history = [{'role': 'user', 'content': 'Earlier synthetic request'}, {'role': 'assistant', 'content': 'Earlier synthetic answer'}]
        case = self.case('Memory', [], history_messages=history)
        report, attachments = create_bundle_input_task(self.client, self.db, case, 'B', 'execute', {})
        self.assertEqual(attachments, []); self.assertEqual(report['upload_count'], 0)
        current = self.db.get_task(report['root_id'])
        self.assertEqual(current['prompt'], case['prompt']); self.assertEqual(current['save_policy'], 'deny')
        self.assertEqual(json.loads(current['attachments_json']), [])
        rows = self.db.fetchall('SELECT task_id,role,content FROM conversation_messages WHERE conversation_id=? ORDER BY sequence_no', (report['conversation_id'],))
        self.assertEqual([(r['role'], r['content']) for r in rows[:2]], [(r['role'], r['content']) for r in history])
        self.assertTrue(all(r['task_id'] == '' for r in rows[:2]))
        self.assertEqual(len(self.db.fetchall('SELECT * FROM tasks')), 1)
        self.assertEqual(self.db.list_model_calls(report['root_id']), [])
        readback = MemoryService(self.db).read(report['root_id'], HistoryRead(source_type='conversation', source_id=report['conversation_id']))
        self.assertIn('Earlier synthetic answer', json.dumps(readback, ensure_ascii=False))
        self.assertNotIn('grading', json.dumps(readback))

    def test_catalog_has_no_root_attachments_and_search_grants_actual_candidates(self):
        from prompt_bundle_cases import install_catalog
        from research_loop.providers.discovery import DiscoveryProvider
        from research_loop.services.papers import PaperService
        catalog = [{'material_index': i, 'title': f'Title {i}', 'abstract': f'Abstract {i}',
            'source_url': f'https://example.org/{i}'} for i in range(2)]
        case = self.case('Catalog', self.materials, catalog=catalog)
        install_catalog(self.stack, DiscoveryProvider, case, self.bundles)
        report, attachments = create_bundle_input_task(self.client, self.db, case, 'A', 'execute', self.bundles)
        self.assertEqual(attachments, []); self.assertFalse(self.db.task_has_paper(report['root_id'], 'paper-0'))
        discovery = DiscoveryProvider.from_settings(self.settings)
        results = PaperService(self.db, discovery, None).search('synthetic', task_id=report['root_id'])
        self.assertEqual(len(results.candidates), 2)
        self.assertTrue(self.db.task_has_paper(report['root_id'], 'paper-0'))
        self.assertEqual(self.db.list_task_evidence(report['root_id']), [])

    def test_multiple_uploads_current_draft_and_save_permission_survive_api(self):
        case = self.case('Deep', self.materials, save_policy='allow')
        report, attachments = create_bundle_input_task(self.client, self.db, case, 'B', 'execute', self.bundles)
        self.assertEqual(len(attachments), 2); self.assertEqual(report['upload_count'], 2)
        current = self.db.get_task(report['root_id'])
        self.assertEqual(current['save_policy'], 'allow')
        self.assertEqual(current['prompt'], case['prompt'])
        self.assertEqual([x['paper_id'] for x in json.loads(current['attachments_json'])], ['paper-0', 'paper-1'])
        self.assertEqual(self.db.list_model_calls(report['root_id']), [])
        self.assertFalse(any(self.settings.output_dir.rglob('*.md')))


if __name__ == '__main__':
    unittest.main()
