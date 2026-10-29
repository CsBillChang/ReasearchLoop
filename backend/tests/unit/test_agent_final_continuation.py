"""续跑合同的只读锚、原时钟和唯一技术重测边界。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from agent_final_continuation import (build_continuation_contract, planned_order,
    retained_rows, validate_continuation, validate_shared_clock_carry)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(root, *, started_at=1000.0):
    """生成可供集成测试复用的96格封存锚与92格候选，无真实业务材料。

    Args:
        root: 调用者创建并拥有的临时目录。
        started_at: 可由集成测试指定的原时钟；默认值仅供纯合同单测。

    Returns:
        manifest、anchor_root、new_root、clock与order；调用者可读取原门禁证明。
    """
    from harness_ab_pair_gate import prepare_pair_gate, evaluate_pair_review, manifest_identity
    from harness_ab_experiment import canonical, sha256
    root = Path(root).resolve()
    old, new = root / 'anchor', root / 'candidate'
    old.mkdir(parents=True)
    new.mkdir()
    for target in (old, new):
        (target / 'inputs/materials').mkdir(parents=True)
    order = []
    cases = []
    for round_ in (1, 2):
        for index in range(24):
            case_id = f'S{index:02}-R{round_}'
            cases.append({'case_id': case_id, 'suite_kind': 'agent_final_v1',
                          'prompt': 'Synthetic question ' + case_id, 'materials': [],
                          'preconditions': {'save_policy': 'deny', 'history_messages': [
                              {'role': 'user', 'content': 'Synthetic history only.'}]}})
            for variant in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
                order.append([case_id, variant])
    snapshot = {'copies': [], 'materials': [], 'model_calls': 0}
    for index in range(54):
        pdf, parsed = old / f'inputs/materials/{index}.pdf', old / f'inputs/materials/{index}.json'
        pdf.write_bytes(b'SYNTHETIC PDF ' + str(index).encode())
        parsed.write_text(json.dumps({'page': 1, 'text': f'Synthetic material {index}'}))
        snapshot['materials'].append({'material_id': f'M{index:02}', 'pdf_path': str(pdf),
            'parsed_path': str(parsed), 'pdf_sha256': digest(pdf),
            'parsed_cache_sha256': digest(parsed), 'physical_pages': 1})
        for path in (pdf, parsed):
            snapshot['copies'].append({'copy': str(path), 'source': str(path),
                                      'sha256': digest(path), 'bytes': path.stat().st_size})
    write_json(old / 'inputs/cases.json', {'cases': cases})
    write_json(old / 'inputs/execution-plan.json', {'execution_order': order})
    write_json(old / 'inputs/material-snapshot.json', snapshot)
    write_json(old / 'inputs/model-manifest.json', {'synthetic': {'version': 'unit-only'}})
    sources, variants, loaded = {}, {}, {}
    role_names = ['supervisor', 'researcher_screen', 'researcher_investigate',
                  'researcher_deep_read', 'reviewer_note', 'reviewer_target']
    for variant in ('A', 'B'):
        source = old / 'snapshots' / variant
        path = source / 'backend/research_loop/__init__.py'
        path.parent.mkdir(parents=True)
        path.write_text('# synthetic business ' + variant + '\n')
        runtime = {'backend/research_loop/__init__.py': digest(path)}
        for relative, value in [('config/synthetic.json', '{"flag":true}\n'),
                                 ('backend/uv.lock', 'version = 1\n'),
                                 ('backend/pyproject.toml', '[project]\nname = "synthetic"\n'),
                                 ('backend/migrations/001.sql', 'SELECT 1;\n'),
                                 ('backend/scripts/synthetic_helper.py', '# unchanged helper\n')]:
            item = source / relative
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_text(value)
            runtime[relative] = digest(item)
        sources[variant] = str(source)
        variants[variant] = {'source_files': runtime}
        loaded[variant] = {'assembled_prompt_sha256': {role: sha256(role + variant) for role in role_names},
                           'registry_sha256': {'synthetic': sha256(variant)},
                           'code_fingerprint_format': 'structured-code-v1',
                           'python_runtime': {'implementation': sys.implementation.name,
                                              'version': list(sys.version_info[:3])}}
    limits = {'allowed_cells': order, 'batch_tokens': 100_000_000, 'batch_requests': 1000,
              'batch_yuan': '1000', 'batch_seconds': 1000, 'root_tokens': 4_000_000,
              'root_requests': 32, 'root_seconds': 60, 'max_output_tokens': 65536,
              'carried_usage': {'tokens': 100, 'requests': 10, 'nano_yuan': 1000, 'seconds': 10}}
    total = started_at + limits['batch_seconds'] - 10
    clock = {'round_started_at': started_at, 'round_absolute_deadline': total,
             'model_absolute_deadline': total - 300, 'batch_seconds': 1000,
             'pre_round_carried_seconds': 10}
    manifest = {'experiment_kind': 'prompt_bundle_v1', 'experiment_root': str(old),
        'prepared_input_root': str(old / 'inputs'), 'runtime_root': str(old / 'runtime'),
        'semantic_gate_root': str(old / 'semantic-gates'), 'cases_path': str(old / 'inputs/cases.json'),
        'authorization_path': str(old / 'formal-execution-auth.json'),
        'model_manifest_path': str(old / 'inputs/model-manifest.json'),
        'source_roots': sources, 'prompt_bundle_source_contract': {'variants': variants},
        'loaded_prompt_contract': loaded, 'settings': {'synthetic': True},
        'variant_settings': {'A': {}, 'B': {}},
        'pair_review_policy': {'required': True, 'poll_seconds': .001},
        'final_suite_policy': {'version': 'agent_final_v1', 'semantic_failures': 'record_and_continue'},
        'execution_order': order, 'ledger_limits': limits,
        'material_snapshot_sha256': digest(old / 'inputs/material-snapshot.json'),
        'final_cases_sha256': digest(old / 'inputs/cases.json'),
        'final_plan_sha256': digest(old / 'inputs/execution-plan.json')}
    write_json(old / 'manifest.formal.json', manifest)
    rows = []
    reservations = []
    for position, (case_id, variant) in enumerate(order[:5]):
        root_id = f'old-root-{position}'
        row = {'case_id': case_id, 'variant': variant, 'root_id': root_id,
               'status': 'completed' if position < 4 else 'partial',
               'issues': [] if position < 4 else ['ledger_halted', 'native_response_unknown'],
               'returncode': 0}
        rows.append(row)
        cell = old / 'runtime/cells' / (case_id + '-' + variant)
        write_json(cell / 'result.json', {'case_id': case_id, 'variant': variant,
            'root_id': root_id, 'status': 'executed',
            'started_at': started_at + position, 'finished_at': started_at + position + .5})
        write_json(cell / 'task-view.json', {'task_id': root_id,
            'status': 'completed' if position < 4 else 'partial',
            'prompt': next(c['prompt'] for c in cases if c['case_id'] == case_id),
            'result': {'answer_markdown': 'Synthetic answer ' + root_id}})
        write_json(cell / 'model-calls.json', [])
        write_json(cell / 'actions.json', {})
        reservations.append({'status': 'settled', 'root_id': root_id,
                             'actual_tokens': 3, 'actual_nano': 10})
    for case_id in [order[0][0], order[2][0]]:
        pair_rows = [r for r in rows if r['case_id'] == case_id]
        challenge = prepare_pair_gate(manifest, case_id, pair_rows, manifest_identity(manifest),
                                      clock=lambda: started_at + 20)
        observations = []
        for variant in ('A', 'B'):
            observations.append({'anonymous_id': challenge['anonymous_ids'][variant],
                'answer_sha256': challenge['identity']['answers'][variant]['answer_sha256'],
                'source_sha256s': [], 'whole_answer_quality': 'full' if variant == 'A' else 'partial_usable',
                'core_complete': True, 'critical_core_error': False, 'requested_scope_complete': True,
                'evidence_notes': 'Synthetic independent observation.', 'execution_complete': True,
                'unauthorized_delivery_write': False, 'forged_success': False})
        challenge_sha = sha256(canonical(challenge))
        proofs = []
        for number in (1, 2):
            proof_path = old / 'scoring' / f'{case_id}.reviewer-{number}.json'
            proof = {'reviewer_identity': f'synthetic-reviewer-{number}',
                     'challenge_sha256': challenge_sha, 'paid_model_calls': 0,
                     'reviews': observations, 'blinded': True}
            write_json(proof_path, proof)
            proofs.append({'path': str(proof_path), 'sha256': digest(proof_path),
                           'identity': proof['reviewer_identity']})
        adjudication = old / 'scoring' / f'{case_id}.adjudication.json'
        write_json(adjudication, {'challenge_sha256': challenge_sha,
                                 'paid_model_calls': 0, 'reviews': observations})
        review = {'version': 1, 'challenge_nonce': challenge['nonce'],
            'challenge_sha256': challenge_sha, 'reviewed_at': started_at + 21,
            'reviewer': {'kind': 'agent_assisted', 'identity': 'two synthetic reviewers',
                         'blinded': True, 'full_authorized_source_checked': True, 'paid_model_calls': 0},
            'reviews': observations, 'independent_reviews': proofs,
            'adjudication_source': {'path': str(adjudication), 'sha256': digest(adjudication)}}
        gate = old / 'semantic-gates' / case_id
        write_json(gate / 'review.json', review)
        decision = evaluate_pair_review(manifest, case_id, pair_rows, manifest_identity(manifest),
                                         challenge, review, clock=lambda: started_at + 22)
        write_json(gate / 'decision.json', decision)
    write_json(old / 'scoring/rubric.json', {'full': 'Synthetic rubric only.'})
    write_json(old / 'scoring/STATIC-SEAL.json', {'files': {
        'rubric.json': {'sha256': digest(old / 'scoring/rubric.json')}}})
    ledger = {'started_at': started_at, 'batch_absolute_deadline': total,
        'carried_usage': limits['carried_usage'], 'reservations': reservations,
        'physical_sent_requests': 5, 'actual_known_tokens': 15,
        'actual_known_estimated_nano_yuan': 50,
        'charged_or_reserved': {'tokens': 115, 'requests': 15, 'nano': 1050},
        'halt_reason': 'synthetic presend technical failure', 'first_error': {'root_id': rows[4]['root_id']}}
    write_json(old / 'runtime/ledger-final.json', ledger)
    write_json(old / 'runtime/batch-results.json', {'status': 'stopped', 'rows': rows,
        'not_created': order[5:], 'started_at': started_at, 'execution_seconds': 30,
        'limits': limits, 'physical_requests': 5, 'actual_known_tokens': 15,
        'actual_known_estimated_nano_yuan': 50})
    write_json(old / 'formal-execution-auth.json', {'approved': False, 'allowed_modes': [],
        'status': 'closed_no_resume', 'round_started_at': started_at,
        'round_absolute_deadline': total, 'expires_at': total - 300,
        'manifest_sha256': manifest_identity(manifest)})
    write_json(old / 'formal-preflight/authorization-closed.json', {'status': 'closed',
        'no_resume_or_resend': True, 'original_started_at': started_at, 'absolute_deadline': total})
    write_json(old / 'formal-control/controller-result.json', {'model_deadline': total - 300,
        'original_absolute_total_deadline': total, 'cleanup_all_passed': True,
        'batch_group_cleaned': True, 'wrapper_finished_before_original_total_deadline': True})
    seal = {'files': {str(p.relative_to(old)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                      for p in old.rglob('*') if p.is_file()}}
    write_json(old / 'SEAL.json', seal)
    candidate = deepcopy(manifest)
    for key, value in [('experiment_root', new), ('prepared_input_root', new / 'inputs'),
                       ('runtime_root', new / 'runtime'), ('semantic_gate_root', new / 'semantic-gates'),
                       ('authorization_path', new / 'formal-execution-auth.json'),
                       ('cases_path', new / 'inputs/cases.json'),
                       ('model_manifest_path', new / 'inputs/model-manifest.json')]:
        candidate[key] = str(value)
    for variant in ('A', 'B'):
        shutil.copytree(old / 'snapshots' / variant, new / 'snapshots' / variant)
        candidate['source_roots'][variant] = str(new / 'snapshots' / variant)
    new_snapshot = deepcopy(snapshot)
    for item in new_snapshot['copies']:
        target = new / Path(item['copy']).relative_to(old)
        shutil.copyfile(item['copy'], target)
        item['copy'] = str(target)
    for item in new_snapshot['materials']:
        for key in ('pdf_path', 'parsed_path'):
            item[key] = str(new / Path(item[key]).relative_to(old))
    for filename in ('cases.json', 'execution-plan.json', 'model-manifest.json'):
        shutil.copyfile(old / 'inputs' / filename, new / 'inputs' / filename)
    write_json(new / 'inputs/material-snapshot.json', new_snapshot)
    candidate['material_snapshot_sha256'] = digest(new / 'inputs/material-snapshot.json')
    contract = build_continuation_contract(old)
    candidate['continuation_contract'] = contract
    candidate['ledger_limits']['carried_usage'] = deepcopy(contract['carried_usage'])
    candidate['carried_budget_source'] = {
        'batch_results_path': str(old / 'runtime/batch-results.json'),
        'batch_results_sha256': digest(old / 'runtime/batch-results.json'),
        'ledger_final_path': str(old / 'runtime/ledger-final.json'),
        'ledger_final_sha256': digest(old / 'runtime/ledger-final.json')}
    return {'manifest': candidate, 'anchor_root': old, 'new_root': new,
            'clock': clock, 'order': order, 'rows': rows}


class ContinuationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = make_fixture(Path(self.temp.name))
        self.manifest = self.fixture['manifest']

    def test_fixed_matrix_retains_four_and_retries_only_fifth_in_suffix(self):
        context = validate_continuation(self.manifest)
        self.assertEqual(len(planned_order(self.manifest)), 92)
        self.assertEqual(planned_order(self.manifest), self.fixture['order'][4:])
        self.assertEqual([r['original_position'] for r in retained_rows(self.manifest)], list(range(4)))
        self.assertEqual(context['failed_attempt']['root_id'], 'old-root-4')
        self.assertEqual(context['failed_attempt']['status'], 'partial')
        self.assertTrue(all(r['provenance']['cell_root'].startswith(str(self.fixture['anchor_root']))
                            for r in context['retained_rows']))

    def test_finance_carries_latest_but_shared_time_is_not_added_again(self):
        self.assertEqual(validate_shared_clock_carry(self.manifest),
                         {'tokens': 115, 'requests': 15, 'nano_yuan': 1050, 'seconds': 10})
        self.assertEqual(validate_continuation(self.manifest)['clock'], self.fixture['clock'])

    def test_legacy_absent_key_only(self):
        legacy = {'execution_order': [['old', 'A']]}
        self.assertIsNone(validate_continuation(legacy))
        self.assertEqual(planned_order(legacy), legacy['execution_order'])
        self.assertEqual(retained_rows(legacy), [])
        for value in (None, {}, {'version': 'misspelled'}):
            with self.subTest(value=value), self.assertRaises(PermissionError):
                validate_continuation({**legacy, 'continuation_contract': value})

    def test_contract_missing_extra_or_changed_fields_rejected(self):
        for change in ('extra', 'missing', 'retry', 'grade', 'bool-clock'):
            manifest = deepcopy(self.manifest); c = manifest['continuation_contract']
            if change == 'extra': c['unexpected'] = True
            elif change == 'missing': del c['failed_attempt']
            elif change == 'retry': c['retry_policy']['max_new_attempts'] = 2
            elif change == 'grade': c['retained_rows'][1]['quality'] = 'full'
            else: c['clock']['pre_round_carried_seconds'] = True
            with self.subTest(change=change), self.assertRaises(PermissionError):
                validate_continuation(manifest)

    def test_no_order_truncation_carry_reset_or_old_runtime_reuse(self):
        for change in ('suffix', 'reorder', 'finance', 'double-time', 'old-runtime', 'settings',
                       'ancestor', 'input-root', 'source-root', 'old-auth'):
            m = deepcopy(self.manifest)
            if change == 'suffix': m['execution_order'] = m['execution_order'][4:]
            elif change == 'reorder': m['execution_order'][4:6] = reversed(m['execution_order'][4:6])
            elif change == 'finance': m['ledger_limits']['carried_usage']['tokens'] = 100
            elif change == 'double-time': m['ledger_limits']['carried_usage']['seconds'] += 30
            elif change == 'old-runtime': m['runtime_root'] = str(self.fixture['anchor_root'] / 'runtime')
            elif change == 'ancestor': m['experiment_root'] = str(self.fixture['anchor_root'].parent)
            elif change == 'input-root': m['prepared_input_root'] = str(self.fixture['anchor_root'] / 'inputs')
            elif change == 'source-root': m['source_roots']['A'] = str(self.fixture['anchor_root'] / 'snapshots/A')
            elif change == 'old-auth': m['authorization_path'] = str(self.fixture['anchor_root'] / 'formal-execution-auth.json')
            else: m['settings']['synthetic'] = False
            with self.subTest(change=change), self.assertRaises(PermissionError):
                validate_continuation(m)

    def test_anchor_critical_file_drift_refused(self):
        anchor = self.fixture['anchor_root']
        for relative in ('SEAL.json', 'formal-execution-auth.json', 'runtime/ledger-final.json',
                         'scoring/S00-R1.reviewer-1.json',
                         'runtime/cells/S02-R1-A/model-calls.json'):
            path = anchor / relative; original = path.read_bytes()
            try:
                path.write_bytes(original + b' ')
                with self.subTest(path=relative), self.assertRaises(PermissionError):
                    validate_continuation(self.manifest)
            finally:
                path.write_bytes(original)

    def test_business_prompt_precondition_and_material_changes_refused(self):
        for change in ('business', 'prompt', 'precondition', 'material'):
            m = deepcopy(self.manifest)
            if change == 'prompt':
                m['loaded_prompt_contract']['B']['assembled_prompt_sha256']['supervisor'] = '0' * 64
                with self.assertRaises(PermissionError): validate_continuation(m)
                continue
            if change == 'business': p = Path(m['source_roots']['B']) / 'backend/research_loop/__init__.py'
            elif change == 'precondition': p = Path(m['cases_path'])
            else: p = self.fixture['new_root'] / 'inputs/materials/0.pdf'
            original = p.read_bytes()
            try:
                if change == 'precondition':
                    d = json.loads(p.read_text());d['cases'][0]['preconditions']['save_policy'] = 'allow';write_json(p, d)
                else: p.write_bytes(original + b'changed')
                with self.subTest(change=change), self.assertRaises(PermissionError):
                    validate_continuation(m)
            finally: p.write_bytes(original)

    def test_path_dependent_registry_hashes_can_change_but_runtime_cannot(self):
        m = deepcopy(self.manifest)
        m['loaded_prompt_contract']['A']['registry_sha256']['synthetic'] = '0' * 64
        self.assertEqual(len(planned_order(m)), 92)
        m['loaded_prompt_contract']['A']['python_runtime']['version'][0] += 1
        with self.assertRaises(PermissionError): validate_continuation(m)

    def test_both_variants_cannot_drift_together_in_runtime_conditions(self):
        for relative in ('config/synthetic.json', 'backend/uv.lock', 'backend/pyproject.toml',
                         'backend/migrations/001.sql', 'backend/scripts/synthetic_helper.py'):
            paths = [Path(self.manifest['source_roots'][v]) / relative for v in ('A', 'B')]
            originals = [p.read_bytes() for p in paths]
            try:
                for path in paths:
                    path.write_bytes(b'same unauthorized change in both variants')
                with self.subTest(relative=relative), self.assertRaises(PermissionError):
                    validate_continuation(self.manifest)
            finally:
                for path, content in zip(paths, originals): path.write_bytes(content)

    def test_only_narrow_experiment_script_additions_allowed_and_no_removal(self):
        sources = [Path(self.manifest['source_roots'][v]) for v in ('A', 'B')]
        for source in sources:
            (source / 'backend/scripts/agent_final_continuation.py').write_text('# new experiment only\n')
        self.assertEqual(len(planned_order(self.manifest)), 92)
        common = sources[0] / 'backend/scripts/agent_final_continuation.py'
        common.write_text('# mismatched experiment variant\n')
        with self.assertRaises(PermissionError): validate_continuation(self.manifest)
        common.write_text('# new experiment only\n')
        extra = sources[0] / 'backend/scripts/unapproved_runtime.py'
        extra.write_text('# outside the seven-file allowance\n')
        with self.assertRaises(PermissionError): validate_continuation(self.manifest)
        extra.unlink()
        missing = sources[0] / 'backend/scripts/synthetic_helper.py'
        missing.unlink()
        with self.assertRaises(PermissionError): validate_continuation(self.manifest)

    def test_path_shaped_extra_precondition_is_not_silently_ignored(self):
        path = Path(self.manifest['cases_path'])
        cases = json.loads(path.read_text())
        cases['cases'][0]['preconditions']['pdf_path'] = '/synthetic/unapproved'
        write_json(path, cases)
        with self.assertRaises(PermissionError): validate_continuation(self.manifest)

    def test_old_authorization_open_or_unknown_physical_rejected_even_when_resealed(self):
        for relative, mutate in [
                ('formal-execution-auth.json', lambda d: d.update(approved=True)),
                ('runtime/ledger-final.json', lambda d: d['reservations'][0].update(status='unknown'))]:
            p = self.fixture['anchor_root'] / relative; old = p.read_bytes()
            seal = self.fixture['anchor_root'] / 'SEAL.json'; old_seal = seal.read_bytes()
            try:
                data = json.loads(p.read_text());mutate(data);write_json(p, data)
                index = json.loads(seal.read_text());index['files'][relative]['sha256'] = digest(p);write_json(seal, index)
                with self.subTest(relative=relative), self.assertRaises(PermissionError):
                    build_continuation_contract(self.fixture['anchor_root'])
            finally:
                p.write_bytes(old);seal.write_bytes(old_seal)


@unittest.skipUnless(os.environ.get('AGENT_FINAL_CONTINUATION_ANCHOR_ROOT'),
                     'real immutable anchor supplied only by an explicit local offline check')
class ReadonlyRealAnchorTests(unittest.TestCase):
    def test_explicit_real_anchor_build_has_four_plus_one_and92(self):
        root = Path(os.environ['AGENT_FINAL_CONTINUATION_ANCHOR_ROOT']).resolve()
        before = digest(root / 'SEAL.json')
        contract = build_continuation_contract(root)
        self.assertEqual(len(contract['retained_rows']), 4)
        self.assertEqual(len(contract['new_execution_order']), 92)
        self.assertEqual(contract['failed_attempt']['original_position'], 4)
        self.assertEqual(digest(root / 'SEAL.json'), before)

    def test_real_cross_path_cold_identity_and_shared_clock_validation(self):
        from prompt_bundle_ab import prepare
        anchor = Path(os.environ['AGENT_FINAL_CONTINUATION_ANCHOR_ROOT']).resolve()
        before = digest(anchor / 'SEAL.json')
        original = json.loads((anchor / 'manifest.formal.json').read_text())
        contract = build_continuation_contract(anchor)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp).resolve()
            roots = {}
            for variant in ('A', 'B'):
                source = target / 'snapshots' / variant
                shutil.copytree(original['source_roots'][variant], source)
                roots[variant] = str(source)
            shutil.copytree(anchor / 'inputs', target / 'inputs')
            material_path = target / 'inputs/material-snapshot.json'
            snapshot = json.loads(material_path.read_text())
            remap = {}
            for item in snapshot['copies']:
                old_path = item['copy']
                item['copy'] = str(target / Path(old_path).relative_to(anchor))
                remap[old_path] = item['copy']
            for item in snapshot['materials']:
                for key in ('pdf_path', 'parsed_path'):
                    item[key] = remap[item[key]]
            write_json(material_path, snapshot)
            case_path = target / 'inputs/cases.json'
            cases = json.loads(case_path.read_text())
            for case in cases['cases']:
                for item in [*case.get('materials', []), *([case['material']] if case.get('material') else [])]:
                    item['pdf_path'] = remap[item['pdf_path']]
            write_json(case_path, cases)
            cold = prepare(roots, target / 'cold')
            self.assertTrue(any(cold['loaded_prompt_contract'][v]['registry_sha256']
                                != original['loaded_prompt_contract'][v]['registry_sha256']
                                for v in ('A', 'B')))
            manifest = {**original, **cold, 'experiment_root': str(target),
                'runtime_root': str(target / 'runtime'), 'semantic_gate_root': str(target / 'semantic-gates'),
                'prepared_input_root': str(target / 'inputs'), 'cases_path': str(case_path),
                'model_manifest_path': str(target / 'inputs/model-manifest.json'),
                'authorization_path': str(target / 'formal-execution-auth.json'),
                'material_snapshot_sha256': digest(material_path), 'final_cases_sha256': digest(case_path),
                'final_plan_sha256': digest(target / 'inputs/execution-plan.json'),
                'continuation_contract': contract,
                'ledger_limits': {**original['ledger_limits'], 'carried_usage': contract['carried_usage']},
                'carried_budget_source': {
                    'batch_results_path': str(anchor / 'runtime/batch-results.json'),
                    'batch_results_sha256': contract['anchors']['runtime/batch-results.json'],
                    'ledger_final_path': str(anchor / 'runtime/ledger-final.json'),
                    'ledger_final_sha256': contract['anchors']['runtime/ledger-final.json']}}
            context = validate_continuation(manifest)
            self.assertEqual(context['clock']['round_started_at'], contract['clock']['round_started_at'])
            self.assertEqual(context['carried_usage']['seconds'],
                             original['ledger_limits']['carried_usage']['seconds'])
            self.assertEqual(len(context['new_execution_order']), 92)
            for variant in ('A', 'B'):
                probe = json.loads((target / 'cold' / variant / 'probe.json').read_text())
                self.assertEqual((probe['generation_calls'], probe['rag_runs'], probe['sqlite_connections']),
                                 (0, 0, []))
        self.assertEqual(digest(anchor / 'SEAL.json'), before)


if __name__ == '__main__':
    unittest.main()
