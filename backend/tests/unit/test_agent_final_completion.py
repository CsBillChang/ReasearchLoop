"""补测合同的原证据、奇数前缀、财务续跑与只读边界。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_final_completion as completion
from test_agent_final_continuation import make_fixture, write_json, digest


def synthetic_completion_fixture(root):
    """仅复用旧输入/源码夹具；锚清单派生使用独立合成context。"""
    fixture = make_fixture(root)
    n = fixture['anchor_root']
    r2 = Path(root) / 'closed-second-anchor'
    r2.mkdir()
    original = json.loads((n / 'manifest.formal.json').read_text())
    latest = deepcopy(original)
    latest['experiment_root'] = str(r2)
    write_json(r2 / 'manifest.formal.json', latest)
    rows = []
    for i, (case, variant) in enumerate(fixture['order'][:11]):
        source = n if i < 4 else r2
        cell = source / 'runtime/cells' / (case + '-' + variant)
        rows.append({'case_id': case, 'variant': variant, 'root_id': 'retained-' + str(i),
            'status': 'partial' if i == 7 else 'completed', 'issues': [], 'original_position': i,
            'retained': True, 'quality': None if i == 10 else 'partial_usable',
            'source_manifest_path': str(source / 'manifest.formal.json'),
            'source_manifest_sha256': digest(source / 'manifest.formal.json'),
            'physical_receipts_verified': True, 'native_status_counts': {'succeeded': 1},
            'physical_status_counts': {'settled': 1}, 'ledger_root': str(source),
            'ledger_path': str(source / 'runtime/budget.sqlite3'),
            'provenance': {'cell_root': str(cell), **({'grade_decision_path': str(source / 'old-grade.json')}
                                                     if i < 10 else {})}})
    base = {'anchor_root': str(r2), 'anchor_seals': {str(n): 'a' * 64, str(r2): 'b' * 64},
        'anchors': {'SEAL.json': 'b' * 64, 'runtime/budget.sqlite3': 'c' * 64},
        'retained_rows': rows, 'retained_count': 11, 'new_execution_order': fixture['order'][11:],
        'full_execution_order': fixture['order'],
        'technical_attempts': [{'original_position': 4, 'status': 'partial'},
                               {'original_position': 11, 'status': 'running',
                                'attempt_outcome': 'interrupted_at_original_model_cutoff'}],
        'carried_usage': {'tokens': 220, 'requests': 23, 'nano_yuan': 1900, 'seconds': 10},
        'closed_ledger_roots': [str(n), str(r2)],
        'pending_reservations': [{'reservation_id': 'unknown-a', 'status': 'sent'},
                                 {'reservation_id': 'unknown-b', 'status': 'sent'}],
        'final_safety_proof': {'original_automatic_gate_passed': False},
        'original_clock': fixture['clock'], 'anchor_manifest': original,
        'latest_anchor_manifest': latest,
        'old_cases': json.loads((n / 'inputs/cases.json').read_text()),
        'old_materials': json.loads((n / 'inputs/material-snapshot.json').read_text())}
    receipt = fixture['new_root'] / 'audit/user-scope.json'
    write_json(receipt, {'user_request': 'Finish every task; cancel the global time limit.',
        'cancel_global_time_limit': True, 'authorized_by_user': True})
    timing = {'policy': 'financial_only', 'global_deadline': None, 'model_deadline': None,
        'authorized_by_user': True, 'scope_receipt_path': str(receipt),
        'scope_receipt_sha256': digest(receipt)}
    manifest = fixture['manifest']
    del manifest['continuation_contract']
    with patch.object(completion, '_anchor', return_value=base):
        contract = completion.build_completion_contract(r2, timing)
    manifest['completion_contract'] = contract
    manifest['evaluation_policy'] = deepcopy(completion.EVALUATION_POLICY)
    manifest['ledger_limits'] = {**deepcopy(original['ledger_limits']),
        'carried_usage': deepcopy(base['carried_usage']), 'enforce_batch_seconds': False}
    manifest['carried_budget_source'] = {'version': completion.VERSION, 'anchor_root': str(r2),
        'ledger_path': str(r2 / 'runtime/budget.sqlite3'),
        'ledger_sha256': base['anchors']['runtime/budget.sqlite3'], 'seal_sha256': 'b' * 64}
    return {**fixture, 'base': base, 'timing': timing, 'receipt': receipt}


class CompletionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = synthetic_completion_fixture(Path(self.temp.name))
        self.manifest = self.fixture['manifest']
        self.anchor_patch = patch.object(completion, '_anchor', return_value=self.fixture['base'])
        self.anchor_patch.start()
        self.addCleanup(self.anchor_patch.stop)

    def test_exact_odd_prefix_retains_partial_and_two_attempts(self):
        context = completion.validate_completion(self.manifest)
        self.assertEqual(context['retained_count'], 11)
        self.assertEqual(len(context['new_execution_order']), 85)
        self.assertEqual(context['new_execution_order'], self.fixture['order'][11:])
        self.assertEqual(context['retained_rows'][7]['status'], 'partial')
        self.assertEqual(context['retained_rows'][7]['quality'], 'partial_usable')
        self.assertIsNone(context['retained_rows'][10]['quality'])
        self.assertEqual([x['original_position'] for x in completion.technical_attempts(context)], [4, 11])
        self.assertFalse(context['final_safety_proof']['original_automatic_gate_passed'])
        self.assertEqual(context['retry_policy']['original_position'], 11)
        self.assertEqual(context['retry_policy']['max_new_attempts'], 1)
        self.assertEqual(context['pending_reservations'], self.fixture['base']['pending_reservations'])

    def test_cancelled_global_time_keeps_old_clock_and_root_financial_caps(self):
        context = completion.validate_completion(self.manifest)
        self.assertEqual(context['clock']['policy'], 'financial_only')
        self.assertIsNone(context['clock']['round_absolute_deadline'])
        self.assertIsNone(context['clock']['model_absolute_deadline'])
        self.assertEqual(context['clock']['round_started_at'], self.fixture['clock']['round_started_at'])
        self.assertEqual(context['original_clock'], self.fixture['clock'])
        self.assertEqual(context['clock']['pre_round_carried_seconds'], 10)
        self.assertFalse(self.manifest['ledger_limits']['enforce_batch_seconds'])
        self.assertEqual(self.manifest['ledger_limits']['root_seconds'], 60)
        self.assertEqual(context['evaluation_policy'], completion.EVALUATION_POLICY)

    def test_helpers_keep_each_original_owner_and_bridge_uses_current_pair(self):
        case, variant = self.fixture['order'][10]
        cell = completion.retained_cell(self.manifest, case, variant)
        self.assertEqual(cell['source_manifest'], self.fixture['base']['latest_anchor_manifest'])
        self.assertTrue(cell['physical_receipts_verified'])
        self.assertEqual(cell['native_status_counts'], {'succeeded': 1})
        self.assertEqual(cell['physical_status_counts'], {'settled': 1})
        self.assertIsNone(cell['grade_decision_path'])
        for offset in (0, 2):
            self.assertEqual(completion.prior_pair_manifest(self.manifest, offset),
                             self.fixture['base']['anchor_manifest'])
        for offset in (4, 6, 8):
            self.assertEqual(completion.prior_pair_manifest(self.manifest, offset),
                             self.fixture['base']['latest_anchor_manifest'])
        for offset in (10, 12, 94):
            self.assertIsNone(completion.prior_pair_manifest(self.manifest, offset))
        self.assertIsNone(completion.retained_cell(self.manifest, *self.fixture['order'][11]))
        for bad in (-2, 1, 11, True, '10'):
            with self.subTest(offset=bad), self.assertRaises(PermissionError):
                completion.prior_pair_manifest(self.manifest, bad)

    def test_missing_key_legacy_only_and_dual_or_invalid_v2_fail_closed(self):
        self.assertIsNone(completion.validate_completion({'execution_order': []}))
        self.assertIsNone(completion.retained_cell({}, 'none', 'A'))
        self.assertEqual(completion.technical_attempts({}), [])
        for bad in (None, {}, {'version': 'misspelled'}):
            with self.subTest(bad=bad), self.assertRaises(PermissionError):
                completion.validate_completion({'completion_contract': bad})
        manifest = deepcopy(self.manifest)
        manifest['continuation_contract'] = {}
        with self.assertRaises(PermissionError):
            completion.validate_completion(manifest)

    def test_declared_contract_cannot_reset_budget_clock_quality_or_suffix(self):
        changes = ('carry', 'pending', 'retry', 'retained-grade', 'retained-state', 'technical',
                   'suffix', 'order', 'caps', 'global-time', 'root-time', '44', 'source-proof', 'kind')
        for change in changes:
            m = deepcopy(self.manifest)
            c = m['completion_contract']
            if change == 'carry': c['carried_usage']['seconds'] += 1
            elif change == 'pending': c['pending_reservations'][0]['status'] = 'not_sent'
            elif change == 'retry': c['retry_policy']['max_new_attempts'] = 2
            elif change == 'retained-grade': c['retained_rows'][7]['quality'] = 'full'
            elif change == 'retained-state': c['retained_rows'][7]['status'] = 'completed'
            elif change == 'technical': c['technical_attempts'].pop()
            elif change == 'suffix': c['new_execution_order'] = c['new_execution_order'][1:]
            elif change == 'order': m['execution_order'] = m['execution_order'][11:]
            elif change == 'caps': m['ledger_limits']['batch_tokens'] += 1
            elif change == 'global-time': m['ledger_limits']['enforce_batch_seconds'] = True
            elif change == 'root-time': m['ledger_limits']['root_seconds'] += 1
            elif change == '44': m['evaluation_policy']['minimum_B_full'] = 44
            elif change == 'source-proof': m['carried_budget_source']['ledger_sha256'] = 'd' * 64
            else: m['experiment_kind'] = 'unreviewed'
            with self.subTest(change=change), self.assertRaises(PermissionError):
                completion.validate_completion(m)

    def test_pending_timing_never_releases_live_or_real_execution(self):
        m = deepcopy(self.manifest)
        m['completion_contract'] = completion.build_completion_contract(self.fixture['base']['anchor_root'], None)
        m['real_execution_released'] = m['live_qualification_released'] = False
        self.assertEqual(completion.validate_completion(m)['clock']['policy'], 'pending')
        for key in ('real_execution_released', 'live_qualification_released'):
            changed = deepcopy(m); changed[key] = True
            with self.subTest(key=key), self.assertRaises(PermissionError):
                completion.validate_completion(changed)

    def test_time_cancellation_requires_explicit_user_receipt_and_exact_hash(self):
        for change in ('wrong-hash', 'finite-time', 'extra-time', 'approval', 'scope'):
            timing = deepcopy(self.fixture['timing'])
            if change == 'wrong-hash': timing['scope_receipt_sha256'] = '0' * 64
            elif change == 'finite-time': timing['global_deadline'] = 10**12
            elif change == 'extra-time': timing['expires_at'] = 10**12
            elif change == 'approval': timing['authorized_by_user'] = False
            else:
                path = self.fixture['new_root'] / 'audit/implicit-scope.json'
                write_json(path, {'user_request': 'Proceed', 'authorized_by_user': True})
                timing['scope_receipt_path'], timing['scope_receipt_sha256'] = str(path), digest(path)
            with self.subTest(change=change), self.assertRaises(PermissionError):
                completion.build_completion_contract(self.fixture['base']['anchor_root'], timing)
        before = self.fixture['receipt'].read_bytes()
        try:
            self.fixture['receipt'].write_bytes(before + b' ')
            with self.assertRaises(PermissionError):
                completion.validate_completion(self.manifest)
        finally:
            self.fixture['receipt'].write_bytes(before)

    def test_owned_geometry_cannot_reuse_closed_roots_or_authorization(self):
        for change in ('ancestor', 'runtime', 'inputs', 'source', 'authorization'):
            m = deepcopy(self.manifest); old = self.fixture['anchor_root']
            if change == 'ancestor': m['experiment_root'] = str(old.parent)
            elif change == 'runtime': m['runtime_root'] = str(old / 'runtime')
            elif change == 'inputs': m['prepared_input_root'] = str(old / 'inputs')
            elif change == 'source': m['source_roots']['A'] = str(old / 'snapshots/A')
            else: m['authorization_path'] = str(old / 'formal-execution-auth.json')
            with self.subTest(change=change), self.assertRaises(PermissionError):
                completion.validate_completion(m)

    def test_frozen_business_prompts_model_and_materials_are_rechecked(self):
        for change in ('business', 'prompt', 'model', 'material', 'unknown-script', 'common-overlay'):
            m = deepcopy(self.manifest)
            if change == 'prompt':
                m['loaded_prompt_contract']['B']['assembled_prompt_sha256']['supervisor'] = '0' * 64
                with self.assertRaises(PermissionError): completion.validate_completion(m)
                continue
            if change == 'business': p = Path(m['source_roots']['B']) / 'backend/research_loop/__init__.py'
            elif change == 'model': p = Path(m['model_manifest_path'])
            elif change == 'material': p = self.fixture['new_root'] / 'inputs/materials/0.pdf'
            elif change == 'unknown-script': p = Path(m['source_roots']['A']) / 'backend/scripts/unapproved.py'
            else: p = Path(m['source_roots']['A']) / 'backend/scripts/harness_ab_budget.py'
            before = p.read_bytes() if p.exists() else None
            try:
                p.write_bytes((before or b'') + b'changed')
                if change == 'common-overlay':
                    m['prompt_bundle_source_contract']['variants']['A']['source_files'][str(p.relative_to(Path(m['source_roots']['A'])))] = digest(p)
                with self.subTest(change=change), self.assertRaises(PermissionError):
                    completion.validate_completion(m)
            finally:
                p.write_bytes(before) if before is not None else p.unlink()

    def test_exact_common_experiment_overlays_allowed(self):
        for variant in ('A', 'B'):
            for name in ('harness_ab_budget.py', 'agent_final_completion.py', 'harness_financial_process.py'):
                relative = 'backend/scripts/' + name
                path = Path(self.manifest['source_roots'][variant]) / relative
                path.write_text('# synthetic identical experiment overlay\n')
                self.manifest['prompt_bundle_source_contract']['variants'][variant]['source_files'][relative] = digest(path)
        self.assertEqual(completion.validate_completion(self.manifest)['retained_count'], 11)


class CompletionReadonlyEvidenceTests(unittest.TestCase):
    def test_specific_failed_automatic_gate_requires_independent_final_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            controller = {'batch_group_cleaned': False, 'cleanup_all_passed': True,
                'status': 'model_hard_deadline', 'wrapper_finished_before_original_total_deadline': True}
            evidence = {
                'formal-control/controller-result.json': controller,
                'audit/readonly-post-run.json': {'status': 'readonly_observation'},
                'audit/post-run-protection.json': {'status': 'failed', 'checks_passed': 3, 'checks_total': 4,
                    'errors': [{'check': 'controller cleanup complete', 'details': None}],
                    'actual_remaining_owned_group_members': [], 'actual_remaining_scoped_processes': [],
                    'audit_finished_before_original_total_deadline': True},
                'audit/final-process-group-readback.json': {'status': 'passed',
                    'initial_controller_result_preserved': True, 'controller_initial_batch_group_cleaned': False,
                    'alive_group_members_after': [], 'before_original_total_deadline': True},
                'formal-preflight/vector-close.json': {'status': 'passed', 'identity': {'running': False}},
                'formal-preflight/authorization-closed.json': {'status': 'closed', 'no_resume_or_resend': True},
                'scoring/metadata-watch-closed.json': {'status': 'closed', 'model_calls': 0}}
            for name, value in evidence.items(): write_json(root / name, value)
            supplement = {'status': 'passed_final_safety_state_with_original_automatic_gate_failure',
                'original_automatic_batch_group_gate_passed': False,
                'final_live_owned_process_group_members': [], 'final_live_scoped_controller_model_metadata_processes': [],
                'sources': {name: digest(root / name) for name in evidence}}
            for name in ('original_controller_record_unchanged', 'original_failed_protection_report_unchanged',
                         'automatic_end_to_end_closure_gate_not_rewritten', 'new6461_verified_stopped',
                         'formal_authorization_verified_closed', 'original5_containers_unchanged',
                         'old6441_6451_verified_stopped', 'protected_hashes_and_seals_frozen_sources_passed',
                         'pending_not_reclassified_or_released', 'before_original_total_deadline'):
                supplement[name] = True
            for name in ('additional_model_calls', 'additional_kills', 'service_operations',
                         'Ledger_instantiations', 'ledger_mutations'):
                supplement[name] = 0
            evidence['audit/post-run-safety-supplement.json'] = supplement

            def load(name, parse=True):
                return deepcopy(evidence[name]) if parse else None

            self.assertFalse(completion._safety(root, load, controller)['original_automatic_batch_group_gate_passed'])
            for change in ('rewritten-false', 'second-failure', 'missing-sources', 'pending-released', 'live-vector', 'new-call'):
                values = deepcopy(evidence); ctl = deepcopy(controller)
                if change == 'rewritten-false': ctl['batch_group_cleaned'] = True
                elif change == 'second-failure': values['audit/post-run-protection.json']['errors'].append({'check': 'other'})
                elif change == 'missing-sources': values['audit/post-run-safety-supplement.json']['sources'] = {}
                elif change == 'pending-released': values['audit/post-run-safety-supplement.json']['pending_not_reclassified_or_released'] = False
                elif change == 'live-vector': values['formal-preflight/vector-close.json']['identity']['running'] = True
                else: values['audit/post-run-safety-supplement.json']['additional_model_calls'] = 1
                with self.subTest(change=change), self.assertRaises(PermissionError):
                    completion._safety(root, lambda name, parse=True: deepcopy(values[name]) if parse else None, ctl)

    def test_physical_receipt_identity_and_usage_are_not_self_asserted(self):
        usage = {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}
        native = [{'status': 'succeeded', 'sent_state': 'sent', 'request_id': 'provider-1',
                   'response_json': json.dumps({'id': 'provider-1'}), 'usage_json': json.dumps(usage)}]
        physical = [{'root_id': 'root-1', 'status': 'settled', 'response_json': json.dumps({'response_id': 'provider-1'}),
                     'usage_json': json.dumps(usage), 'actual_input': 2, 'actual_output': 3, 'actual_tokens': 5}]
        proof = completion._receipts('root-1', native, physical)
        self.assertTrue(proof['physical_receipts_verified'])
        for change in ('native-unknown', 'pending', 'foreign-root', 'different-id', 'usage', 'duplicate', 'empty'):
            n, p = deepcopy(native), deepcopy(physical)
            if change == 'native-unknown': n[0]['status'] = 'unknown'
            elif change == 'pending': p[0]['status'] = 'sent'
            elif change == 'foreign-root': p[0]['root_id'] = 'foreign'
            elif change == 'different-id': p[0]['response_json'] = json.dumps({'response_id': 'provider-2'})
            elif change == 'usage': p[0]['actual_tokens'] = 4
            elif change == 'duplicate': n *= 2; p *= 2
            else: n = []; p = []
            with self.subTest(change=change), self.assertRaises(PermissionError):
                completion._receipts('root-1', n, p)

    def test_full_seal_rejects_added_changed_or_redirected_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); path = root / 'evidence.json'
            write_json(path, {'synthetic': True})
            write_json(root / 'SEAL.json', {'files': {'evidence.json': {'sha256': digest(path), 'bytes': path.stat().st_size}}})
            seal_sha = digest(root / 'SEAL.json')
            (root / 'SEAL.sha256').write_text(seal_sha + '  SEAL.json\n')
            write_json(root / 'SEAL-READBACK.json', {'status': 'passed', 'seal_sha256': seal_sha})
            self.assertEqual(completion._sealed(root)[2], seal_sha)
            before = path.read_bytes()
            try:
                path.write_bytes(before + b' ')
                with self.assertRaises(PermissionError): completion._sealed(root)
            finally: path.write_bytes(before)
            extra = root / 'unsealed.json'; extra.write_text('{}')
            with self.assertRaises(PermissionError): completion._sealed(root)
            extra.unlink(); extra.symlink_to(path)
            with self.assertRaises(PermissionError): completion._sealed(root)

    def test_sqlite_anchor_is_readonly_and_uncheckpointed_wal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); path = root / 'runtime/budget.sqlite3'; path.parent.mkdir()
            with sqlite3.connect(path) as db:
                db.executescript('CREATE TABLE metadata(key TEXT,value TEXT); CREATE TABLE roots(root_id TEXT);'
                                 'CREATE TABLE reservations(reservation_id TEXT,created_at REAL);')
                db.execute('INSERT INTO metadata VALUES (?,?)', ('config', '{}'))
            before = digest(path)
            calls = []
            context = completion._ledger(root, lambda relative, parse=True: calls.append((relative, parse)))
            self.assertEqual(context['metadata'], {'config': {}})
            self.assertEqual(digest(path), before)
            self.assertEqual(calls, [('runtime/budget.sqlite3', False)])
            path.with_name(path.name + '-wal').write_bytes(b'uncheckpointed')
            with self.assertRaises(PermissionError): completion._ledger(root, lambda *args: None)


@unittest.skipUnless(os.environ.get('AGENT_FINAL_COMPLETION_ANCHOR_ROOT'), 'explicit readonly sealed anchor not supplied')
class CompletionRealAnchorReadonlyTests(unittest.TestCase):
    def test_real_sealed_anchor_derives_without_model_or_ledger_mutation(self):
        root = Path(os.environ['AGENT_FINAL_COMPLETION_ANCHOR_ROOT'])
        paths = [root / 'SEAL.json', root / 'runtime/budget.sqlite3', root / 'formal-execution-auth.json',
                 root / 'formal-control/controller-result.json']
        before = {str(p): digest(p) for p in paths}
        contract = completion.build_completion_contract(root, None)
        self.assertEqual(contract['retained_count'], 11)
        self.assertEqual(len(contract['full_execution_order']), 96)
        self.assertEqual(contract['new_execution_order'], contract['full_execution_order'][11:])
        self.assertEqual(len(contract['new_execution_order']), 85)
        self.assertEqual([x['original_position'] for x in contract['technical_attempts']], [4, 11])
        self.assertEqual(len(contract['pending_reservations']), 2)
        self.assertTrue(all(x['status'] == 'sent' and x['actual_tokens'] is None for x in contract['pending_reservations']))
        self.assertEqual(contract['retained_rows'][7]['status'], 'partial')
        self.assertEqual(contract['retained_rows'][7]['quality'], 'partial_usable')
        self.assertTrue(all(x['physical_receipts_verified'] for x in contract['retained_rows']))
        self.assertFalse(contract['final_safety_proof']['original_automatic_gate_passed'])
        self.assertEqual(contract['clock']['policy'], 'pending')
        self.assertEqual(before, {str(p): digest(p) for p in paths})


if __name__ == '__main__':
    unittest.main()
