"""发送资格绑定与取消收尾的零模型合同；授权正例仅是合成夹具。"""
import hashlib
import json
from pathlib import Path
from copy import deepcopy
import tempfile
import unittest

from experiments.mechanism_pilot.contract import RunContract, authorize_execution, finalize_reservations, settings_profile
from experiments.mechanism_pilot.budget import UNBOUNDED_LIMITS
from research_loop.config import Settings
from experiments.mechanism_pilot.resources import runtime_namespace


class ContractTests(unittest.TestCase):
    """每例独立文件，不加载本机配置或建立网络传输。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'synthetic-source.txt'
        source.write_text('frozen synthetic material')
        self.manifest = {'files': [{'path': str(source),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}],
            'runtime_root': str(self.root / 'runtime'), 'model': 'deepseek-flash',
            'context_window_tokens': 1000000,
            'phases': {'R': {'max_output_tokens': 8192, 'input_token_limit': 1048576}},
            'limits': {'tokens': 3000000, 'requests': 96, 'nano_yuan': 12000000000,
                       'admission_seconds': 2400, 'total_seconds': 2700},
            'egress': ['role_prompts', 'frozen_user_questions', 'frozen_paper_pages', 'tool_results']}
        self.auth_path = self.root / 'authorization.json'
        self.contract = RunContract(self.manifest, self.auth_path, clock=lambda: 1000)
        case = self.root / 'runtime' / 'R01'
        self.settings = Settings(project_root=case, data_dir=case/'data', output_dir=case/'output',
            database_url='sqlite:///'+str(case/'runtime.sqlite3'),
            deepseek_model='deepseek-flash', deepseek_api_key=None,
            deepseek_base_url='https://fixture.invalid', context_window_tokens=1000000,
            context_output_reserve=8192, model_max_output_tokens=8192,
            retrieval_namespace=runtime_namespace(case),
            web_search_enabled=False, web_fetch_enabled=False)
        self.manifest['settings_profiles'] = {'R': settings_profile(self.settings)}
        self.contract = RunContract(self.manifest, self.auth_path, clock=lambda: 1000)

    def grant(self, **updates):
        value = {'status': 'approved', 'scope': 'mechanism-14',
            'approval_reference': 'synthetic approval only, no external send',
            'freeze_sha256': self.contract.freeze_sha256, 'started_at': 900,
            'limits': self.manifest['limits'], 'egress': self.manifest['egress']}
        value.update(updates)
        self.auth_path.write_text(json.dumps(value))

    def real_settings(self):
        return self.settings.model_copy(update={'deepseek_api_key': 'synthetic-key',
            'deepseek_base_url': 'https://api.deepseek.com'})

    def test_draft_or_missing_authorization_rejects_before_side_effects(self):
        settings = self.real_settings()
        with self.assertRaisesRegex(PermissionError, 'AUTHORIZATION_REQUIRED'):
            self.contract.validate(settings, settings.project_root, phase='R', real=True)
        self.grant(status='draft')
        with self.assertRaisesRegex(PermissionError, 'AUTHORIZATION_NOT_MATCHED'):
            self.contract.validate(settings, settings.project_root, phase='R', real=True)
        self.assertFalse(settings.project_root.exists())

    def test_approved_synthetic_contract_and_fixed_window(self):
        self.grant()
        settings = self.real_settings()
        callback = self.contract.authorizer(settings, settings.project_root, phase='R', real=True)
        grant = authorize_execution(settings, 'real', callback, self.contract.freeze_sha256)
        self.assertTrue(grant['allow_real_model'])
        self.assertEqual(grant['hard_deadline'], 3600)
        self.grant(started_at=1001)
        with self.assertRaisesRegex(PermissionError, 'TIME_BOUND'):
            callback()
        self.grant(started_at=-1400)
        with self.assertRaisesRegex(PermissionError, 'TIME_BOUND'):
            callback()

    def test_source_drift_and_permission_config_changes_are_rejected(self):
        self.assertFalse(self.contract.validate(self.settings, self.settings.project_root, phase='R')['allow_real_model'])
        for update in ({'database_url': 'sqlite:////unrelated/deploy.db'},
                       {'output_dir': self.root/'outside'}, {'web_fetch_enabled': True},
                       {'retrieval_route': 'hybrid_rrf'}, {'context_window_tokens': 128000},
                       {'model_max_output_tokens': 4096}):
            with self.subTest(update=update), self.assertRaises(PermissionError):
                self.contract.validate(self.settings.model_copy(update=update), self.settings.project_root, phase='R')
        Path(self.manifest['files'][0]['path']).write_text('changed')
        with self.assertRaisesRegex(PermissionError, 'FROZEN_FILE_DRIFT'):
            self.contract.validate(self.settings, self.settings.project_root, phase='R')

    def test_limits_scope_and_freeze_cannot_be_changed(self):
        for update in ({'limits': {}}, {'egress': ['all_files']},
                       {'freeze_sha256': '0'*64}, {'approval_reference': ''}):
            self.grant(**update)
            with self.subTest(update=update), self.assertRaisesRegex(PermissionError, 'NOT_MATCHED'):
                self.contract.validate(self.real_settings(), self.settings.project_root, phase='R', real=True)

    def test_explicit_unbounded_authorization_preserves_start_without_expiry(self):
        self.manifest['limits']=deepcopy(UNBOUNDED_LIMITS)
        self.contract=RunContract(self.manifest,self.auth_path,clock=lambda:100_000)
        self.grant(started_at=900,budget_mode='unbounded')
        settings=self.real_settings()
        result=self.contract.validate(settings,settings.project_root,phase='R',real=True)
        self.assertEqual(result['started_at'],900)
        self.assertIsNone(result['hard_deadline'])
        self.assertEqual(result['budget_mode'],'unbounded')
        for changed in ({'budget_mode':'bounded'}, {'limits':{'tokens':None}},
                        {'started_at':float('nan')}, {'started_at':float('inf')}, {'started_at':100_001}):
            value={'status':'approved','scope':'mechanism-14','approval_reference':'synthetic',
                'freeze_sha256':self.contract.freeze_sha256,'started_at':900,
                'limits':self.manifest['limits'],'egress':self.manifest['egress'],'budget_mode':'unbounded',**changed}
            self.auth_path.write_text(json.dumps(value))
            with self.subTest(changed=changed),self.assertRaises(PermissionError):
                self.contract.validate(settings,settings.project_root,phase='R',real=True)
        self.grant(started_at=900)
        with self.assertRaisesRegex(PermissionError,'NOT_MATCHED'):
            self.contract.validate(settings,settings.project_root,phase='R',real=True)

    def test_finalize_requires_confirmed_exit_and_preserves_sent_reservation(self):
        class Ledger:
            def __init__(self): self.events = []
            def snapshot(self):
                return {'reservations': [{'reservation_id': 'a', 'status': 'reserved', 'root_id': 'own'},
                    {'reservation_id': 'b', 'status': 'sent', 'root_id': 'own'},
                    {'reservation_id': 'c', 'status': 'settled', 'root_id': 'own'},
                    {'reservation_id': 'other', 'status': 'reserved', 'root_id': 'another-root'}]}
            def not_sent(self, identity, reason): self.events.append(('not_sent', identity))
            def unknown(self, identity, reason): self.events.append(('unknown', identity))
        ledger = Ledger()
        with self.assertRaises(PermissionError): finalize_reservations(ledger, {'process_group_stopped': False})
        self.assertEqual(ledger.events, [])
        finalize_reservations(ledger, {'process_group_stopped': True, 'callback_threads_alive': [], 'owned_root_id': 'own'})
        self.assertEqual(ledger.events, [('not_sent', 'a'), ('unknown', 'b')])
