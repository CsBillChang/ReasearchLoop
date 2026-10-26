"""在独立替身子进程核验MPS实验文件与资源边界，不导入模型库。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class MpsExperimentGuardTest(unittest.TestCase):
    """每项使用新进程，避免不可卸载审计钩子污染测试进程。"""

    def run_child(self, body: str) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for name in ('output', 'cache', 'tmp', 'inputs', 'outside'):
                (root / name).mkdir()
            (root / 'inputs/frozen.txt').write_text('original')
            (root / 'inputs/.env').write_text('private-placeholder')
            script = '''
import json, os, socket, sqlite3, subprocess, sys
from pathlib import Path
from scripts.mps_experiment_guard import install_guards, BoundaryViolation, write_json_new
base = Path(sys.argv[1])
output, cache, tmp, inputs, outside = [base / n for n in ('output','cache','tmp','inputs','outside')]
counts = install_guards(output, [inputs], [cache, tmp])
def rejected(action):
    try:
        action()
    except BoundaryViolation as exc:
        return str(exc)
    raise AssertionError('operation was not rejected')
'''
            script += textwrap.dedent(body)
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run([sys.executable, '-c', script, str(root)],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual((root / 'inputs/frozen.txt').read_text(), 'original')
            return json.loads(result.stdout)

    def test_exclusive_evidence_and_read_only_input(self):
        result = self.run_child('''
assert (inputs / 'frozen.txt').read_text() == 'original'
write_json_new(output / 'result.json', {'ok': True})
assert (output / 'result.json').is_file()
print(json.dumps(counts))
''')
        self.assertEqual(result['exclusive_files_created'], 1)
        self.assertEqual(result['blocked_write'], 0)

    def test_evidence_overwrite_append_and_remove_are_denied(self):
        result = self.run_child('''
write_json_new(output / 'result.json', {'ok': True})
errors = [rejected(lambda: (output / 'result.json').write_text('new')),
          rejected(lambda: (output / 'result.json').open('a')),
          rejected(lambda: (output / 'result.json').unlink())]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_write'], 3)

    def test_mutable_cache_allows_atomic_replacement_and_cleanup(self):
        result = self.run_child('''
first = cache / 'first'
first.write_text('one')
first.write_text('two')
second = cache / 'second'
second.write_text('three')
os.replace(first, second)
assert second.read_text() == 'two'
second.unlink()
print(json.dumps(counts))
''')
        self.assertEqual(result['mutable_file_opens'], 3)
        self.assertEqual(result['mutable_mutations'], 2)
        self.assertEqual(result['blocked_write'], 0)

    def test_tempfile_fdopen_and_dir_fd_cleanup_are_supported(self):
        result = self.run_child('''
import tempfile, shutil
tempfile.tempdir = str(tmp)
with tempfile.TemporaryDirectory() as temporary:
    child = Path(temporary) / 'file'
    child.write_text('scratch')
    with tempfile.TemporaryFile(dir=temporary, mode='w+') as stream:
        stream.write('scratch')
assert not Path(temporary).exists()
print(json.dumps(counts))
''')
        self.assertEqual(result['blocked_write'], 0)
        self.assertEqual(result['blocked_read'], 0)

    def test_no_write_to_input_dependency_or_outside(self):
        result = self.run_child('''
errors = [rejected(lambda: (inputs / 'frozen.txt').write_text('changed')),
          rejected(lambda: (outside / 'new').write_text('bad')),
          rejected(lambda: (Path(sys.prefix) / 'mps-guard-never-created').write_text('bad'))]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_write'], 3)

    def test_dotenv_reads_and_outside_reads_are_denied(self):
        result = self.run_child('''
errors = [rejected(lambda: (inputs / '.env').read_text()),
          rejected(lambda: (outside / 'not-present').read_text())]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_read'], 2)

    def test_darwin_missing_linux_maps_preserves_system_file_not_found(self):
        result = self.run_child('''
errors = []
for action in (lambda: Path('/proc/self/maps').read_text(),
               lambda: os.open('/proc/self/maps', os.O_RDONLY)):
    try:
        action()
    except FileNotFoundError:
        errors.append('FileNotFoundError')
    except BoundaryViolation:
        errors.append('BoundaryViolation')
    else:
        raise AssertionError('maps unexpectedly readable')
print(json.dumps({'errors': errors, 'platform': sys.platform, 'counts': counts}))
''')
        if result['platform'] == 'darwin':
            self.assertEqual(result['errors'], ['FileNotFoundError'] * 2)
            self.assertEqual(result['counts']['absent_platform_probes'], 2)
            self.assertEqual(result['counts']['blocked_read'], 0)
        else:
            self.assertEqual(result['errors'], ['BoundaryViolation'] * 2)

    def test_linux_maps_probe_never_allows_existing_files_or_symlinks(self):
        result = self.run_child('''
from unittest.mock import patch
original_lexists = os.path.lexists
original_symlink_check = Path.is_symlink
with patch('os.path.lexists', side_effect=lambda p: True if str(p) == '/proc/self/maps' else original_lexists(p)):
    existing = rejected(lambda: Path('/proc/self/maps').read_text())
with patch('os.path.lexists', side_effect=lambda p: False if str(p) == '/proc/self/maps' else original_lexists(p)):
    with patch.object(Path, 'is_symlink', lambda p: True if str(p) == '/proc/self' else original_symlink_check(p)):
        symlink = rejected(lambda: Path('/proc/self/maps').read_text())
print(json.dumps({'errors': [existing, symlink], 'counts': counts}))
''')
        self.assertEqual(result['errors'], ['MPS_PLATFORM_PROBE_IS_NOT_ABSENT'] * 2)
        self.assertEqual(result['counts']['absent_platform_probes'], 0)

    def test_missing_linux_maps_exception_does_not_allow_other_proc_paths(self):
        result = self.run_child('''
errors = [rejected(lambda: Path('/proc/self/other').read_text()),
          rejected(lambda: Path('/proc/other/maps').read_text()),
          rejected(lambda: Path('/proc/self/../self/maps').read_text())]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_read'], 3)
        self.assertEqual(result['counts']['absent_platform_probes'], 0)

    def test_maps_permission_error_is_not_treated_as_absence(self):
        result = self.run_child('''
from unittest.mock import patch
original_lstat = os.lstat
def denied_lstat(path, *args, **kwargs):
    if str(path) == '/proc/self/maps':
        raise PermissionError('synthetic denied path')
    return original_lstat(path, *args, **kwargs)
with patch('os.lstat', side_effect=denied_lstat):
    error = rejected(lambda: Path('/proc/self/maps').read_text())
print(json.dumps({'error': error, 'counts': counts}))
''')
        self.assertEqual(result['error'], 'MPS_PLATFORM_PROBE_IS_NOT_ABSENT')
        self.assertEqual(result['counts']['absent_platform_probes'], 0)

    def test_blocked_read_path_diagnostics_are_bounded_and_content_free(self):
        result = self.run_child('''
for i in range(12):
    rejected(lambda i=i: (outside / f'rejected-{i}').read_text())
print(json.dumps(counts))
''')
        self.assertEqual(result['blocked_read'], 12)
        self.assertEqual(len(result['blocked_read_paths']), 8)
        self.assertTrue(all('/outside/rejected-' in p for p in result['blocked_read_paths']))

    def test_cache_cannot_move_input_or_evidence(self):
        result = self.run_child('''
write_json_new(output / 'result.json', {'ok': True})
errors = [rejected(lambda: os.rename(inputs / 'frozen.txt', cache / 'stolen')),
          rejected(lambda: os.rename(output / 'result.json', cache / 'hidden')),
          rejected(lambda: os.rename(cache / 'missing', outside / 'escaped'))]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_write'], 3)

    def test_network_sql_and_subprocess_are_denied(self):
        result = self.run_child('''
errors = [rejected(lambda: socket.create_connection(('127.0.0.1', 1))),
          rejected(lambda: socket.socket().bind(('127.0.0.1', 0))),
          rejected(lambda: sqlite3.connect(':memory:')),
          rejected(lambda: subprocess.Popen(['/usr/bin/true']))]
assert socket.has_ipv6 is False
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_network'], 2)
        self.assertEqual(result['counts']['blocked_database'], 1)
        self.assertEqual(result['counts']['blocked_subprocess'], 1)

    def test_low_level_multiprocessing_spawn_is_denied_before_native_exec(self):
        result = self.run_child('''
try:
    import _posixsubprocess
except ImportError:
    print(json.dumps({'supported': False, 'counts': counts}))
else:
    from multiprocessing.util import spawnv_passfds
    errors = [rejected(lambda: spawnv_passfds('/usr/bin/true', ['/usr/bin/true'], [])),
              rejected(lambda: _posixsubprocess.fork_exec())]
    if hasattr(subprocess, '_fork_exec'):
        errors.append(rejected(lambda: subprocess._fork_exec()))
    print(json.dumps({'supported': True, 'errors': errors, 'counts': counts}))
''')
        if result['supported']:
            self.assertEqual(result['errors'], ['MPS_SUBPROCESS_FORBIDDEN'] * len(result['errors']))
            self.assertEqual(result['counts']['blocked_subprocess'], len(result['errors']))
            self.assertIn('_posixsubprocess.fork_exec', result['counts']['low_level_spawn_entrypoints_blocked'])

    def test_links_are_denied_inside_mutable_cache(self):
        result = self.run_child('''
errors = [rejected(lambda: os.symlink(inputs / 'frozen.txt', cache / 'alias')),
          rejected(lambda: os.link(inputs / 'frozen.txt', cache / 'hard'))]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_write'], 2)

    def test_directory_fd_escape_is_denied(self):
        result = self.run_child('''
fd = os.open(cache, os.O_RDONLY)
try:
    error = rejected(lambda: os.open('../outside/escape', os.O_CREAT|os.O_EXCL|os.O_WRONLY, dir_fd=fd))
finally:
    os.close(fd)
print(json.dumps({'error': error, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_write'], 1)

    def test_mutable_root_cannot_be_deleted_or_moved(self):
        result = self.run_child('''
errors = [rejected(lambda: cache.rmdir()),
          rejected(lambda: os.rename(cache, tmp / 'moved-cache'))]
print(json.dumps({'errors': errors, 'counts': counts}))
''')
        self.assertEqual(result['counts']['blocked_write'], 2)

    def test_guard_cannot_be_installed_twice(self):
        result = self.run_child('''
try:
    install_guards(output, [inputs], [cache, tmp])
except RuntimeError as exc:
    print(json.dumps({'error': str(exc)}))
else:
    raise AssertionError('second guard was accepted')
''')
        self.assertEqual(result['error'], 'MPS_GUARDS_ALREADY_INSTALLED')


if __name__ == '__main__':
    unittest.main()
