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
