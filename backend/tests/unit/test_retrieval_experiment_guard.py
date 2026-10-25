"""通过子进程验证不可逆实验守卫，避免污染unittest主进程。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


class RetrievalExperimentGuardTests(unittest.TestCase):
    """只使用合成文件和被阻断的接口，不发送网络或执行模型。"""

    def child(self, code):
        """每例单独安装审计钩子；子进程在临时目录结束。"""
        with tempfile.TemporaryDirectory() as directory:
            program = '''
from pathlib import Path
import json, os, socket, sqlite3, subprocess, sys
from scripts.retrieval_experiment_guard import install_guards, write_json_new, write_bytes_new, BoundaryViolation
base = Path(sys.argv[1]).resolve()
output = base / 'output'; output.mkdir()
source = base / 'source.txt'; source.write_text('frozen')
outside = base / 'outside.txt'; outside.write_text('original')
foreign_fd = os.open(outside, os.O_WRONLY)
counts = install_guards(output, [source, outside])

def rejected(function):
    try:
        function()
    except BoundaryViolation:
        return True
    raise AssertionError('operation was not rejected')
'''
            result = subprocess.run([sys.executable, '-c', textwrap.dedent(program) + '\n' + textwrap.dedent(code), directory],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_exclusive_files_and_directories_are_supported(self):
        """新目录、新JSON及os.open/write成功；只读材料保持。"""
        result = self.child('''
sub = output / 'raw'; sub.mkdir()
write_json_new(sub / 'one.json', {'answer': 3})
assert json.loads((sub / 'one.json').read_text()) == {'answer': 3}
fd = os.open(sub / 'two.bin', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(fd, b'two'); os.close(fd)
assert (sub / 'two.bin').read_bytes() == b'two'
assert source.read_text() == 'frozen'
try:
    write_json_new(sub / 'one.json', {'bad': 1})
except FileExistsError:
    pass
else:
    raise AssertionError('old evidence overwritten')
print(json.dumps(counts))
''')
        self.assertEqual(result['exclusive_files_created'], 2)
        self.assertEqual(result['output_directories_created'], 1)

    def test_outside_append_and_mutations_are_rejected(self):
        """根目录外写、追加、截断、删改与链接均拒绝。"""
        result = self.child('''
write_bytes_new(output / 'kept', b'kept')
assert rejected(lambda: write_bytes_new(base / 'new', b'bad'))
assert rejected(lambda: outside.open('w'))
assert rejected(lambda: (output / 'kept').open('ab'))
assert rejected(lambda: (output / 'kept').open('wb'))
assert rejected(lambda: os.write(foreign_fd, b'bad'))
assert rejected(lambda: os.unlink(output / 'kept'))
assert rejected(lambda: os.rename(output / 'kept', output / 'other'))
assert rejected(lambda: os.mkdir(base / 'elsewhere'))
assert rejected(lambda: os.symlink(source, output / 'link'))
assert rejected(lambda: os.link(source, output / 'hardlink'))
assert (output / 'kept').read_bytes() == b'kept'
assert outside.read_text() == 'original'
print(json.dumps(counts))
''')
        self.assertEqual(result['blocked_write'], 10)

    def test_network_database_and_subprocess_are_rejected(self):
        """真实发送前拒绝多种入口，内存SQLite也不例外。"""
        result = self.child('''
s = socket.socket()
assert rejected(lambda: s.connect(('127.0.0.1', 6339)))
assert rejected(lambda: s.connect_ex(('127.0.0.1', 6351)))
assert rejected(lambda: socket.getaddrinfo('example.invalid', 443))
assert rejected(lambda: s.sendto(b'x', ('127.0.0.1', 33316)))
assert rejected(lambda: sqlite3.connect(':memory:'))
assert rejected(lambda: subprocess.run([sys.executable, '-c', 'print(1)']))
assert rejected(lambda: os.system('true'))
s.close()
print(json.dumps(counts))
''')
        self.assertEqual(result['blocked_network'], 4)
        self.assertEqual(result['blocked_database'], 1)
        self.assertEqual(result['blocked_subprocess'], 2)

    def test_read_allowlist_and_private_config_are_enforced(self):
        """显式允许的原件可读；其他文件和.env均不读。"""
        result = self.child('''
assert source.read_text() == 'frozen'
assert rejected(lambda: (base / 'missing.txt').read_bytes())
assert rejected(lambda: (output / '.env').read_bytes())
assert rejected(lambda: os.open('source.txt', os.O_RDONLY, dir_fd=foreign_fd))
print(json.dumps(counts))
''')
        self.assertEqual(result['blocked_read'], 3)

    def test_optional_http_and_mysql_final_entrances_are_rejected(self):
        """存在可选包时检查最终send/连接入口；缺包则保留明确覆盖。"""
        result = self.child('''
covered = []
try:
    import httpx
except ImportError:
    pass
else:
    client = object.__new__(httpx.Client)
    assert rejected(lambda: client.send(None)); covered.append('httpx')
try:
    import pymysql
except ImportError:
    pass
else:
    assert rejected(lambda: pymysql.connect()); covered.append('pymysql')
print(json.dumps({'counts': counts, 'covered': covered}))
''')
        if 'httpx' in result['covered']:
            self.assertEqual(result['counts']['blocked_network'], 1)
        if 'pymysql' in result['covered']:
            self.assertEqual(result['counts']['blocked_database'], 1)

    def test_nonfinite_json_is_rejected_before_creating_evidence(self):
        """NaN和无穷值不能成为合法JSON证据，也不留下半文件。"""
        result = self.child('''
for number in (float('nan'), float('inf'), float('-inf')):
    try:
        write_json_new(output / 'invalid.json', {'number': number})
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite JSON accepted')
assert not (output / 'invalid.json').exists()
write_json_new(output / 'terminal.json', {'status': 'stopped', 'reason': 'synthetic'})
assert json.loads((output / 'terminal.json').read_text())['status'] == 'stopped'
print(json.dumps(counts))
''')
        self.assertEqual(result['exclusive_files_created'], 1)

    def test_pythonpath_does_not_authorize_an_extra_deployment_directory(self):
        """修改sys.path不能绕过调用方显式读取范围。"""
        with tempfile.TemporaryDirectory() as directory:
            program = """
from pathlib import Path
import json, sys
from scripts.retrieval_experiment_guard import install_guards, BoundaryViolation
base = Path(sys.argv[1]).resolve(); output = base / 'output'; output.mkdir()
private = base / 'deployment'; private.mkdir(); (private/'database.sqlite').write_bytes(b'private')
sys.path.append(str(private))
counts = install_guards(output, [])
try:
    (private/'database.sqlite').read_bytes()
except BoundaryViolation:
    pass
else:
    raise AssertionError('sys.path expanded read access')
print(json.dumps(counts))
"""
            result = subprocess.run([sys.executable, '-c', textwrap.dedent(program), directory],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['blocked_read'], 1)

    def test_existing_numeric_runtime_imports_without_shared_cache_writes(self):
        """只导入已有数值包，不装载模型；守卫不妨碍CPU运行时准备。"""
        result = self.child('''
covered = []
for name in ('numpy', 'onnxruntime', 'tokenizers'):
    try:
        __import__(name)
    except ImportError:
        continue
    covered.append(name)
print(json.dumps({'counts': counts, 'covered': covered}))
''')
        self.assertEqual(result['counts']['blocked_write'], 0)
        self.assertEqual(result['counts']['blocked_network'], 0)

    def test_nonempty_or_alias_output_refuses_installation(self):
        """安装前即拒绝不合格目录，不留下半安装钩子。"""
        with tempfile.TemporaryDirectory() as directory:
            program = '''
from pathlib import Path
import sys
from scripts.retrieval_experiment_guard import install_guards
base = Path(sys.argv[1]).resolve(); output = base / 'output'; output.mkdir()
(output/'old').write_text('old')
try:
    install_guards(output, [])
except ValueError:
    pass
else:
    raise AssertionError('nonempty output accepted')
link = base / 'alias'; link.symlink_to(output, target_is_directory=True)
try:
    install_guards(link, [])
except ValueError:
    pass
else:
    raise AssertionError('alias output accepted')
(output/'still-writable').write_text('no partial hook')
'''
            result = subprocess.run([sys.executable, '-c', textwrap.dedent(program), directory],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
