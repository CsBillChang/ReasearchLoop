"""冷备份恢复合同：完整性、路径重绑定、原始收据与禁止覆盖；零模型。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_loop.config import Settings
from research_loop.domain.tasks import TaskCreate
from research_loop.services.tasks import TaskService
from research_loop.storage.mysql import Database
from scripts.backup_restore import backup, restore, verify_backup, remap


class BackupRestoreTests(unittest.TestCase):
    """构造有 saved 文件、外部原件和 unknown 收据的最小数据库。"""

    def setUp(self):
        """每例独立目录；合成 PDF 仅测试字节保护，不冒充解析验收。"""
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        root_patch=patch('scripts.backup_restore.ROOT',self.root)
        root_patch.start();self.addCleanup(root_patch.stop)
        self.settings=Settings(project_root=self.root,data_dir=self.root/'data',output_dir=self.root/'notes',database_url='sqlite:///'+str(self.root/'source.db'))
        self.db=Database(self.settings.database_url);self.db.ensure_schema()
        self.tid=TaskService(self.db).create(TaskCreate(prompt='只测试恢复',idempotency_key='restore-test',save_policy='deny')).task_id
        pdf=b'%PDF- synthetic backup fixture';self.hash=hashlib.sha256(pdf).hexdigest()
        external=self.root/'external/cache';(external/'pdf').mkdir(parents=True);(external/'parsed').mkdir()
        self.pdf=external/'pdf'/(self.hash+'.pdf');self.pdf.write_bytes(pdf)
        self.cache=external/'parsed/doc.json';self.cache.write_text(json.dumps({'pages':[]}))
        self.db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?)',('doc','paper',self.hash,'dpr','1','a'*64,1,str(self.cache),'[]','now'))
        self.settings.output_dir.mkdir();self.note=self.settings.output_dir/'note.md';self.note.write_text('用户内容保留')
        self.db.execute('INSERT INTO save_receipts VALUES(?,?,?,?,?,?)',('save','draft','b'*64,str(self.note),'saved','now'))
        self.db.execute('INSERT INTO model_calls(call_id,task_id,sequence_no,status,request_json,usage_json,sent_state,created_at) VALUES(?,?,?,?,?,?,?,?)',('unknown',self.tid,0,'unknown',json.dumps({'path':str(self.pdf)}),'{}','unknown','now'))
        self.back=self.root/'backup'

    def test_roundtrip_and_unknown(self):
        """文件与表关联恢复，未知请求原样保留，目标默认冻结执行。"""
        manifest=backup(self.settings,self.back)
        report=restore(self.back,'sqlite:///'+str(self.root/'new.db'),self.root/'restored')
        target=Database('sqlite:///'+str(self.root/'new.db'))
        self.assertTrue(report['model_receipts_unchanged'])
        self.assertFalse(report['execution_enabled'])
        self.assertEqual(target.fetchone('SELECT status FROM model_calls')['status'],'unknown')
        note=Path(target.fetchone('SELECT path FROM save_receipts')['path'])
        self.assertNotEqual(note,self.note);self.assertEqual(note.read_bytes(),self.note.read_bytes())
        cache=Path(target.fetchone('SELECT cache_path FROM documents')['cache_path'])
        self.assertTrue((cache.parent.parent/'pdf'/(self.hash+'.pdf')).exists())
        self.assertTrue((Path(report['data_dir'])/'restore-hold.json').exists())
        self.assertFalse(manifest['config']['secrets_included'])

    def test_neutral_source_configuration_is_redacted_in_backup(self):
        """新文件名纳入脱敏配置；原本机来源文件不被修改。"""
        source = self.root / 'data/config/paper-sources.json'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'supabase': {'url': 'https://papers.example',
                                                  'anon_key': 'fixture-private'}}))
        original = source.read_bytes()
        manifest = backup(self.settings, self.back)
        sanitized = manifest['config']['paper_sources_without_credentials']
        self.assertEqual(sanitized['supabase']['anon_key'], '[configure locally]')
        self.assertNotIn('fixture-private', json.dumps(manifest['config']))
        self.assertEqual(source.read_bytes(), original)

    def test_backup_refuses_running(self):
        """执行中任务不通过强杀或改状态制造一致性备份。"""
        self.db.execute("UPDATE tasks SET status='running'")
        with self.assertRaisesRegex(RuntimeError,'RUNNING_TASKS'):
            backup(self.settings,self.back)
        self.assertFalse(self.back.exists())

    def test_backup_missing_original(self):
        """缺 PDF 会明确失败，而非创建看似完整的备份。"""
        self.pdf.unlink()
        with self.assertRaisesRegex(ValueError,'MISSING_OR_CHANGED'):
            backup(self.settings,self.back)

    def test_corruption_rejected_before_restore(self):
        """备份被修改时不创建目标库或目标目录。"""
        backup(self.settings,self.back)
        (self.back/'database.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):
            restore(self.back,'sqlite:///'+str(self.root/'new.db'),self.root/'restored')
        self.assertFalse((self.root/'new.db').exists())

    def test_existing_database_and_directory_refused(self):
        """恢复接口不能覆盖生产库或已有文件目录。"""
        backup(self.settings,self.back)
        with self.assertRaisesRegex(ValueError,'NEW_RESTORE_DATABASE'):
            restore(self.back,self.settings.database_url,self.root/'restored')
        with self.assertRaisesRegex(ValueError,'ISOLATED_RESTORE_DATABASE'):
            restore(self.back,'mysql://u:p@127.0.0.1/researchloop',self.root/'restored')
        with self.assertRaisesRegex(ValueError,'NEW_RESTORE_DIRECTORY'):
            restore(self.back,'sqlite:///'+str(self.root/'new.db'),self.root)

    def test_path_traversal_rejected(self):
        """即使清单被编辑也不能把恢复文件写出目标目录。"""
        backup(self.settings,self.back)
        path=self.back/'manifest.json';manifest=json.loads(path.read_text());manifest['files'][0]['archive']='../escape'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'UNSAFE_ARCHIVE'):
            verify_backup(self.back)
        # 篡改原始路径也不能借 data 根目录匹配把目标指向父目录。
        manifest['files'][0]['archive']=next(p.relative_to(self.back).as_posix() for p in (self.back/'files').rglob('*') if p.is_file())
        manifest['files'][0]['original']=str(self.settings.data_dir)+'/../../escape'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'UNSAFE_ORIGINAL'):
            restore(self.back,'sqlite:///'+str(self.root/'escape.db'),self.root/'restored')
        self.assertFalse((self.root/'escape.db').exists())

    def test_file_corruption_rejected(self):
        """原件备份 hash 错误不会被容忍。"""
        manifest=backup(self.settings,self.back)
        (self.back/manifest['files'][0]['archive']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'FILE_HASH_MISMATCH'):
            verify_backup(self.back)

    def test_remap_does_not_rewrite_content(self):
        """只替换完整路径值，不替换包含旧路径的正文或原始提示词。"""
        self.assertEqual(remap('论文说明 /old/path',{'/old/path':'/new/path'}),'论文说明 /old/path')
        self.assertEqual(remap({'path':'/old/path'},{'/old/path':'/new/path'}),{'path':'/new/path'})
