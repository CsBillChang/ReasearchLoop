"""验证中性配置入口、已有来源文件兼容和空路径默认值；不访问网络。"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_loop.config import Settings, paper_sources_path


class SourceConfigurationTests(unittest.TestCase):
    """用临时项目配置验证优先级，不读取真实密钥或改变原件。"""

    def setUp(self):
        """每例使用独立目录及清空环境，避免本机配置影响结果。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write_config(self, name: str) -> Path:
        """创建包含只读合成连接的来源配置，不使用生产凭据。"""
        path = self.root / 'data' / 'config' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'source_backends': {'arxiv': {
            'url': 'https://papers.example', 'anon_key': 'fixture', 'enabled': True}}}))
        return path

    def test_new_filename_is_default_and_preferred(self):
        """新文件默认生效；同时存在旧文件也不覆盖新配置。"""
        preferred = self.write_config('paper-sources.json')
        self.write_config('dpr-sources.json')
        self.assertEqual(paper_sources_path(self.root), preferred)
        self.assertEqual(Settings.from_env(self.root).discovery_backends['arxiv']['url'],
                         'https://papers.example')

    def test_legacy_file_is_read_without_rewriting(self):
        """原本机来源文件仍可读取，其字节和文件名保持。"""
        legacy = self.write_config('dpr-sources.json')
        content = legacy.read_bytes()
        self.assertEqual(paper_sources_path(self.root), legacy)
        self.assertEqual(Settings.from_env(self.root).discovery_backends['arxiv']['anon_key'], 'fixture')
        self.assertEqual(legacy.read_bytes(), content)
        self.assertFalse((legacy.parent / 'paper-sources.json').exists())

    def test_new_explicit_variable_wins_and_legacy_variable_still_works(self):
        """显式配置优先，支持原变量但不会覆盖用户的新选择。"""
        legacy = self.write_config('old.json')
        preferred = self.write_config('new.json')
        os.environ['RESEARCHLOOP_DPR_CONFIG'] = str(legacy)
        self.assertEqual(paper_sources_path(self.root), legacy)
        os.environ['RESEARCHLOOP_PAPER_SOURCES_CONFIG'] = str(preferred)
        self.assertEqual(paper_sources_path(self.root), preferred)

    def test_blank_optional_directories_use_project_defaults(self):
        """空目录变量不会把缓存或成果目录指向启动工作目录。"""
        os.environ.update(RESEARCHLOOP_DATA_DIR='', RESEARCHLOOP_OUTPUT_DIR='')
        settings = Settings.from_env(self.root)
        self.assertEqual(settings.data_dir, self.root / 'data' / 'ag1')
        self.assertEqual(settings.output_dir, settings.data_dir / 'notes')

    def test_minimal_template_has_same_required_keys_as_local_setup(self):
        """公开模板仅激活两个最小模型项；示例密钥为空，其余为可选说明。"""
        root = Path(__file__).resolve().parents[3]
        entries = {line.split('=', 1)[0]: line.split('=', 1)[1]
                   for line in (root / '.env.example').read_text().splitlines()
                   if line and not line.startswith('#') and '=' in line}
        self.assertEqual(set(entries), {'DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL'})
        self.assertEqual(entries['DEEPSEEK_API_KEY'], '')
        self.assertTrue(entries['DEEPSEEK_MODEL'])

    def test_process_environment_wins_over_local_dotenv(self):
        """加载本地文件不会替换进程传入的模型和目录配置。"""
        (self.root / '.env').write_text('DEEPSEEK_MODEL=file-model\nRESEARCHLOOP_DATA_DIR=\n')
        os.environ['DEEPSEEK_MODEL'] = 'process-model'
        self.assertEqual(Settings.from_env(self.root).deepseek_model, 'process-model')
