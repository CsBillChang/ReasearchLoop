"""最终交付冻结合同；默认不运行真实模型或旧评测。"""
import unittest
from unittest.mock import patch

from research_loop.config import Settings
from scripts.run_ag7 import SPEC, ROOT, configuration


class AG7ContractTests(unittest.TestCase):
    """固定同材料两问、12次上限、禁止保存与隔离配置。"""

    def test_scope_and_budget(self):
        """真实执行只能使用已讨论的两问，不隐式扩大到旧论文批次。"""
        self.assertEqual(len(SPEC['cases']),2)
        self.assertEqual(SPEC['physical_cap'],12)
        self.assertEqual(SPEC['save_policy'],'deny')
        self.assertFalse(SPEC['retry_batch']);self.assertFalse(SPEC['new_protocol'])

    def test_historical_local_pdf_when_available(self):
        """本机PDF单独核对，缺失不影响公开范围/预算合同的零模型测试。"""
        pdf = ROOT / SPEC['pdf']
        if not pdf.is_file():
            self.skipTest('historical local AG7 PDF is not distributed')
        self.assertGreater(pdf.stat().st_size, 0)

    def test_isolated_configuration(self):
        """真实最终检查的队列、文件和向量集合都不指向生产。"""
        # 只测试配置变换；不读取发布者的.env、不连接数据库或要求个人凭据。
        base = Settings(project_root=ROOT, data_dir=ROOT/'data', output_dir=ROOT/'data/notes',
                        database_url='mysql://example:example@127.0.0.1:3306/researchloop')
        with patch('scripts.run_ag7.Settings.from_env', return_value=base):
            settings=configuration()
        self.assertIn('/researchloop_test_ag75_',settings.database_url)
        self.assertIn('ag7-5-local-delivery',str(settings.data_dir))
        self.assertTrue(settings.retrieval_namespace.startswith('researchloop_test_'))

    def test_isolated_configuration_rejects_non_local_database(self):
        """SQLite或远程地址不能误用为历史真实验收库；全程不连接服务。"""
        for url in ('sqlite:///:memory:', 'mysql://example:example@database.test/researchloop'):
            with self.subTest(url=url):
                base = Settings(project_root=ROOT, data_dir=ROOT/'data', output_dir=ROOT/'data/notes',
                                database_url=url)
                with patch('scripts.run_ag7.Settings.from_env', return_value=base):
                    with self.assertRaisesRegex(ValueError, 'LOCAL_MYSQL_REQUIRED'):
                        configuration()
