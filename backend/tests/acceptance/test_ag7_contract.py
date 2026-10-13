"""最终交付冻结合同；默认不运行真实模型或旧评测。"""
import unittest
from scripts.run_ag7 import SPEC, ROOT, configuration


class AG7ContractTests(unittest.TestCase):
    """固定同材料两问、12次上限、禁止保存与隔离配置。"""

    def test_scope_and_budget(self):
        """真实执行只能使用已讨论的两问，不隐式扩大到旧论文批次。"""
        self.assertEqual(len(SPEC['cases']),2)
        self.assertEqual(SPEC['physical_cap'],12)
        self.assertEqual(SPEC['save_policy'],'deny')
        self.assertFalse(SPEC['retry_batch']);self.assertFalse(SPEC['new_protocol'])
        self.assertTrue((ROOT/SPEC['pdf']).is_file())

    def test_isolated_configuration(self):
        """真实最终检查的队列、文件和向量集合都不指向生产。"""
        settings=configuration()
        self.assertIn('/researchloop_test_ag75_',settings.database_url)
        self.assertIn('ag7-5-local-delivery',str(settings.data_dir))
        self.assertTrue(settings.retrieval_namespace.startswith('researchloop_test_'))
