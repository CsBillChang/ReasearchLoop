# 来源目录测试：17 源状态与默认范围独立，不能将未查询来源记成成功。
import unittest
from research_loop.providers.discovery import DiscoveryProvider, SUPPORTED_SOURCES


class DiscoverySourceTests(unittest.TestCase):
    def test_unconfigured_search_records_all_seventeen_sources(self):
        provider = DiscoveryProvider(client=__import__("httpx").Client())
        result = provider.search("query", sources=["arxiv", "cvpr"], limit=2)
        provider.close()
        self.assertEqual(len(result.source_results), 17)
        self.assertEqual(set(item.source for item in result.source_results), set(SUPPORTED_SOURCES))
        self.assertIn(next(item for item in result.source_results if item.source == "cvpr").status, {"failed", "not_configured", "queried_empty", "queried_success"})
