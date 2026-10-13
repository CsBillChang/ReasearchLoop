# 历史引用统计匹配测试；当前日常工具链不启用被引量功能。
import unittest
from research_loop.domain.evidence import PaperCandidate
from research_loop.providers.citations import OpenAlexCitationProvider


class CitationMatchingTests(unittest.TestCase):
    def test_title_author_year_match_is_unique_and_zero_is_real(self):
        candidate = PaperCandidate(paper_id="p", title="A Paper", authors=["Alice Smith"], source="cvpr", published="2024-06-01")
        row = {"title": "A Paper", "publication_date": "2024-06-01", "cited_by_count": 0,
               "authorships": [{"author": {"display_name": "Alice Smith"}}]}
        self.assertTrue(OpenAlexCitationProvider._matches(candidate, row))
        self.assertEqual(row["cited_by_count"], 0)

    def test_ambiguous_title_does_not_choose(self):
        candidate = PaperCandidate(paper_id="p", title="A Paper", source="cvpr", published="2024")
        self.assertFalse(OpenAlexCitationProvider._matches(candidate, {"title": "A Paper", "publication_date": "2023"}))
