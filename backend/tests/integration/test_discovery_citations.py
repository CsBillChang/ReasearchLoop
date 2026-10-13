# 历史发现/引用接口集成测试；引用统计退出活动工具后仍保留兼容测试证据。
import unittest
import httpx
from research_loop.domain.evidence import PaperCandidate
from research_loop.providers.citations import OpenAlexCitationProvider


class CitationIntegrationTests(unittest.TestCase):
    def test_openalex_provider_parses_real_zero_from_fixed_response(self):
        def handler(request):
            return httpx.Response(200, json={"results": [{"title": "A Paper", "publication_date": "2024-01-01",
                "cited_by_count": 0, "id": "https://openalex.org/W1", "authorships": []}]})
        provider = OpenAlexCitationProvider(client=httpx.Client(transport=httpx.MockTransport(handler)))
        candidate = PaperCandidate(paper_id="p", title="A Paper", source="cvpr", published="2024-01-01")
        result = provider.lookup(candidate)
        provider.close()
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.count, 0)

    def test_missing_count_is_not_fabricated_as_zero(self):
        def handler(request):
            return httpx.Response(200,json={'results':[{'title':'A Paper','publication_date':'2024-01-01','id':'https://openalex.org/W1','authorships':[]}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result=OpenAlexCitationProvider(client=client).lookup(PaperCandidate(paper_id='p',title='A Paper',source='cvpr',published='2024-01-01'))
        self.assertIsNone(result.count);self.assertEqual(result.status,'unavailable')

    def test_doi_fallback_records_actual_matching_method(self):
        def handler(request):
            if '/https://doi.org/' in str(request.url):return httpx.Response(404,json={})
            return httpx.Response(200,json={'results':[{'title':'A Paper','publication_date':'2024-01-01','cited_by_count':3,'id':'https://openalex.org/W1','authorships':[]}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result=OpenAlexCitationProvider(client=client).lookup(PaperCandidate(paper_id='p',title='A Paper',source='cvpr',published='2024-01-01',doi='10.0/test'))
        self.assertEqual(result.matched_by,'title_authors_year');self.assertEqual(result.count,3)

    def test_429_preserves_bounded_rate_limit_diagnostic(self):
        def handler(request):
            return httpx.Response(429, headers={'retry-after': '12', 'x-ratelimit-remaining': '0'},
                                  json={'error': 'rate_limited', 'message': 'budget exhausted'})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            result = OpenAlexCitationProvider(client=client).lookup(
                PaperCandidate(paper_id='p', title='A Paper', source='cvpr', published='2024-01-01'))
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.http_status, 429)
        self.assertEqual(result.retry_after_seconds, 12)
        self.assertEqual(result.rate_limit_remaining, 0)
        self.assertIn('budget exhausted', result.message)
        self.assertEqual(result.access_mode, 'anonymous')
