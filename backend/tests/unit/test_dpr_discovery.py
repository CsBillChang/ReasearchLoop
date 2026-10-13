# DPR 适配测试：覆盖逐源 RPC、候选映射与日期条件；替身响应不代表线上来源可用。
import json
import unittest
import httpx
from research_loop.providers.discovery import DiscoveryProvider, SUPPORTED_SOURCES
from research_loop.providers.dpr_sources import resolve_backends, RPC_NAMES


class DPRDiscoveryTests(unittest.TestCase):
    def config(self):
        return resolve_backends({'supabase':{'url':'https://dpr.example','anon_key':'public-fixture','bm25_rpc':'match_arxiv_papers_bm25'}})

    def test_seventeen_rpc_contracts_and_dates(self):
        calls=[]
        def handler(request):
            calls.append(request); body=json.loads(request.content)
            self.assertEqual(set(body),{'query_text','match_count','filter_published_start','filter_published_end'})
            self.assertEqual(body['filter_published_end'],'2026-10-09T00:00:00+00:00')
            self.assertEqual(request.headers['accept-profile'],'public')
            source=next(s for s,rpc in RPC_NAMES.items() if request.url.path.endswith('/'+rpc))
            return httpx.Response(200,json=[{'id':source+':123','source':source.upper()+'-2025-Accepted','title':'Paper',
                'published':'2025-06-01T12:00:00Z','link':'https://paper.example/detail','pdf_url':'https://paper.example/a.pdf','score':0.7}])
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            r=DiscoveryProvider(backends=self.config(),client=client).search('object detection','2023-10-08','2026-10-08',list(SUPPORTED_SOURCES),20)
        self.assertEqual(len(calls),17);self.assertEqual(len(r.candidates),17)
        self.assertEqual(len({r.url.path for r in calls}),17)
        self.assertTrue(all(s.status=='queried_success' for s in r.source_results))
        self.assertEqual(r.source_results[0].request['rpc'], RPC_NAMES['arxiv'])
        self.assertEqual(r.source_results[0].request['payload']['filter_published_start'], '2023-10-08T00:00:00+00:00')
        self.assertNotIn('anon_key', r.source_results[0].request)
        self.assertEqual(r.candidates[0].metadata['source_raw'],'ARXIV-2025-Accepted')
        self.assertEqual(r.candidates[0].paper_url,'https://paper.example/detail')
        self.assertNotIn('cited_by_count',r.candidates[0].metadata)

    def test_missing_config_never_silently_calls_other_sources(self):
        with httpx.Client(transport=httpx.MockTransport(lambda _:self.fail('unexpected network'))) as client:
            r=DiscoveryProvider(client=client).search('query',sources=['arxiv','iclr','cvpr'])
        self.assertEqual([s.status for s in r.source_results if s.source in ['arxiv','iclr','cvpr']],['not_configured']*3)

    def test_empty_and_failure_are_distinct_no_fallback(self):
        def handler(req):
            if 'arxiv' in req.url.path:return httpx.Response(200,json=[])
            return httpx.Response(429,json={'message':'rate limited'})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            r=DiscoveryProvider(backends=self.config(),client=client).search('query',sources=['arxiv','cvpr'])
        statuses={s.source:s.status for s in r.source_results}
        self.assertEqual(statuses['arxiv'],'queried_empty');self.assertEqual(statuses['cvpr'],'failed')
        failed = next(s for s in r.source_results if s.source == 'cvpr')
        self.assertEqual(failed.request['http_status'], 429)

    def test_date_filter_and_source_diversity_preserve_identity(self):
        def handler(req):
            source='arxiv' if 'arxiv' in req.url.path else 'cvpr'
            return httpx.Response(200,json=[{'id':source+str(i),'title':'P','published':d} for i,d in enumerate(['2020-01-01','2025-01-01','2026-01-01'])])
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            r=DiscoveryProvider(backends=self.config(),client=client).search('query','2023-10-08','2026-10-08',['arxiv','cvpr'],2)
        self.assertEqual([p.source for p in r.candidates],['arxiv','cvpr']);self.assertTrue(r.truncated)

    def test_per_source_overrides_do_not_replace_other_rpcs(self):
        b=resolve_backends({'supabase':{'url':'https://shared','anon_key':'k'},'source_backends':{'iclr':{'url':'https://other','bm25_rpc':'custom'}}})
        self.assertEqual(b['iclr']['url'],'https://other');self.assertEqual(b['cvpr']['bm25_rpc'],'match_cvpr_papers_bm25')

    def test_arxiv_pdf_url_is_resolved_only_from_valid_source_identity(self):
        paper=DiscoveryProvider._row_to_candidate({'id':'2601.12345v2','title':'P'},'arxiv')
        self.assertEqual(paper.pdf_url,'https://arxiv.org/pdf/2601.12345v2')
        self.assertIsNone(DiscoveryProvider._row_to_candidate({'id':'cvpr-arbitrary','title':'P'},'cvpr').pdf_url)

    def test_settings_load_local_snapshot_and_honor_explicit_override(self):
        import os,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from research_loop.config import Settings
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=root/'data/config/dpr-sources.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'source_backends':{'arxiv':{'url':'https://old','anon_key':'fixture'},'cvpr':{'url':'https://old','anon_key':'fixture'}}}))
            with patch.dict(os.environ,{'SUPABASE_URL':'https://override'},clear=True):
                settings=Settings.from_env(root)
            self.assertEqual(settings.discovery_backends['cvpr']['url'],'https://override')
            self.assertEqual(settings.discovery_backends['cvpr']['bm25_rpc'],'match_cvpr_papers_bm25')
            self.assertEqual(settings.discovery_backends['arxiv']['bm25_rpc'],'match_arxiv_papers_bm25')


class DefaultProjectRootTests(__import__('unittest').TestCase):
    def test_implicit_root_is_repository_not_workspace_parent(self):
        from pathlib import Path
        from research_loop.config import Settings
        expected=Path(__file__).resolve().parents[3]
        self.assertEqual(Settings.from_env().project_root,expected)
