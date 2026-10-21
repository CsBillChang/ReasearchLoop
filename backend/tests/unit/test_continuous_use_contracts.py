"""连续使用缺陷的通用变体：来源子集、同版本地址、可选核验与当前进展。"""
import copy
import json
import unittest
from unittest.mock import Mock

from research_loop.agents.researcher import finish_spec
from research_loop.domain.context import project_research_handoffs
from research_loop.domain.evidence import PaperCandidate, SearchResult
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.providers.paper_acquisition import pdf_from_detail, resolve_pdf_url
from research_loop.services.research_progress import evidence_progress, readback_handoff
from research_loop.tools.base import ToolContext
from research_loop.tools.papers import specs
from research_loop.tools.registry import ToolRegistry
from unit.test_parent_handoff_projection import progress


class ContinuousUseContracts(unittest.TestCase):
    """不调用网络/模型，不使用真实论文身份或原文。"""

    def test_authorized_source_subset_reaches_service(self):
        """缩小合法来源集合必须传到执行层，不能被旧scope覆盖。"""
        papers = Mock()
        papers.search.return_value = SearchResult()
        context = ToolContext('task', SavePolicy.deny, services={'papers':papers,
                              'scope':{'sources':['CVPR','ICCV','ECCV']}}, role='researcher')
        registry = ToolRegistry(specs())
        registry.execute('search_papers', {'query':'detection', 'sources':['CVPR']}, context)
        self.assertEqual(papers.search.call_args.args[3], ['CVPR'])
        rejected = registry.execute('search_papers', {'query':'detection', 'sources':['arXiv']}, context)
        self.assertFalse(rejected.ok)
        self.assertEqual(papers.search.call_count, 1)

    def test_unsupported_source_does_not_block_supported_source(self):
        """未知来源明确标注未配置；合法源仍返回候选，不制造已查询状态。"""
        provider = DiscoveryProvider(backends={'cvpr':{'url':'https://fixture.invalid', 'anon_key':'synthetic',
                                                     'bm25_rpc':'fixture', 'enabled':True}})
        self.addCleanup(provider.close)
        candidate = PaperCandidate(paper_id='fixture', title='Synthetic detection', source='cvpr')
        provider._search_windows = Mock(return_value=([candidate], 1, {}))
        result = provider.search('detection', sources=['ICCV','CVPR'])
        self.assertEqual([p.paper_id for p in result.candidates], ['fixture'])
        unknown = next(r for r in result.source_results if r.source == 'iccv')
        self.assertEqual(unknown.status, 'not_configured')
        self.assertIn('unsupported', unknown.message)
        provider._search_windows.assert_called_once()

    def test_arxiv_versions_hosts_and_detail_metadata(self):
        """新版旧版编号均保持vN；不把相似域名当作官方来源。"""
        for identifier in ['2401.12345v3', '2501.1234v1', 'hep-th/9901001v2']:
            original = 'https://arxiv.org/abs/' + identifier
            pdf = 'https://arxiv.org/pdf/' + identifier
            self.assertEqual(resolve_pdf_url(None, original), pdf)
            self.assertEqual(resolve_pdf_url(None, pdf), pdf)
            self.assertEqual(pdf_from_detail(f'<meta name="citation_pdf_url" content="{pdf}">', original), pdf)
        fake = 'https://arxiv.org.evil.invalid/abs/2401.12345v3'
        self.assertEqual(resolve_pdf_url(None, fake), fake)
        self.assertEqual(resolve_pdf_url(None, 'https://arxiv.org/abs/not-a-paper'), 'https://arxiv.org/abs/not-a-paper')

    def test_current_progress_single_copy_and_old_versions_preserved(self):
        """当前成果精确相同时引用状态；旧版或改文不能引用当前版。"""
        own = progress(task='root')
        raw = copy.deepcopy(own)
        raw.pop('action_id')
        for finding in raw['findings']:
            finding.pop('source_integrity')
        messages = [{'role':'assistant','tool_calls':[{'id':'progress', 'function':{
            'name':'record_research_progress', 'arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'progress','content':json.dumps({'ok':True,'data':{'research_progress':raw}})}]
        saved = copy.deepcopy(messages)
        projected, context, _ = project_research_handoffs(messages, {'task_id':'root','research_progress':own})
        value = json.loads(projected[-1]['content'])['data']['research_progress']
        self.assertEqual(value['context_path'], 'current_task.research_progress')
        self.assertEqual(messages, saved)
        self.assertIn('source_catalog', context['research_progress'])
        for field, replacement in [('revision', 1), ('objective', '额外要求')]:
            changed = copy.deepcopy(saved)
            data = json.loads(changed[-1]['content']); data['data']['research_progress'][field] = replacement
            changed[-1]['content'] = json.dumps(data)
            out, _, _ = project_research_handoffs(changed, {'research_progress':own})
            self.assertEqual(out, changed)

    def test_missing_window_then_visible_repeat_have_different_routes(self):
        """真实冻结请求缺页时恢复；已展示的相同窗口才进入交接。"""
        source = {'document_id':'different-doc', 'content_sha256':'version-a', 'pages':[
            {'document_id':'different-doc', 'evidence_id':'different-page','page_number':19,'text':'actual source'}]}
        args = {'target_ref':'attachment', 'start_page':19, 'page_count':1}
        original = {'action_id':'one','tool_name':'read_paper','status':'succeeded', 'arguments':args,
                    'observation':{'ok':True,'data':{**source,'research_increment':{'kind':'new_evidence'}}}}
        observation = {'ok':True,'data':copy.deepcopy(source)}
        observation['data']['research_increment'] = evidence_progress([original], 'read_paper', args, observation, [])
        readback = {'action_id':'two','tool_name':'read_paper','status':'succeeded','arguments':args,'observation':observation}
        self.assertEqual(observation['data']['research_increment']['kind'], 'context_readback')
        self.assertIsNone(readback_handoff([original,readback]))
        messages = [{'role':'assistant','tool_calls':[{'id':'shown','function':{'name':'read_paper','arguments':'{}'}}]},
                    {'role':'tool','tool_call_id':'shown','content':json.dumps({'ok':True,'data':source})}]
        repeated = copy.deepcopy(readback); repeated['action_id'] = 'three'
        repeated['observation']['data']['research_increment'] = evidence_progress([original,readback], 'read_paper', args, observation, messages)
        self.assertEqual(repeated['observation']['data']['research_increment']['kind'], 'same_evidence')
        self.assertIsNotNone(readback_handoff([original,readback,repeated]))

    def test_invalid_json_never_becomes_empty_valid_handoff(self):
        """全默认字段的结束参数也必须拒绝截断JSON，不猜测内容。"""
        registry = ToolRegistry([finish_spec()])
        context = ToolContext('task', SavePolicy.deny, services={}, role='researcher')
        result = registry.execute('finish_research', {'_invalid_json':'{"findings": ['}, context)
        self.assertFalse(result.ok)
        self.assertEqual(result.data['error_code'], 'INVALID_TOOL_JSON')
        self.assertFalse(result.data['tool_executed'])
