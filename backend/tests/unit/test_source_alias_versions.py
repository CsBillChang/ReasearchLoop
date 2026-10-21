"""来源别名版本比较的纯内存正反例，不连接网络或生产数据库。"""
import copy
import json
import socket
import unittest
from unittest.mock import Mock, patch

from research_loop.services.research_progress import ResearchProgressService


class Catalogue:
    """仅允许题录只读查询，任何写入均无可用入口。"""

    def __init__(self, entries):
        """保留合成题录及调用次数，用于检查同身份不额外查别名。"""
        self.entries, self.reads = entries, []

    def fetchone(self, statement, params):
        """按精确论文ID返回已存题录。"""
        assert statement == 'SELECT metadata_json FROM papers WHERE paper_id=?'
        self.reads.append(params[0])
        row = self.entries.get(params[0])
        return {'metadata_json':json.dumps(row)} if row is not None else None


class SourceAliasVersionTests(unittest.TestCase):
    """完整字段和同版locator双重保护；不泛化忽略paper_id。"""

    def setUp(self):
        """构造一份明确v1的旧URL身份及当前arXiv题录。"""
        self.old = {'evidence_id':'evidence:A','document_id':'doc:A','paper_id':'user-url:A',
                    'source_sha256':'a'*64,'page_number':1,'text_sha256':'b'*64}
        self.current = {**self.old,'paper_id':'2603.00595v1'}
        self.entries = {
            'user-url:A':{'paper_id':'user-url:A','paper_url':'https://arxiv.org/pdf/2603.00595v1','version':'v1'},
            '2603.00595v1':{'paper_id':'2603.00595v1','paper_url':'https://arxiv.org/abs/2603.00595v1',
                           'pdf_url':'https://arxiv.org/pdf/2603.00595v1','version':None}}
        self.db = Catalogue(self.entries)
        self.service = ResearchProgressService(self.db)
        self.service.source = Mock(side_effect=lambda *_:copy.deepcopy(self.current))
        for owner,name in ((socket.socket,'connect'),(socket,'create_connection')):
            blocker = patch.object(owner,name,side_effect=AssertionError('NETWORK_FORBIDDEN'))
            blocker.start(); self.addCleanup(blocker.stop)

    def test_same_paper_keeps_strict_equality_without_catalogue_lookup(self):
        """同paper_id沿用完整版本相等，不依赖别名元数据可用性。"""
        self.current = copy.deepcopy(self.old)
        self.assertTrue(self.service.source_version_matches('task-A',self.old))
        self.assertEqual(self.db.reads,[])

    def test_explicit_v1_alias_preserves_old_reference_and_current_source(self):
        """仅paper_id不同且明确同版时兼容，旧记录及当前源均不被修改。"""
        before = copy.deepcopy((self.old,self.current,self.entries))
        self.assertTrue(self.service.source_version_matches('task-A',self.old))
        self.assertEqual((self.old,self.current,self.entries),before)

    def test_every_immutable_field_difference_rejects_alias(self):
        """不同证据、文档/解析身份、内容、页位置或页文hash均拒绝。"""
        for key,value in {'evidence_id':'evidence:B','document_id':'doc:other-parser-config',
                          'source_sha256':'c'*64,'page_number':2,'text_sha256':'d'*64}.items():
            with self.subTest(field=key):
                self.current = {**self.old,'paper_id':'2603.00595v1',key:value}
                self.assertFalse(self.service.source_version_matches('task-A',self.old))

    def test_missing_extra_and_illegal_fields_fail_closed(self):
        """缺字段、额外字段、非法页码或hash不能被当作同版。"""
        cases = [{k:v for k,v in self.old.items() if k!=missing} for missing in self.old]
        cases += [{**self.old,'extra':'display'}]
        for value in [0,-1,True,'1',1.5]:
            cases.append({**self.old,'page_number':value})
        for key in ['paper_id','document_id','evidence_id','source_sha256','text_sha256']:
            for value in ['',None,False]:
                cases.append({**self.old,key:value})
        cases += [{**self.old,'source_sha256':'not-a-hash'},{**self.old,'text_sha256':'B'*64}]
        for item in cases:
            with self.subTest(reference=item):
                self.assertFalse(self.service.source_version_matches('task-A',item))

    def test_latest_wrong_version_wrong_paper_and_unverified_locator_reject(self):
        """不按同标题或同hash放宽未标版本、不同论文、跨版本或非支持地址。"""
        original = copy.deepcopy(self.entries['2603.00595v1'])
        replacements = [
            {'paper_url':'https://arxiv.org/abs/2603.00595','pdf_url':None},
            {'paper_url':'https://arxiv.org/abs/2603.00595v2','pdf_url':'https://arxiv.org/pdf/2603.00595v2'},
            {'paper_url':'https://arxiv.org/abs/2603.10978v1','pdf_url':None},
            {'paper_url':'https://example.invalid/same-title-v1','pdf_url':None},
            {'paper_url':None,'pdf_url':None},
            {'paper_url':'https://arxiv.org/abs/2603.00595v1','pdf_url':'https://arxiv.org/pdf/2603.00595v2'},
            {'version':'v2'},{'version':True},{'version':''},{'paper_id':'wrong-record'}]
        for change in replacements:
            with self.subTest(change=change):
                self.entries['2603.00595v1']={**original,**change,'title':'Same title'}
                self.assertFalse(self.service.source_version_matches('task-A',self.old))

    def test_explicit_identifier_cannot_conflict_with_stored_v1_urls(self):
        """独审指出的v2题录ID/v1链接矛盾和不同论文ID均拒绝。"""
        for identifier in ['2603.00595v2','2603.10978v1']:
            with self.subTest(identifier=identifier):
                self.current = {**self.old,'paper_id':identifier}
                self.entries[identifier]={'paper_id':identifier,'paper_url':'https://arxiv.org/pdf/2603.00595v1','version':None}
                self.assertFalse(self.service.source_version_matches('task-A',self.old))

    def test_missing_catalogue_or_malformed_metadata_cannot_prove_alias(self):
        """当前共享文档有身份不等于两别名已被证实。"""
        del self.entries['user-url:A']
        self.assertFalse(self.service.source_version_matches('task-A',self.old))
        self.db.fetchone = Mock(return_value={'metadata_json':'not-json'})
        self.assertFalse(self.service.source_version_matches('task-A',self.old))

    def test_task_evidence_authorization_is_never_bypassed(self):
        """没有当前evidence授权时保留原PermissionError，不由别名恢复材料权限。"""
        self.service.source = Mock(side_effect=PermissionError('outside task'))
        with self.assertRaises(PermissionError):
            self.service.source_version_matches('foreign-task',self.old)
        self.assertEqual(self.db.reads,[])

    def test_commit_comparison_reads_catalogue_in_supplied_transaction(self):
        """提交路径在同一事务读取别名，不绕过原子版本检查。"""
        transaction = Mock()
        transaction.one.side_effect = self.db.fetchone
        self.assertTrue(self.service.source_version_matches('task-A',self.old,transaction))
        self.service.source.assert_called_once_with('task-A',self.old['evidence_id'],transaction)
        self.assertEqual(transaction.one.call_count,2)

    def test_unpaginated_text_keeps_none_and_rejects_zero_page(self):
        """Jina无物理页码的None可比较，不把零页当作同一位置。"""
        self.old['page_number']=None; self.current['page_number']=None
        self.assertTrue(self.service.source_version_matches('task-A',self.old))
        self.current['page_number']=0
        self.assertFalse(self.service.source_version_matches('task-A',self.old))
