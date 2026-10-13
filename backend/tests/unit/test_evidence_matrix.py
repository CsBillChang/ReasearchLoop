"""矩阵结构与出处验证：合成页明确分离于真实论文语义验收。"""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from research_loop.domain.research import DIMENSIONS, EvidenceMatrixInput
from research_loop.domain.tasks import ActionRecord, Observation
from research_loop.services.evidence_matrix import EvidenceMatrixService, quote_span
from research_loop.storage.mysql import Database


def seed(db, tid, suffix=''):
    """建立同任务的两页合成文档，返回合法五维参数。"""
    did='doc-'+(suffix or uuid4().hex);pid='paper-'+did
    db.save_paper(pid,{'paper_id':pid,'title':'合成方法 '+did})
    db.save_document({'document_id':did,'paper_id':pid,'content_sha256':hashlib.sha256(did.encode()).hexdigest(),
        'parser_name':'fixture','parser_version':'1','config_sha256':'b'*64,'total_pages':2,'cache_path':'not-a-real-pdf'})
    texts=['We study object detection. Our method uses language queries. Evaluation uses Dataset A. AP is 42 with backbone-only measurement. No deployment study was conducted.',
           'Another setting reports AP of 39.']
    refs=[]
    for i,text in enumerate(texts,1):
        eid=did+'-p'+str(i);refs.append(eid)
        db.save_page({'evidence_id':eid,'document_id':did,'page_number':i,'text':text})
        db.mark_task_evidence(tid,eid)
    quotes=['We study object detection.','Our method uses language queries.','Evaluation uses Dataset A.','AP is 42 with backbone-only measurement.']
    cells=[{'dimension':d,'status':'evidenced','value':q,'conditions':['Dataset A; backbone-only'] if d=='metric' else [],
            'references':[{'evidence_id':refs[0],'quote':q}]} for d,q in zip(DIMENSIONS,quotes)]
    cells.append({'dimension':'limitation','status':'missing','reason':'当前两页没有部署硬件说明，不能推断全篇未报告。'})
    return {'question':'比较原文中的设置','rows':[{'document_id':did,'inspected_evidence_ids':refs,'cells':cells}]}


def persist(db,tid,args,sequence=1,owner=None):
    """模拟通用图的先意图后成功观察事务，不伪造语义审校。"""
    service=EvidenceMatrixService(db);matrix=service.prepare(tid,EvidenceMatrixInput.model_validate(args))
    action=ActionRecord(task_id=tid,sequence=sequence,tool_name='record_evidence_matrix',arguments=args)
    db.save_action(action.model_dump(mode='json'))
    action.status='succeeded';action.observation=Observation(ok=True,kind='success',data={'evidence_matrix':matrix}).model_dump()
    service.commit_action(action.model_dump(mode='json'),owner)
    return matrix


class MatrixUnitTests(unittest.TestCase):
    """身份与内容版本在SQLite中独立测试，不发网络请求。"""

    def setUp(self):
        """隔离每例数据，保留真实服务而非绕过校验。"""
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.db=Database('sqlite:///'+str(Path(temp.name)/'matrix.db'));self.db.ensure_schema()
        self.db.create_task('root',None,'hash','matrix','deny',[])
        self.db.claim_task('w',task_id='root');self.service=EvidenceMatrixService(self.db)
        self.args=seed(self.db,'root');self.row=self.args['rows'][0]

    def test_provenance_and_missing_scope(self):
        """页码/hash服务端派生，缺证值留空，不写文件或审校状态。"""
        matrix=persist(self.db,'root',self.args)
        row=matrix['rows'][0];ref=row['cells'][0]['references'][0]
        page=self.db.get_page(ref['evidence_id'])
        self.assertEqual(page['text'][ref['start']:ref['end']],ref['quote'])
        self.assertEqual(row['scope'][0]['page_number'],1)
        self.assertEqual(row['cells'][-1]['value'],'')
        self.assertEqual(self.service.delivery_ref('root',matrix['matrix_hash'])['cell_counts']['missing'],1)
        self.assertEqual(self.db.get_task('root')['quality_status'],'unreviewed')

    def test_schema_rejects_untraceable_and_duplicate_dimensions(self):
        """缺证伪事实、无引句、指标无条件、冲突单引句、重复维度均拒绝。"""
        mutations=[lambda x:x['rows'][0]['cells'][-1].update(value='The paper never discusses hardware.'),
                   lambda x:x['rows'][0]['cells'][0].update(references=[]),
                   lambda x:x['rows'][0]['cells'][3].update(conditions=[]),
                   lambda x:x['rows'][0]['cells'][0].update(status='conflict',reason='contradiction'),
                   lambda x:x['rows'][0]['cells'][0].update(dimension='method'),
                   lambda x:x['rows'][0].update(paper_id='invented')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad=copy.deepcopy(self.args);mutate(bad)
                with self.assertRaises(ValueError):EvidenceMatrixInput.model_validate(bad)

    def test_wrong_task_and_document_and_fabricated_quote(self):
        """库中存在不等于当前任务可用；他篇引句不能绑定到本行。"""
        self.db.create_task('foreign',None,'h','other','deny',[])
        with self.assertRaises(ValueError):self.service.prepare('foreign',EvidenceMatrixInput(**self.args))
        other=seed(self.db,'root')
        bad=copy.deepcopy(self.args);bad['rows'][0]['inspected_evidence_ids']=other['rows'][0]['inspected_evidence_ids']
        with self.assertRaises(ValueError):self.service.prepare('root',EvidenceMatrixInput(**bad))
        bad=copy.deepcopy(self.args);bad['rows'][0]['cells'][0]['references'][0]['quote']='The result is 99 percent.'
        with self.assertRaises(ValueError):self.service.prepare('root',EvidenceMatrixInput(**bad))

    def test_only_whitespace_normalization(self):
        """不把42.0改成420，不删否定词，也不猜重复短语位置。"""
        text='The AP\n is 42.0, not 420.'
        start,end=quote_span(text,'The AP is 42.0, not 420.')
        self.assertEqual((start,end),(0,len(text)))
        for quote in ['The AP is 420, not 420.','The AP is 42.0, 420.']:
            with self.assertRaises(ValueError):quote_span(text,quote)
        with self.assertRaises(ValueError):quote_span('abc abc','abc')

    def test_line_wrap_offsets_and_ambiguous_matches(self):
        """换行归一只用于匹配，保存引句仍逐字来自唯一原文区间。"""
        raw = '前缀\n  The query-\r\n evidence model is fine-tuned fol-\nlowing prior work. 后缀'
        quote = 'The query-evidence model is fine-tuned following prior work.'
        start,end = quote_span(raw,quote)
        self.assertEqual(raw[start:end],raw[raw.index('The'):raw.index(' 后缀')])
        self.assertIn('fol-\nlowing',raw[start:end])
        for source,quote in [('fusion fu-\nsion','fusion'),('aaaa','aaa'),('value 2e-\n4','value 2e4'),
                             ('The model is not good.','The model is good.'),('re-covered data','recovered data')]:
            with self.subTest(source=source),self.assertRaises(ValueError):
                quote_span(source,quote)

    def test_reports_all_cell_locations_without_committing(self):
        """同次反馈包含每处错引的行、维度、引用序号，且不留下半份矩阵。"""
        for cell in self.row['cells'][:2]:
            cell['references'][0]['quote']='This invented sentence must be rejected.'
        with self.assertRaises(ValueError) as raised:
            self.service.prepare('root',EvidenceMatrixInput(**self.args))
        error=str(raised.exception)
        for token in ('rows[0]','cells[0] dimension=question','cells[1] dimension=method',
                      'references[0]',self.row['document_id'],self.row['inspected_evidence_ids'][0]):
            self.assertIn(token,error)
        self.assertEqual(self.service.history('root'),[])

    def test_reason_error_includes_document_and_dimension(self):
        """结构错误继续拒绝，工具观察给出两处确切缺字段的位置。"""
        from research_loop.tools.base import ToolContext, execute_spec
        from research_loop.tools.delegation import specs
        from research_loop.domain.tasks import SavePolicy
        self.row['cells'][0].update(status='uncertain',reason='')
        self.row['cells'][1].update(status='uncertain',reason='')
        spec=next(s for s in specs() if s.name=='record_evidence_matrix')
        result=execute_spec(spec,self.args,ToolContext(task_id='root',save_policy=SavePolicy.deny))
        self.assertFalse(result.ok)
        for token in (self.row['document_id'],'dimension=question','dimension=method','requires reason'):
            self.assertIn(token,result.message)

    def test_version_replay_history_and_stale_delivery(self):
        """同内容不新造版本；修改需前版，旧版不能再次用于当前交付。"""
        first=persist(self.db,'root',self.args)
        self.assertEqual(persist(self.db,'root',self.args,2)['matrix_hash'],first['matrix_hash'])
        self.assertEqual(len(self.service.history('root')),1)
        updated=copy.deepcopy(self.args);updated['question']='新的比较问题'
        with self.assertRaises(ValueError):self.service.prepare('root',EvidenceMatrixInput(**updated))
        updated['base_matrix_hash']=first['matrix_hash'];second=persist(self.db,'root',updated,3)
        self.assertEqual(len(self.service.history('root')),2)
        with self.assertRaises(ValueError):self.service.delivery_ref('root',first['matrix_hash'])
        self.assertEqual(self.service.latest('root')['matrix_hash'],second['matrix_hash'])

    def test_source_mutation_is_visible_and_blocks_delivery(self):
        """原文文本或版本改变时旧产物可查，但不被当作当前有效矩阵。"""
        matrix=persist(self.db,'root',self.args)
        self.db.execute('UPDATE page_evidence SET text=? WHERE evidence_id=?',('changed',self.row['inspected_evidence_ids'][0]))
        self.assertEqual(self.service.latest('root')['integrity'],'stale')
        with self.assertRaises(ValueError):self.service.delivery_ref('root',matrix['matrix_hash'])

    def test_conflict_preserves_distinct_excerpts(self):
        """原文两种设置可并列，程序只验证出处而不判断比较成立。"""
        cell=self.row['cells'][3];cell.update(status='conflict',reason='设置不同，不能直接比较')
        cell['references'].append({'evidence_id':self.row['inspected_evidence_ids'][1],'quote':'Another setting reports AP of 39.'})
        matrix=persist(self.db,'root',self.args)
        self.assertEqual(matrix['semantic_review'],'not_performed')
        self.assertEqual(len(matrix['rows'][0]['cells'][3]['references']),2)

    def test_cancel_or_lost_lease_rejects_commit(self):
        """计算完成后取消/失权也不能把成功矩阵写入行动。"""
        self.db.update_task('root',cancel_requested=1)
        with self.assertRaises(PermissionError):persist(self.db,'root',self.args,owner=('root','w','wrong'))
        self.assertEqual(self.service.history('root'),[])

    def test_no_physical_page_keeps_null(self):
        """Jina全文无物理页码时不伪造第1页。"""
        eid=self.row['inspected_evidence_ids'][0]
        self.db.execute('UPDATE page_evidence SET page_number=0 WHERE evidence_id=?',(eid,))
        matrix=persist(self.db,'root',self.args)
        self.assertIsNone(matrix['rows'][0]['cells'][0]['references'][0]['page_number'])
