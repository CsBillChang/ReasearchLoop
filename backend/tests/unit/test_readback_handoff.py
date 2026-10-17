"""重复只读取证的行为路由测试；不以次数/Token额度代替进展判断。"""
import copy
import unittest
from research_loop.services.research_progress import readback_handoff


class ReadbackHandoffTests(unittest.TestCase):
    """阶段变化、合法补读、新来源与错误须分开处理。"""

    @staticmethod
    def read(n, kind='context_readback', target='x', status='succeeded'):
        """构造已结算的只读观察，不构造研究结论。"""
        return {'action_id':str(n),'tool_name':'read_paper','status':status,
            'arguments':{'target_ref':target,'start_page':7,'page_count':1},
            'observation':{'data':{'research_increment':{'kind':kind}}}}

    @staticmethod
    def stage(n, text='candidate'):
        """构造经现有进展服务提交后的候选记录。"""
        return {'action_id':str(n),'tool_name':'record_research_progress','status':'succeeded',
            'observation':{'data':{'research_progress':{'revision':n,'base_revision':n-1,
                'findings':[{'text':text}],'gaps':[],'next_question':''}}}}

    def test_first_readback_and_new_sources_are_allowed(self):
        """第一次恢复压缩原文不触发交接；新版本材料重新开放研究。"""
        self.assertIsNone(readback_handoff([self.read(1,'new_evidence'),self.read(2,target='other')]))
        self.assertIsNone(readback_handoff([self.read(1),self.read(2,'new_evidence')]))

    def test_same_read_without_stage_routes_to_handoff(self):
        """相同材料已实际再次返回，此后请求利用已有文字交接。"""
        self.assertEqual(readback_handoff([self.read(1,'new_evidence'),self.read(2)])['trigger_action_id'],'2')

    def test_changed_stage_reopens_targeted_read(self):
        """真实阶段变化后，为新缺口补读不与之前状态混算。"""
        self.assertIsNone(readback_handoff([self.read(1),self.stage(2),self.read(3)]))

    def test_rewriting_same_stage_does_not_reset_route(self):
        """仅刷新修订号或重复旧状态不算新研究进展。"""
        actions=[self.stage(1),self.read(2),self.stage(3),self.read(4)]
        self.assertIsNotNone(readback_handoff(actions))

    def test_failed_action_is_not_a_scientific_no_progress_verdict(self):
        """参数失败仍由原有失败合同处理，不冒充相同原文或反证。"""
        self.assertIsNone(readback_handoff([self.read(1),self.read(2,status='failed')]))

    def test_query_rewording_with_identical_hits_still_hands_off(self):
        """换查询词但反复返回同一原文片段，不制造新的材料进展。"""
        a=self.read(1,'same_evidence')
        a.update(tool_name='retrieve_evidence',arguments={'query':'first'})
        a['observation']['data']['hits']=[{'document_id':'doc','source_sha256':'v','page_number':3,'char_start':10,'text':'exact condition'}]
        b=copy.deepcopy(a);b.update(action_id='2',arguments={'query':'reworded'})
        self.assertIsNotNone(readback_handoff([a,b]))
        b['observation']['data']['hits'][0]['char_start']=30
        self.assertIsNone(readback_handoff([a,b]))
