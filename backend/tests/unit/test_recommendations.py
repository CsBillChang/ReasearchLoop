"""候选身份、画像优先级和时区边界的固定组件例。"""
import unittest
from datetime import datetime, timezone
from research_loop.services.recommendations import paper_identity,rank_candidates,markdown_text
from research_loop.domain.subscriptions import SubscriptionInput,schedule_slot


class RecommendationTests(unittest.TestCase):
    """不调用来源、模型或数据库的排名合同。"""
    def test_explicit_priority_no_title_dedup(self):
        """相同slot显式偏好覆盖推断；相似标题不是同篇身份。"""
        config={'directions':['detection'],'sources':['arxiv'],'use_profile':True,'limit':5}
        base={'source':'arxiv','title':'language detection','_queries':['detection']}
        p1={**base,'paper_id':'2501.00001v1'};p2={**base,'paper_id':'2501.00002v1'}
        self.assertNotEqual(paper_identity(p1),paper_identity(p2))
        self.assertNotEqual(paper_identity(p1),paper_identity({**p1,'paper_id':'2501.00001v2'}))
        profile={'explicit_preferences':[{'slot':'interest','content':'language','memory_id':'explicit','version':1,'kind':'explicit'}], 'inferred_interests':[{'slot':'interest','content':'detection','memory_id':'inferred','version':1,'kind':'inferred'}]}
        items=rank_candidates([p1,p2],config,profile,set(),set(),set())
        self.assertEqual(len(items),2);self.assertEqual(items[0]['memory_sources'][0]['memory_id'],'explicit')
        self.assertNotIn('inferred',str(items))

    def test_timezone_dst_and_user_validation(self):
        """北京时间跨日和纽约DST重复/不存在小时不产生同日双跑。"""
        now=datetime(2026,10,10,23,tzinfo=timezone.utc)
        self.assertEqual(schedule_slot({'timezone':'Asia/Shanghai','at':'09:00'},now,following=True).isoformat(),'2026-10-11T01:00:00+00:00')
        spring=datetime(2026,3,8,6,tzinfo=timezone.utc)
        self.assertEqual(schedule_slot({'timezone':'America/New_York','at':'02:30'},spring,following=True).hour,7)
        fall=datetime(2026,11,1,6,15,tzinfo=timezone.utc)
        self.assertEqual(schedule_slot({'timezone':'America/New_York','at':'01:30'},fall,following=True).day,2)
        with self.assertRaises(ValueError):SubscriptionInput(directions=[' '],idempotency_key='a')
        with self.assertRaises(ValueError):SubscriptionInput(directions=['a'],idempotency_key='a',limit=6)

    def test_metadata_is_literal_text(self):
        """上游摘要中的HTML/Markdown不被直接当指令链接渲染。"""
        text=markdown_text('<script>x</script> [evil](javascript:x)')
        self.assertNotIn('<script>',text);self.assertIn('\\[',text)
