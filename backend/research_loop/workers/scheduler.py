"""本机简报调度器：只读DPR元数据，有限请求与租约恢复，不调用研究Agent或模型。"""
from __future__ import annotations
import json
import time
from research_loop.domain.subscriptions import now_utc
from research_loop.providers.discovery import DiscoveryProvider
from research_loop.services.subscriptions import SubscriptionService, request_lane
from research_loop.storage.mysql import Database


class BriefScheduler:
    """独立轮询已启用订阅；API创建配置不会在HTTP处理期间运行研究任务。"""

    def __init__(self, settings, *, provider_factory=None, clock=now_utc, after_receipt=None,
                 request_cap=102, duration_seconds=1800):
        """注入来源和时钟；默认每来源/查询最多两次，每轮最多102次物理RPC。

        Args:
            settings: 数据库与既有DPR连接配置。
            provider_factory: 组件测试替身，默认严格单请求的现有DPR适配器。
            clock: 日历时间；真实运行不修改系统时钟。
            after_receipt: 测试故障点，发生于响应落库后、发布前。
            request_cap: 单轮总请求上限，真实验收可收紧。
            duration_seconds: 单次持租执行时间上限；未查来源明确保留缺口。
        """
        self.db=Database(settings.database_url)
        self.service=SubscriptionService(self.db,clock)
        self.provider_factory=provider_factory or (lambda: DiscoveryProvider.from_settings(settings, timeout=30,max_date_windows=1,max_attempts=1,retry_wait_seconds=0))
        self.after_receipt=after_receipt
        self.request_cap=min(102,max(0,request_cap))
        self.duration_seconds=duration_seconds

    def run_once(self, subscription_id: str | None = None) -> list[dict]:
        """领取到期订阅并交付；已有活动租约跳过，不新建逐日补跑队列。"""
        results=[]
        subscriptions=self.service.list()
        for sub in subscriptions:
            if subscription_id and subscription_id!=sub['subscription_id']:
                continue
            run=self.service.claim(sub['subscription_id'])
            if run:
                try:
                    results.append(self.execute(run))
                except PermissionError:
                    results.append({'run_id':run['run_id'],'status':'cancelled_or_reclaimed'})
        return results

    def execute(self, run: dict) -> dict:
        """精确窗口复用收据，失败最多两次，然后原子交付已有候选与缺口。"""
        started=time.monotonic()
        provider=self.provider_factory()
        results=[]
        try:
            for window in run['windows']:
                result=None
                while time.monotonic()-started < self.duration_seconds:
                    attempt=self.service.begin_request(run,window,self.request_cap)
                    if 'cached' in attempt:
                        result=attempt['cached']
                        break
                    if attempt.get('exhausted'):
                        rows=self.db.fetchall('SELECT result_json FROM brief_requests WHERE run_id=? AND lane=? AND result_json IS NOT NULL ORDER BY attempt DESC',(run['run_id'],request_lane(window)))
                        if rows:
                            result=json.loads(rows[0]['result_json'])
                        break
                    try:
                        found=provider.search(window['query'],window['from'],window['to'],[window['source']],20)
                        source=next(s for s in found.source_results if s.source==window['source'])
                        status='truncated' if found.truncated and source.status in {'queried_success','queried_empty'} else source.status
                        result={'coverage':{**window,'status':status,'error':source.message if status not in {'queried_success','queried_empty'} else '', 'diagnostic':source.model_dump(mode='json')},
                                'candidates':[p.model_dump(mode='json') for p in found.candidates]}
                    except Exception as exc:
                        # 不保存可能含认证头的异常字符串；实际HTTP诊断由适配器脱敏保存。
                        result={'coverage':{**window,'status':'failed','error':type(exc).__name__},'candidates':[]}
                    self.service.settle_request(run,attempt['request_id'],result)
                    if self.after_receipt:
                        self.after_receipt(run,result)
                    if result['coverage']['status']!='failed':
                        break
                if result is None:
                    result={'coverage':{**window,'status':'budget_exhausted','error':'未取得可复用返回；请求或运行时限已用完'},'candidates':[]}
                results.append(result)
            return self.service.publish(run,results)
        finally:
            provider.close()

    def run_forever(self, interval: float = 30):
        """持续检查本机到期记录；休眠期间不执行，恢复时由claim合并日期窗口。"""
        import logging
        while True:
            try:
                self.run_once()
            except Exception:
                logging.exception('简报调度失败；已持久化记录保留，下次按租约恢复')
            time.sleep(interval)
