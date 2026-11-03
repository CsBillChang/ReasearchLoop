"""实验层失败分级；只控制发送门，不替模型补参或改变业务工具观察。"""
from __future__ import annotations

import json
import re


BOUNDED_PARAMETER_RECOVERY = {
    'version': 'bounded-parameter-recovery-v1',
    'max_recoverable_failures': 2,
}


class FailedActionGate:
    """保留首错和后续失败；显式资格才允许少量参数更正。"""

    def __init__(self, policy=None):
        if policy is not None and policy != BOUNDED_PARAMETER_RECOVERY:
            raise ValueError('MECHANISM_RECOVERY_POLICY_NOT_SUPPORTED')
        self.policy = dict(policy) if policy is not None else None
        self.first_failed_action = None
        self.failed_actions = []
        self.case_stop = None
        self.global_stop = None
        self._seen = set()
        self._recoverable_signatures = set()

    @staticmethod
    def _classification(action):
        """只放行已知可更正的合同/参数错，未知类型仍关闭整批发送门。"""
        observation = action.get('observation') or {}
        data = observation.get('data') or {}
        code = data.get('error_code')
        blocker = (data.get('coverage_blocker') or {}).get('code')
        message = observation.get('message', '').lower()
        if (observation.get('kind') == 'validation_error' and observation.get('retryable')
                and re.match(r'^\d+ validation errors? for [a-z]', message)):
            # Pydantic拒绝时handler尚未执行；漏填hash与实际版本校验失败不同。
            return 'recoverable_parameter'
        # 业务观察未保留异常类型，因此先保守识别权限和当前身份约束。
        sensitive = ('permission', 'unauthorized', 'not authorized', 'forbidden', 'denied',
                     'outside', 'mismatch', 'hash', 'version', 'lease', 'budget', 'ledger',
                     'identity', 'cannot use', '越权', '权限', '未授权', '版本', '身份', '租约')
        if any(word in message for word in sensitive):
            return 'global_stop'
        if code == 'REPEATED_NONRETRYABLE_ACTION':
            return 'case_stop'
        if blocker in {'GOAL_SUPPORT_BINDING_MISSING', 'GOAL_QUALIFICATION_BLOCKED'}:
            return 'recoverable_parameter'
        if code in {'INVALID_TOOL_JSON', 'TOOL_DEPENDENCY_UNRESOLVED'}:
            return 'recoverable_parameter'
        if (action.get('tool_name') == 'set_research_plan' and observation.get('retryable')
                and message.startswith('研究计划至少需要一个active user/legacy且min_papers>0的材料目标')):
            # 计划服务用 execution_error 返回该明确参数反馈，不能按异常类别误停整批。
            return 'recoverable_parameter'
        if observation.get('kind') == 'validation_error' and observation.get('retryable'):
            return 'recoverable_parameter'
        return 'global_stop'

    def observe(self, actions):
        """消费真实行动记录，重复读回不重复扣更正机会。"""
        for action in actions:
            if action.get('status') != 'failed' or action['action_id'] in self._seen:
                continue
            self._seen.add(action['action_id'])
            if self.first_failed_action is None:
                self.first_failed_action = action
            classification = self._classification(action) if self.policy else 'legacy_global_stop'
            self.failed_actions.append({'action': action, 'classification': classification})
            if self.global_stop:
                continue
            reason = 'failed_action:' + action['tool_name']
            if classification in {'global_stop', 'legacy_global_stop'}:
                self.global_stop = reason
            elif classification == 'case_stop':
                self.case_stop = self.case_stop or reason
            else:
                signature = json.dumps([action.get('task_id'), action['tool_name'],
                    action.get('arguments')], sort_keys=True, ensure_ascii=False)
                if signature in self._recoverable_signatures:
                    self.case_stop = self.case_stop or 'repeated_parameter_failure:' + action['tool_name']
                self._recoverable_signatures.add(signature)
                attempts = sum(row['classification'] == 'recoverable_parameter' for row in self.failed_actions)
                if attempts > self.policy['max_recoverable_failures']:
                    self.case_stop = self.case_stop or 'parameter_recovery_exhausted:' + action['tool_name']
        return self.global_stop or self.case_stop

    def poll(self, db, task_id):
        """查询同根真实失败，不写数据库或工具反馈。"""
        if db is None or not task_id:
            return self.global_stop or self.case_stop
        rows = db.fetchall('SELECT a.* FROM actions a JOIN tasks t ON t.task_id=a.task_id '
            "WHERE (t.task_id=? OR t.root_task_id=?) AND a.status='failed' "
            'ORDER BY a.created_at,a.sequence_no,a.action_id', (task_id, task_id))
        actions = []
        for row in rows:
            action = dict(row)
            action['arguments'] = json.loads(action.pop('arguments_json'))
            action['observation'] = json.loads(action.pop('observation_json'))
            actions.append(action)
        return self.observe(actions)

    def audit(self):
        """失败原件和分级一并交付，局部停止不伪装成完成。"""
        return {'recovery_policy': self.policy, 'failed_action_history': self.failed_actions,
                'case_stop': self.case_stop}
