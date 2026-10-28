"""在临时数据库运行接续实验及回归，导出合成演示和可审阅结果。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend'), str(ROOT / 'backend/tests')]


def main() -> int:
    """阻断网络后运行指定测试；只允许新建本worktree的本机工件目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        parser.error('输出必须是当前worktree artifacts下不存在的新目录')
    output.mkdir(parents=True)
    # 清掉外部追踪及显式服务测试配置，避免已有shell设置改变隔离范围。
    for key in list(os.environ):
        if (key.startswith(('LANGSMITH_', 'LANGCHAIN_', 'RESEARCHLOOP_', 'DEEPSEEK_', 'SUPABASE_', 'OPENALEX_'))
                or key == 'DATABASE_URL' or key.endswith('_TEST_DATABASE_URL')):
            os.environ.pop(key)
    os.environ['LANGSMITH_TRACING'] = 'false'
    from research_loop.storage.mysql import Database
    original_init = Database.__init__

    def isolated_database(database, url, *positional, **keyword):
        """拒绝创建任何MySQL连接；夹具目录和数据库由测试自身临时创建。"""
        if not url.startswith('sqlite:///'):
            raise AssertionError('离线接续实验只允许显式SQLite')
        path = Path(url[len('sqlite:///'):]).resolve()
        if not path.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise AssertionError('离线接续实验数据库必须位于系统临时目录')
        return original_init(database, url, *positional, **keyword)

    modules = [
        'backend.tests.unit.test_continuation_memory',
        'backend.tests.unit.test_continuation_real_gate',
        'backend.tests.unit.test_continuation_real_runner',
        'backend.tests.integration.test_continuation_memory',
        'backend.tests.integration.test_plan_quote_contiguity',
        'backend.tests.integration.test_zero_paper_plan_routing',
        'backend.tests.integration.test_plan_ownership_screening',
        'backend.tests.unit.test_context_continuity',
        'backend.tests.unit.test_context_workspace',
        'backend.tests.unit.test_context_working_set',
        'backend.tests.unit.test_interest_contracts',
        'backend.tests.integration.test_context_continuity',
        'backend.tests.integration.test_memory',
        'backend.tests.integration.test_behavioral_interests',
        'backend.tests.integration.test_history_attachment_grants',
    ]
    with patch('socket.socket.connect', side_effect=AssertionError('离线实验禁止网络')), \
            patch('socket.socket.connect_ex', side_effect=AssertionError('离线实验禁止网络')), \
            patch.object(Database, '__init__', isolated_database):
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        with (output / 'tests.log').open('w') as stream, redirect_stderr(stream):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report = {'scope': '临时SQLite、合成数据和模型替身；非真实模型语义验收',
            'tests_run': result.testsRun, 'failures': len(result.failures),
            'errors': len(result.errors), 'skipped': len(result.skipped),
            'passed': result.wasSuccessful(), 'real_model_calls': 0,
            'network_allowed': False, 'modules': modules,
            'failed_cases': [test.id() for test, _ in [*result.failures, *result.errors]]}
        (output / 'tests.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if result.wasSuccessful():
            from backend.tests.integration.test_continuation_memory import ContinuationMemoryTests, ContinuationProvider
            case = ContinuationMemoryTests()
            case.setUp()
            try:
                first = case.submit('合成讨论：先记录旧比较口径，不保存。')
                old_answer = '旧合成结论：采用固定分母比较。'
                case.execute(first, ContinuationProvider(old_answer, case.candidate(old_answer)))
                second = case.submit('合成更正：分母应随保留集合变化，请接着解释，不保存。')
                answer = '新合成结论：分母随保留集合变化，删除正确项也会改变比例。'
                case.execute(second, ContinuationProvider(answer, case.correction_candidate(answer, first)))
                third = case.submit('按更正后的口径继续，不保存。')
                case.continuation.enable_new_task(third)
                context = case.continuation.context(third, [first])
                demo = {'scope': '合成模型响应演示，不是真实模型结果',
                    'original_answer': case.record(first)['answer'],
                    'original_memory': case.record(first)['continuation_memory'],
                    'corrected_answer': case.record(second)['answer'],
                    'corrected_memory': case.record(second)['continuation_memory'],
                    'next_context': context, 'real_model_calls': 0}
                (output / 'synthetic-demo.json').write_text(json.dumps(demo, ensure_ascii=False, indent=2))
            finally:
                case.doCleanups()
    print(json.dumps(report, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
