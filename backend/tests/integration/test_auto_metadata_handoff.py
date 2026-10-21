"""自动题录交接的隔离回归；网络只返回合成响应，不调用真实服务。"""
import unittest
from uuid import uuid4
from unittest.mock import patch

from integration import test_fallback_handoff as fixtures
from research_loop.domain.tasks import SavePolicy
from research_loop.providers.paper_acquisition import PaperAcquisitionError
from research_loop.tools.base import ToolContext


TITLE = fixtures.fixtures.TITLE
AUTHOR = fixtures.fixtures.AUTHOR
ORIGINAL = fixtures.ORIGINAL


class AutomaticMetadataHandoffTests(unittest.TestCase):
    """只验证当前用户明确身份，不能从模型计划或候选数推断关联。"""

    add_paper = fixtures.FallbackHandoffTests.add_paper
    paid_calls = fixtures.FallbackHandoffTests.paid_calls
    respond = fixtures.FallbackHandoffTests.respond
    run_acquire = fixtures.FallbackHandoffTests.run_acquire
    save_action = fixtures.FallbackHandoffTests.save_action

    def setUp(self):
        """每例建立独立任务及临时SQLite/显式测试MySQL，使用MockTransport。"""
        fixtures.FallbackHandoffTests.setUp(self)
        self.set_prompt(f'请核验 {AUTHOR} 的《{TITLE}》。先尝试 {ORIGINAL} 原文；失败后允许 Search/Fetch 寻找合法公开 PDF。不保存文件。')

    def set_prompt(self, prompt):
        """通过持久用户指令设置身份依据，模拟模型参数不能代替该依据。"""
        self.db.execute('UPDATE tasks SET prompt=? WHERE task_id=?', (prompt, self.task))

    def assert_reference_required(self, expected_status=None):
        """不确定身份仅允许原址尝试，不发送Search/Fetch或产生页证据。"""
        result = self.run_acquire()
        self.assertFalse(result.ok, result.model_dump())
        self.assertEqual(result.data['status'], 'metadata_reference_required')
        if expected_status:
            self.assertEqual(result.data['diagnostic']['automatic_metadata_reference']['status'], expected_status)
        self.assertEqual(self.paid_calls(), [])
        self.assertEqual(self.db.list_task_evidence(self.task), [])
        return result

    def test_exact_current_title_authors_continue_in_same_tool_call(self):
        """单次获取完成原址失败→明确题录→Search/Fetch→PDF页，不要求LLM补参。"""
        result = self.run_acquire()
        self.assertTrue(result.ok, result.model_dump())
        handoff = result.data['identity_handoff']
        self.assertEqual(handoff['automatic_metadata_reference']['status'], 'unique_match')
        self.assertEqual(handoff['metadata_reference']['paper_id'], self.reference.paper_id)
        self.assertEqual(result.data['returned_pages'], [1, 2])
        self.assertEqual(len(self.paid_calls()), 2)
        self.assertTrue(handoff['original_locator_identity_unverified'])
        self.assertTrue(result.data['alternate_version'])
        self.assertEqual(self.db.get_paper(handoff['resolved_paper_id'])['metadata']['title'], '')
        self.assertNotEqual(result.data['paper_id'], self.reference.paper_id)
        self.assertEqual(self.db.get_task(self.task)['save_policy'], 'deny')

    def test_primary_success_needs_neither_metadata_nor_search(self):
        """普通原PDF成功不进入额外身份解析或付费备用步骤。"""
        self.original_succeeds = True
        self.set_prompt('请读取 ' + ORIGINAL)
        result = self.run_acquire(page_count=1)
        self.assertTrue(result.ok, result.model_dump())
        self.assertNotIn('automatic_metadata_reference', result.data['identity_handoff'])
        self.assertEqual(self.paid_calls(), [])

    def test_single_candidate_without_identity_is_not_a_match(self):
        """列表仅一个候选仍不证明与用户URL相关。"""
        self.set_prompt('请读取 ' + ORIGINAL)
        self.assert_reference_required('explicit_title_authors_or_doi_required')

    def test_title_without_authors_or_doi_does_not_match(self):
        """仅有完整标题不能自动绑定跨站身份。"""
        self.set_prompt(f'请读取《{TITLE}》，原址 {ORIGINAL}')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_partial_and_conflicting_author_names_do_not_match(self):
        """姓氏、名字子串及额外作者不作为完整作者列表匹配。"""
        for authors in ('Author', 'Synthetic Author Junior', 'Another Author', AUTHOR + '、Another Author'):
            with self.subTest(authors=authors):
                self.set_prompt(f'请核验 {authors} 的《{TITLE}》。原址 {ORIGINAL}')
                self.assert_reference_required('no_authorized_metadata_match')

    def test_multiple_explicit_urls_or_titles_do_not_autobind(self):
        """两个用户目标不靠当前授权列表剩一篇来消除歧义。"""
        for extra in ('另一个原址 https://openreview.net/forum?id=synthetic-other', '另请比较《Another Synthetic Paper》'):
            with self.subTest(extra=extra):
                self.set_prompt(f'请核验 {AUTHOR} 的《{TITLE}》。原址 {ORIGINAL} 原文。{extra}')
                self.assert_reference_required('multiple_explicit_targets')

    def test_two_authorized_versions_remain_ambiguous(self):
        """同题名作者的两个arXiv版本不会自动选择其中一个。"""
        self.add_paper(paper_id='synthetic-second:' + uuid4().hex, source='arxiv',
                       paper_url='https://arxiv.org/abs/2601.12345v2', version='v2')
        self.assert_reference_required('ambiguous_authorized_metadata')

    def test_exact_doi_can_select_but_does_not_override_title_conflict(self):
        """DOI可作为明确依据，同时存在不同题名时拒绝自动绑定。"""
        self.reference = self.add_paper(paper_url=fixtures.ARXIV, pdf_url=fixtures.PDF_URL,
                                       version='v1', source='arxiv', doi='10.1234/synthetic-test')
        self.set_prompt(f'请读取 DOI: 10.1234/synthetic-test。原址 {ORIGINAL}')
        reference, detail = self.service._automatic_metadata_reference(self.task)
        self.assertEqual(reference.paper_id, self.reference.paper_id)
        self.assertEqual(detail['basis'], 'current_user_doi')
        self.set_prompt(f'请读取《Another Synthetic Paper》，DOI: 10.1234/synthetic-test。原址 {ORIGINAL}')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_doi_does_not_override_author_conflict(self):
        """同DOI但用户明确不同作者仍需澄清。"""
        self.add_paper(paper_url=fixtures.ARXIV, pdf_url=fixtures.PDF_URL,
                       version='v1', source='arxiv', doi='10.1234/synthetic-test')
        self.set_prompt(f'请核验 Another Author 的《{TITLE}》。DOI: 10.1234/synthetic-test。原址 {ORIGINAL}')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_doi_url_continues_complete_acquisition(self):
        """显式 DOI URL 能交接，Fetch仍须核对该DOI，不把线索当核验完成。"""
        self.add_paper(paper_url=fixtures.ARXIV, pdf_url=fixtures.PDF_URL,
                       version='v1', source='arxiv', doi='https://doi.org/10.1234/synthetic-test')
        self.fetch_content += '\n DOI: 10.1234/synthetic-test'
        self.set_prompt(f'请读取 https://doi.org/10.1234/synthetic-test ，先尝试 {ORIGINAL} 原文。')
        result = self.run_acquire()
        self.assertTrue(result.ok, result.model_dump())
        self.assertEqual(result.data['returned_pages'], [1, 2])
        self.assertEqual(result.data['identity_handoff']['automatic_metadata_reference']['basis'], 'current_user_doi')
        self.assertEqual(len(self.paid_calls()), 2)

    def test_negated_doi_is_not_a_positive_identity_claim(self):
        """用户否认的DOI不可被当成肯定身份线索。"""
        self.add_paper(doi='10.1234/synthetic-test')
        self.set_prompt(f'原址 {ORIGINAL} 原文。它不是 DOI: 10.1234/synthetic-test。')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_negated_or_background_title_does_not_authorize_fallback(self):
        """明确否定/背景提及即使完整作者题名匹配，也不是当前URL的肯定身份。"""
        claims = (
            f'{AUTHOR} 的《{TITLE}》不是本次目标论文。',
            f'请核验 {AUTHOR} 的《{TITLE}》，它不是本次目标论文。',
            f'请核验 {AUTHOR} 的《{TITLE}》。不要用该题录来匹配该URL。',
            f'{AUTHOR} 的《{TITLE}》是我以前读过的背景资料。',
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.set_prompt(claim + f'请读取 {ORIGINAL} 原文；原址不可读时允许合法公开备用。')
                self.assert_reference_required('identity_intent_requires_explicit_reference')
                self.assertIn(('GET', fixtures.ORIGINAL_PDF), self.calls)
                identities = self.db.fetchall('SELECT paper_id FROM task_papers WHERE task_id=?', (self.task,))
                self.assertFalse(any(row['paper_id'].startswith('webvariant:') for row in identities))

    def test_background_doi_and_changed_target_do_not_autobind(self):
        """背景DOI和后续另一篇目标不因只有一个候选而自动绑定。"""
        self.add_paper(doi='10.1234/synthetic-test')
        self.set_prompt(f'有关背景参见 DOI: 10.1234/synthetic-test。现在请读取另一篇论文 {ORIGINAL} 原文。')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_positive_old_identity_does_not_bind_a_new_target(self):
        """前文肯定读取题名也不能被隐式挪给后续明确的另一篇URL。"""
        self.set_prompt(f'请核验 {AUTHOR} 的《{TITLE}》。现在请读取另一篇论文 {ORIGINAL} 原文。')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_later_identity_denial_overrides_prior_positive_reference(self):
        """后续用户撤回题录关联时，不复用根指令中的旧肯定身份。"""
        clarification_id = str(uuid4())
        self.db.save_clarification({'clarification_id': clarification_id, 'task_id': self.task, 'question': '确认题录对应关系？'})
        self.db.answer_clarification(clarification_id, '上述题录不是这个URL对应的论文，不要用来匹配该URL。')
        self.assert_reference_required('identity_intent_requires_explicit_reference')

    def test_identity_denial_does_not_block_successful_original_pdf(self):
        """否定参考身份只禁止自动备用绑定，已授权原址可正常读出。"""
        self.original_succeeds = True
        self.set_prompt(f'{AUTHOR} 的《{TITLE}》不是本次目标论文。请读取 {ORIGINAL} 原文。')
        result = self.run_acquire(page_count=1)
        self.assertTrue(result.ok, result.model_dump())
        self.assertEqual(result.data['returned_pages'], [1])
        self.assertNotIn('metadata_reference', result.data['identity_handoff'])
        self.assertEqual(self.paid_calls(), [])

    def test_other_task_or_role_reference_cannot_be_used(self):
        """用户明确身份不替代当前任务及角色授权。"""
        self.db.execute('DELETE FROM task_papers WHERE task_id=? AND paper_id=?', (self.task, self.reference.paper_id))
        self.assert_reference_required('no_authorized_metadata_match')
        self.db.grant_paper(self.task, self.reference.paper_id, self.task)
        candidate, detail = self.service._automatic_metadata_reference(self.task, ['different-assigned-id'])
        self.assertIsNone(candidate)
        self.assertEqual(detail['status'], 'no_authorized_metadata_match')

    def test_model_plan_and_child_goal_do_not_supply_identity(self):
        """子任务模型目标不能补造根用户未表达的身份依据。"""
        self.set_prompt('请读取 ' + ORIGINAL)
        child = 'web-test-' + uuid4().hex
        self.db.create_task(child, None, 'synthetic', f'请核验 {AUTHOR} 的《{TITLE}》。', 'deny', [],
                            parent_task_id=self.task, root_task_id=self.task)
        self.db.grant_paper(child, self.reference.paper_id, self.task)
        context = ToolContext(child, SavePolicy.deny, services={'papers': self.service})
        result = self.registry.execute('acquire_paper_pdf', {'target_ref': ORIGINAL}, context)
        self.assertFalse(result.ok, result.model_dump())
        self.assertEqual(result.data['status'], 'metadata_reference_required')
        self.assertEqual(self.paid_calls(), [])

    def test_only_answered_current_user_clarification_can_supply_identity(self):
        """未回答的模型澄清问题不是用户事实，已提交答案才进入匹配。"""
        self.set_prompt('请读取 ' + ORIGINAL)
        clarification_id = str(uuid4())
        identity = f'请核验 {AUTHOR} 的《{TITLE}》。'
        self.db.save_clarification({'clarification_id': clarification_id, 'task_id': self.task, 'question': identity})
        reference, _ = self.service._automatic_metadata_reference(self.task)
        self.assertIsNone(reference)
        self.db.answer_clarification(clarification_id, identity)
        result = self.run_acquire()
        self.assertTrue(result.ok, result.model_dump())
        self.assertEqual(result.data['identity_handoff']['metadata_reference']['paper_id'], self.reference.paper_id)

    def test_auto_reference_preserves_explicit_locator_version_check(self):
        """原URL明确v2而参考为v1时，即使题名作者匹配仍拒绝关联。"""
        target = 'https://arxiv.org/abs/2601.12345v2'
        self.set_prompt(f'请核验 {AUTHOR} 的《{TITLE}》。原址 {target} 原文。')
        failure = PaperAcquisitionError('PDF_HTTP_ERROR', 'synthetic not found', acquisition={'status': 404})
        with patch.object(self.service, 'read', side_effect=failure):
            result = self.run_acquire(target)
        self.assertFalse(result.ok, result.model_dump())
        self.assertEqual(result.data['error_code'], 'PDF_METADATA_VERSION_MISMATCH')
        self.assertEqual(self.calls, [])
        self.assertEqual(self.db.list_task_evidence(self.task), [])

    def test_explicit_metadata_reference_still_works_without_auto_identity(self):
        """保留旧显式补参入口；它仍只是待Fetch核实的线索。"""
        self.set_prompt('请读取 ' + ORIGINAL)
        result = self.run_acquire(metadata_ref=self.reference.paper_id)
        self.assertTrue(result.ok, result.model_dump())
        self.assertNotIn('automatic_metadata_reference', result.data['identity_handoff'])
        self.assertTrue(result.data['identity_handoff']['original_locator_identity_unverified'])

    def test_auto_reference_cannot_cross_explicit_version_or_page_scope(self):
        """自动参考不会解除用户指定版本/物理页限制。"""
        for restriction in ('只使用指定版本v1。', '只读取物理第1页。'):
            with self.subTest(restriction=restriction):
                self.set_prompt(f'请核验 {AUTHOR} 的《{TITLE}》。原址 {ORIGINAL} 原文。{restriction}')
                result = self.run_acquire()
                self.assertFalse(result.ok, result.model_dump())
                self.assertEqual(result.data['status'], 'unconfirmed')
                self.assertEqual(self.db.list_task_evidence(self.task), [])
        self.assertFalse(any('/fetch' in url for _, url in self.calls))

    def test_explicit_hard_host_scope_blocks_alternate(self):
        """仅原站的硬约束不被自动身份和来源偏好处理覆盖。"""
        self.set_prompt(f'请核验 {AUTHOR} 的《{TITLE}》。原址 {ORIGINAL} 原文。只使用 openreview.net 获取原文。')
        result = self.run_acquire()
        self.assertFalse(result.ok, result.model_dump())
        self.assertEqual(result.data['status'], 'unconfirmed')
        self.assertFalse(any('/fetch' in url for _, url in self.calls))
        self.assertEqual(self.db.list_task_evidence(self.task), [])

    def test_complete_author_list_accepts_unicode_spacing_and_punctuation(self):
        """匹配完整作者和标点归一后的全标题，不依赖特定论文硬编码。"""
        self.add_paper(authors=['Synthetic Author', 'Second Author'], paper_url=fixtures.ARXIV,
                       pdf_url=fixtures.PDF_URL, source='arxiv', version='v1')
        self.set_prompt(f'请核验 Synthetic Author、Second Author 的《{TITLE.upper()}》。原址 {ORIGINAL}')
        reference, detail = self.service._automatic_metadata_reference(self.task)
        self.assertEqual(reference.paper_id, self.reference.paper_id)
        self.assertEqual(detail['status'], 'unique_match')

    def test_explicit_chinese_author_does_not_require_ascii_spacing(self):
        """中文明确核验句式不需要人为插入空格，不放宽完整作者比较。"""
        self.add_paper(authors=['张三'], paper_url=fixtures.ARXIV, pdf_url=fixtures.PDF_URL,
                       source='arxiv', version='v1')
        self.set_prompt(f'请核验张三的《{TITLE}》。原址 {ORIGINAL} 原文。不保存文件。')
        reference, detail = self.service._automatic_metadata_reference(self.task)
        self.assertEqual(reference.paper_id, self.reference.paper_id)
        self.assertEqual(detail['status'], 'unique_match')


if __name__ == '__main__':
    unittest.main()
