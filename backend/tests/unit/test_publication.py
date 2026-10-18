"""验证公开边界扫描不会放行私有配置，也不会回显发现的密钥。"""
import unittest
from pathlib import Path

from scripts.check_publication import check_file, REVIEWED_SCREENSHOTS


class PublicationTests(unittest.TestCase):
    """覆盖公开文本、敏感模式、路径、大小及文件模式。"""

    def test_source_and_empty_configuration_are_allowed(self):
        """普通源码和空凭据模板可以发布。"""
        self.assertEqual(check_file("backend/research_loop/example.py", b'"""module"""'), [])
        self.assertEqual(check_file(".env.example", b"DEEPSEEK_API_KEY=\n"), [])

    def test_private_locations_and_attachments_are_rejected(self):
        """运行数据、历史说明、依赖目录和论文不进入公开树。"""
        for name in (".env", "docs/archive/old.md", "docs/status.md", "docs/evidence/retrieval.json",
                     "data/README.md", "artifacts/README.md", "THIRD_PARTY_NOTICES.md",
                     "AGENTS.md", "agents.md", "NOTICE", "notice",
                     "data/profile.json", "artifacts/response.json", "frontend/node_modules/x.js", "docs/paper.pdf"):
            with self.subTest(name=name):
                self.assertTrue(check_file(name, b"private"))

    def test_secrets_and_personal_paths_are_not_echoed(self):
        """命中凭据时只报告规则，不能把敏感值复制到日志。"""
        secret = "sk-" + "z" * 24
        sample = (secret + "\n" + "/Users" + "/private-person/file").encode()
        findings = check_file("backend/example.py", sample)
        self.assertEqual({item["rule"] for item in findings}, {"provider_key", "personal_home"})
        self.assertNotIn(secret, repr(findings))

    def test_binary_large_and_symlink_are_rejected(self):
        """文件名合法仍需检查内容和Git模式。"""
        self.assertTrue(check_file("docs/x.md", b"\x00"))
        self.assertTrue(check_file("docs/x.md", b"x" * 1_000_001))
        self.assertTrue(check_file("docs/x.md", b"target", "120000"))

    def test_only_reviewed_screenshot_bytes_are_allowed(self):
        """两份已审查图片能发布；同名图片的任何字节变化都拒绝。"""
        root = Path(__file__).resolve().parents[3]
        for name in REVIEWED_SCREENSHOTS:
            with self.subTest(name=name):
                data = (root / name).read_bytes()
                self.assertEqual(check_file(name, data), [])
                self.assertTrue(check_file(name, data + b'changed'))
                self.assertTrue(check_file(name, data, '120000'))
                self.assertTrue(check_file(name, data, '100755'))

    def test_unreviewed_image_path_is_still_rejected(self):
        """截图目录不是整目录豁免；新图片及路径变形仍拒绝。"""
        for name in ('frontend/assets/screenshots/private.jpg',
                     'frontend/assets/screenshots/../screenshots/chat.jpg',
                     'frontend/assets/private.png'):
            with self.subTest(name=name):
                self.assertTrue(check_file(name, b'\xff\xd8\xffunreviewed'))
