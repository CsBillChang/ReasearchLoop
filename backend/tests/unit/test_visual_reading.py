"""视觉组件纯函数与真实本地渲染测试；不请求供应商。"""
import base64
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fitz

from research_loop.domain.visual_reading import VisualConfig, VisualInput, build_request
from research_loop.providers.paper_media import PaperMediaProvider
from research_loop.storage.files import FileStore
from research_loop.storage.memory import dump
from research_loop.storage.token_budget import estimate_details, estimate_request


def fixture(root: Path):
    """生成含矢量图形的两页合成PDF与匹配缓存，不调用解析器或网络。"""
    files = FileStore(root/'data', root/'notes', root)
    doc = fitz.open()
    for n in (1, 2):
        page = doc.new_page(width=600, height=800)
        page.insert_text((70, 80), f'Synthetic Table {n}: A 12.0, B 24.0')
        page.draw_rect((50, 100, 350, 300), color=(1, 0, 0))
        page.draw_line((50, 100), (350, 300), color=(0, 0, 1))
    payload = doc.tobytes(); doc.close()
    h = sha256(payload).hexdigest(); pdf = files.cache_pdf_path(h); files.write_bytes_atomic(pdf, payload)
    parser_config = {'text_route': 'cache-jina-pymupdf', 'min_text_chars': 1200}
    cfg = sha256(json.dumps(parser_config, sort_keys=True).encode()).hexdigest()
    key = sha256(f'{h}:pymupdf:1:{cfg}'.encode()).hexdigest(); did = 'doc:' + key[:32]
    pages = [{'evidence_id': 'evidence:' + sha256(f'{did}:{n}'.encode()).hexdigest()[:40],
              'document_id': did, 'page_number': n, 'text': f'Synthetic Table {n}: A 12.0, B 24.0',
              'parsed_markdown': '', 'assets': [], 'gaps': []} for n in (1, 2)]
    files.write_bytes_atomic(files.cache_parse_path(key), json.dumps({'total_pages': 2, 'pages': pages, 'gaps': []}).encode())
    item = VisualInput(target_ref='input-synthetic', document_id=did, pdf_sha256=h,
        page_text_sha256=sha256(pages[0]['text'].encode()).hexdigest(), physical_page=1,
        question='读出表中A与B的值。', group='visual', bbox=(0.05, 0.05, 0.8, 0.5))
    return files, pdf, item, pages


class VisualUnitTests(unittest.TestCase):
    """覆盖身份、几何、消息和Token计量，避免只验证实现形状。"""

    def setUp(self):
        """每个测试使用新目录，合成数据不会进入正式证据库。"""
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.files, self.pdf, self.item, self.pages = fixture(Path(temp.name))
        self.media = PaperMediaProvider(self.files)

    def prepared(self):
        """模拟已由服务校验的内容，仅供消息纯函数测试。"""
        result = self.media.render(self.pdf, self.item.pdf_sha256, 1, self.item.bbox)
        return {'input': self.item.model_dump(mode='json'), 'text': self.pages[0]['text'],
                'assets': result['assets'], 'manifest': {'document_id': self.item.document_id}}

    def test_vector_page_render_and_region_are_stable(self):
        """无内嵌图片的矢量页仍可渲染，重复输出字节与坐标稳定。"""
        with fitz.open(self.pdf) as doc:
            self.assertEqual(doc[0].get_images(), [])
        one = self.media.render(self.pdf, self.item.pdf_sha256, 1, self.item.bbox)
        two = self.media.render(self.pdf, self.item.pdf_sha256, 1, self.item.bbox)
        self.assertEqual(one['assets'], two['assets'])
        self.assertEqual(max(one['assets'][0]['width'], one['assets'][0]['height']), 1800)
        self.assertEqual(one['assets'][1]['bbox'], [30.0, 40.0, 480.0, 400.0])
        self.assertFalse(one['model_received_images'])

    def test_hash_page_and_crop_rejections(self):
        """错版本、越页、空裁区和NaN在渲染前拒绝。"""
        for page, h, bbox in [(3, self.item.pdf_sha256, None), (True, self.item.pdf_sha256, None),
                (1, '0'*64, None), (1, self.item.pdf_sha256, (0,0,0,1)),
                (1, self.item.pdf_sha256, (0,0,float('nan'),1))]:
            with self.subTest(page=page, bbox=bbox), self.assertRaises(ValueError):
                self.media.render(self.pdf, h, page, bbox)

    def test_rotated_page_keeps_original_coordinates(self):
        """旋转页同时保留显示坐标与原页坐标，不猜测来源位置。"""
        with fitz.open(self.pdf) as doc:
            doc[0].set_rotation(90); data = doc.tobytes()
        path = self.files.cache_pdf_path(sha256(data).hexdigest()); path.write_bytes(data)
        value = self.media.render(path, sha256(data).hexdigest(), 1, (0,0,.5,.5))
        self.assertEqual(value['assets'][1]['page_rotation'], 90)
        self.assertNotEqual(value['assets'][1]['bbox'], value['assets'][1]['pdf_bbox'])

    def test_ab_messages_only_differ_in_images(self):
        """相同问题与原文不被标签污染；A/B只差图片输入块。"""
        b = self.prepared(); b['gold'] = 'DO NOT LEAK LABEL'
        a = deepcopy(b); a['input']['group'] = 'text'
        ar, br = [build_request(p, VisualConfig(), self.files.cache_dir) for p in (a,b)]
        self.assertEqual(ar['messages'][0], br['messages'][0])
        self.assertEqual(ar['messages'][1]['content'], br['messages'][1]['content'][:1])
        self.assertEqual(len(br['messages'][1]['content']), 3)
        self.assertNotIn('DO NOT LEAK LABEL', dump(br))
        self.assertEqual(br['thinking'], {'type':'disabled'})

    def test_changed_text_image_and_path_are_rejected(self):
        """读回时再次查hash与目录边界，不能替换成其他本机图片。"""
        prepared = self.prepared(); bad = deepcopy(prepared); bad['text'] += 'changed'
        with self.assertRaises(ValueError): build_request(bad, VisualConfig(), self.files.cache_dir)
        bad = deepcopy(prepared); bad['assets'][0]['path'] = str(self.pdf.parent.parent.parent/'outside.png')
        with self.assertRaises(PermissionError): build_request(bad, VisualConfig(), self.files.cache_dir)
        Path(prepared['assets'][0]['path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError): build_request(prepared, VisualConfig(), self.files.cache_dir)

    def test_legacy_text_estimate_unchanged(self):
        """文字、工具和Unicode旧预估逐值相等，输入不突变。"""
        request = {'messages':[{'role':'user','content':'中文 test'}], 'tools':[{'name':'x'}], 'max_tokens':8192}
        before = deepcopy(request)
        expected = len(dump({'messages':request['messages'],'tools':request['tools']}).encode()) + 1024 + 8192
        self.assertEqual(estimate_request(request), expected); self.assertEqual(request, before)

    def test_image_estimate_excludes_base64_but_retains_request(self):
        """图片按上限预留，原始请求及hash仍保留实际发送字节。"""
        request = build_request(self.prepared(), VisualConfig(), self.files.cache_dir); old = deepcopy(request)
        detail = estimate_details(request)
        self.assertEqual(detail['image_tokens_upper_bound'], 2048)
        self.assertLess(detail['reserved_tokens'], detail['transport_bytes'] + 8192)
        self.assertEqual(request, old)

    def test_image_validation_does_not_accept_remote_fake_or_wrong_model(self):
        """不靠图片标签绕过估算：必须是可信PNG、user角色和已知模型。"""
        original = build_request(self.prepared(), VisualConfig(), self.files.cache_dir)
        for change in ('url','bytes','model','role'):
            value = deepcopy(original)
            if change=='url': value['messages'][1]['content'][1]['image_url']['url']='https://example.com/a.png'
            if change=='bytes': value['messages'][1]['content'][1]['image_url']['url']='data:image/png;base64,'+base64.b64encode(b'fake').decode()
            if change=='model': value['model']='unknown'
            if change=='role': value['messages'][1]['role']='tool'
            with self.subTest(change=change), self.assertRaises(ValueError): estimate_request(value)

    def test_runner_requires_explicit_execution_and_rejects_production_db(self):
        """默认入口不能请求模型，隔离库名检查在连接前执行。"""
        from scripts.evaluate_visual import run, settings
        with self.assertRaises(PermissionError): run(self.files.data_dir)
        with self.assertRaises(ValueError): settings(self.files.data_dir, 'researchloop')

    def test_report_counts_regressions_and_keeps_unknown_separate(self):
        """报告同时显示退化、缺答和未知用量，不以完成状态猜测语义分。"""
        from scripts.evaluate_visual import report
        root=self.files.data_dir; labels=root/'labels.json'
        labels.write_text(json.dumps({'cases':{'V1':{'fields':{'value':'12','relation':'A'}}}}))
        (root/'freeze.json').write_text(json.dumps({'labels_path':str(labels)}))
        (root/'real-results.json').write_text(json.dumps({'results':[
            {'id':'V1-text','result':{'status':'complete','model_seconds':1,'usage':{'total_tokens':100}}},
            {'id':'V1-visual','result':{'status':'partial','model_seconds':2,'usage':{}}}]}))
        scores=root/'scores.json';scores.write_text(json.dumps({
            'V1-text':{'fields':{'value':'incorrect','relation':'correct'}},
            'V1-visual':{'fields':{'value':'correct','relation':'missing'},'unsupported_claims':['synthetic claim']}}))
        with patch('scripts.evaluate_visual.verify_freeze',return_value={}):
            result=report(root,scores)
        self.assertEqual(result['counts']['not_executed'],8)
        self.assertEqual(result['paired_changes'][0]['improved'],['value'])
        self.assertEqual(result['paired_changes'][0]['regressed'],['relation'])
        self.assertEqual(result['usage_unknown_cases'],['V1-visual'])
        self.assertEqual(result['cases'][1]['unsupported_claims'],['synthetic claim'])

    def test_freeze_validation_rejects_changed_file(self):
        """冻结后任何输入变化必须在发送前被识别。"""
        from scripts.evaluate_visual import verify_freeze, write
        root=self.files.data_dir; source=root/'source.txt';source.write_text('original')
        write(root/'freeze.json',{'files':{str(source):sha256(source.read_bytes()).hexdigest()}})
        verify_freeze(root);source.write_text('changed')
        with self.assertRaises(ValueError): verify_freeze(root)
        with self.assertRaises(FileExistsError): write(root/'freeze.json',{'different':True})


if __name__ == '__main__': unittest.main()
