// 只验证真实marked的URL词法与HTML输出；不调用浏览器或外部网络。
import test from 'node:test';
import assert from 'node:assert/strict';
import {Marked,Tokenizer} from 'marked';
import {withChineseUrlBoundary} from '../src/markdown-links.js';

function render(text) {
  const parser=new Marked({gfm:true,tokenizer:{url:withChineseUrlBoundary(Tokenizer.prototype.url)}});
  const html=parser.parse(text);
  return {html,hrefs:[...html.matchAll(/href="([^"]*)"/g)].map(match=>match[1])};
}

test('裸URL在中文括号与句读之前结束，尾注文字保持',()=>{
  for(const suffix of ['（abs 由编号推得）','（同上，由编号推得）','。下一句','，说明','；说明','！说明','？说明','）说明','】说明','》说明','”说明','’说明']) {
    const {html,hrefs}=render('摘要页：https://arxiv.org/abs/2608.30653v1'+suffix);
    assert.deepEqual(hrefs,['https://arxiv.org/abs/2608.30653v1'],suffix);
    assert.ok(html.includes('</a>'+suffix),html);
  }
});

test('合法ASCII路径、查询参数与配对英文括号保持',()=>{
  const {hrefs}=render('See https://example.org/a_(b)?x=1&y=2 and https://openreview.net/forum?id=Ab_C-9.');
  assert.deepEqual(hrefs,['https://example.org/a_(b)?x=1&amp;y=2','https://openreview.net/forum?id=Ab_C-9']);
  const encoded='https://example.org/a%EF%BC%88b%EF%BC%89';
  assert.deepEqual(render(encoded+'（尾注）').hrefs,[encoded]);
});

test('显式Markdown与尖括号链接保留中文路径及括号',()=>{
  for(const text of ['[资料](https://example.org/资料（第一版）?词=中文)','<https://example.org/资料（第一版）?词=中文>']) {
    const {hrefs}=render(text);
    assert.equal(hrefs.length,1);
    assert.equal(decodeURI(hrefs[0]),'https://example.org/资料（第一版）?词=中文');
  }
});

test('代码不自动链接，URL后的HTML标记不吞入href',()=>{
  assert.deepEqual(render('`https://arxiv.org/abs/2608.30653v1（说明）`').hrefs,[]);
  assert.deepEqual(render('```\nhttps://arxiv.org/abs/2608.30653v1（说明）\n```').hrefs,[]);
  const {html,hrefs}=render('https://arxiv.org/abs/2608.30653v1（<script>bad()</script>）');
  assert.deepEqual(hrefs,['https://arxiv.org/abs/2608.30653v1']);
  assert.ok(html.includes('</a>（'));
});
