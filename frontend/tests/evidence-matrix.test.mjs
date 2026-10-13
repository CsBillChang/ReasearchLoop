/** 矩阵纯渲染安全与语义标签；真实点击由独立浏览器验收负责。 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {matrixMarkup} from '../src/components/evidence-matrix.js';
const fixture=()=>({matrix_hash:'a'.repeat(64),created_at:'2026-10-12',question:'比较 <script>alert(1)</script>',integrity:'intact',rows:[{title:'论文甲',content_sha256:'b'.repeat(64),document_id:'doc',parser_name:'fixture',parser_version:'1',scope:[{page_number:1,gaps:[]}],cells:[{dimension:'metric',status:'uncertain',value:'42',conditions:['仅骨干网络'],reason:'比较条件不同',references:[{evidence_id:'e" onclick="bad',page_number:null,quote:'AP < 43'}]}]}]});
test('矩阵转义文本与属性，显示条件及无物理页码',()=>{const html=matrixMarkup([fixture()]);assert.ok(!html.includes('<script>'));assert.ok(!html.includes(' onclick="bad'));assert.ok(html.includes('仅骨干网络'));assert.ok(html.includes('无物理页码'));assert.ok(html.includes('未进行矩阵语义审校'));});
test('来源变化和历史版本保留，失效跳转禁用',()=>{const a=fixture(),b=fixture();b.integrity='stale';const html=matrixMarkup([a,b]);assert.ok(html.includes('历史矩阵'));assert.ok(html.includes('此版本仅作历史参考'));assert.ok(html.includes('disabled'));});
test('无矩阵的旧任务仍有明确说明',()=>assert.ok(matrixMarkup([]).includes('原回答与笔记仍可阅读')));
