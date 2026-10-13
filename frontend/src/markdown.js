/** Markdown/数学统一渲染：保留公式定界符，经净化后插入DOM，不允许模型HTML执行。 */
import {marked} from '/vendor/marked.js';
import purify from '/vendor/purify.js';
import katex from '/vendor/katex.js';

// 原始HTML只显示文字；允许的style全部来自KaTeX可信输出。
marked.use({renderer:{html({text}){return escape(text);}}});

for (const [name,open,close,display] of [
  ['blockDollar','$$','$$',true],['blockBracket','\\[','\\]',true],
  ['inlineParen','\\(','\\)',false],['inlineDollar','$','$',false]
]) {
  marked.use({extensions:[{name,level:'inline',start:src=>src.indexOf(open),
    tokenizer(src) {
      if(!src.startsWith(open) || (open==='$' && src.startsWith('$$'))) return;
      const end=src.indexOf(close,open.length);
      if(end<open.length || (!display && src.slice(open.length,end).includes('\n'))) return;
      return {type:name,raw:src.slice(0,end+close.length),text:src.slice(open.length,end),display};
    },renderer(token){
      return katex.renderToString(token.text,{displayMode:token.display,throwOnError:false,trust:false,strict:false});
    }}]});
}
/** 将不可信Markdown渲染为安全DOM；禁外链图片以避免自动远端请求。 */
export function markdown(element,text) {
  element.innerHTML=purify.sanitize(marked.parse(text||'',{gfm:true}),{
    FORBID_TAGS:['img','iframe','style','form','input','button']});
  // KaTeX依赖布局style，单独允许由可信渲染器生成的style，通过sanitize默认属性集保留。
  element.querySelectorAll('a').forEach(a=>{a.target='_blank';a.rel='noopener noreferrer';});
  element.classList.add('markdown');
}
/** 所有普通标签/标题先转义；只将markdown函数用于正文。 */
export const escape=text=>String(text??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function download(text,name='研究笔记.md') {
  const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
