/** 聊天侧面板：按真实任务身份查看论文、证据、草稿和版本差异。 */
import {api,key} from '../api.js';
import {readingButton} from './reading-list.js';
import {renderMatrices} from './evidence-matrix.js';
import {markdown,escape,download} from '../markdown.js';
const panel=()=>document.querySelector('#artifact-panel');
const body=()=>document.querySelector('#panel-body');
export function openPanel(title) {
  document.querySelector('#panel-title').textContent=title;body().replaceChildren();
  if(!panel().open)panel().showModal();
  return body();
}
/** 读取产物目录；找不到全文时仍显示候选和明确缺口。 */
export async function showArtifacts(tid,onError) {
  const el=openPanel('论文与研究笔记');el.textContent='正在读取…';
  const [data,list,health]=await Promise.all([api('/tasks/'+tid+'/artifacts'),api('/reading-list'),api('/health')]);if(!panel().open)return;
  el.innerHTML='<h2>结构化证据矩阵</h2><div id="matrix-list"></div><h2>研究笔记</h2><div id="draft-list"></div><h2>来源与证据</h2><div id="paper-list"></div>';
  renderMatrices(el.querySelector('#matrix-list'),data.evidence_matrices,eid=>showEvidence(tid,eid).catch(onError));
  const ds=el.querySelector('#draft-list');
  if(!data.drafts.length)ds.textContent='本任务没有生成独立笔记，回答保留在对话中。';
  data.drafts.forEach(d=>{const b=document.createElement('button');b.textContent=d.title+' · '+(d.saved?'已保存':'草稿');b.onclick=()=>showDraft(tid,d.draft_id).catch(onError);ds.append(b);});
  const ps=el.querySelector('#paper-list');
  if(!data.papers.length)ps.textContent='尚未取得论文记录。';
  data.papers.forEach(p=>{
    const card=document.createElement('article');card.className='paper-card';card.innerHTML='<h3>'+escape(p.title)+'</h3><p>'+escape(p.source)+' · '+escape(p.published||'日期未知')+'</p>';
    const url=p.paper_url||p.pdf_url;if(/^https?:\/\//.test(url||'')){const a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener';a.textContent='打开来源 ↗';card.append(a);}
    const add=readingButton(p,tid,list.items.find(item=>item.paper_id===p.paper_id),onError);
    if(health.mode==='readonly'||p.metadata_origin==='authorized_read'){add.disabled=true;add.title=health.mode==='readonly'?'当前入口只读':'这是附件定位记录，尚无可收藏的论文元数据';}
    card.append(add);
    const evidence=data.evidence.filter(e=>e.paper_id===p.paper_id);
    const rows=document.createElement('div');evidence.forEach(e=>{const b=document.createElement('button');b.textContent=e.page_number?'第 '+e.page_number+' 页':'正文（无物理页码）';b.onclick=()=>showEvidence(tid,e.evidence_id).catch(onError);rows.append(b);});
    if(!evidence.length)rows.textContent='尚未读取正文';card.append(rows);ps.append(card);
  });
  if(data.attachments.length){const a=document.createElement('p');a.className='status-line';a.textContent='本次附件：'+data.attachments.map(x=>x.filename||x.url||x.input_ref).join('、');el.append(a);}
}
export async function showEvidence(tid,eid) {
  const data=await api('/tasks/'+tid+'/evidence/'+encodeURIComponent(eid));const el=openPanel(data.page_number?'原文 · 第 '+data.page_number+' 页':'原文 · 无物理页码');
  const text=document.createElement('div');text.className='evidence-text';text.textContent=data.text;el.append(text);
  if(data.gaps.length){const p=document.createElement('p');p.className='warning';p.textContent=data.gaps.join('；');el.append(p);}
  if(data.pdf_available){const frame=document.createElement('iframe');frame.className='pdf-frame';frame.title='论文原始PDF';frame.src='/api/tasks/'+tid+'/evidence/'+encodeURIComponent(eid)+'/pdf'+(data.page_number?'#page='+data.page_number:'');el.append(frame);}
}
/** 笔记按hash保存；编辑必须先预览差异，再确认新版本，不静默覆盖。 */
export async function showDraft(tid,did) {
  const raw=await api('/tasks/'+tid+'/drafts/'+did);
  const data=raw.saved?await api('/notes/'+did):raw;
  const el=openPanel(data.title);const actions=document.createElement('div');actions.className='message-actions';el.append(actions);
  const status=document.createElement('p');status.className='status-line';status.textContent=raw.saved?'已保存到项目 · 文件状态：'+data.integrity:'草稿 · 尚未保存到项目';el.append(status);
  // 只显示绑定当前内容hash的权威核验结果，正文中的“已核验”不能充当成功记录。
  const verification=raw.verification?.content_hash===data.content_hash?raw.verification:{status:'not_run'};
  const verified=document.createElement('p');verified.className='status-line';verified.dataset.verification=verification.status;
  verified.textContent=({supported:'独立核验：所核范围有原文支持，不代表全文无误',issues:'独立核验：仍有待解决问题',insufficient:'独立核验：当前证据不足',execution_failed:'独立核验未完成：不能把正文中的自述视为核验通过',running:'独立核验进行中',not_run:'此内容版本尚无独立核验结果'})[verification.status]||'此内容版本尚无独立核验结果';
  if(verification.status==='execution_failed')verified.classList.add('warning');el.append(verified);
  const content=document.createElement('div');markdown(content,data.markdown);el.append(content);
  const dl=document.createElement('button');dl.textContent='下载 Markdown';dl.onclick=()=>download(data.markdown,data.title+'.md');actions.append(dl);
  const error=document.createElement('p');error.className='warning';error.hidden=true;el.append(error);
  const save=document.createElement('button');save.textContent=raw.saved?'编辑与版本':'保存此稿到项目';actions.append(save);
  save.onclick=async()=>{try {
    if(!raw.saved){save.disabled=true;await api('/tasks/'+tid+'/drafts/'+did+'/save',{content_hash:data.content_hash,confirm:true});await showDraft(tid,did);return;}
    save.disabled=true;
    const editor=document.createElement('textarea');editor.rows=16;editor.value=data.markdown;editor.setAttribute('aria-label','编辑笔记Markdown');el.append(editor);
    const preview=document.createElement('button');preview.textContent='预览差异';el.append(preview);
    const diff=document.createElement('pre');diff.className='diff';el.append(diff);
    const confirm=document.createElement('button');confirm.textContent='确认保存新版本';confirm.hidden=true;el.append(confirm);
    let payload;
    editor.oninput=()=>{confirm.hidden=true;payload=null;};
    preview.onclick=async()=>{try{payload={base_hash:data.content_hash,markdown:editor.value,idempotency_key:key(),confirm:false};const p=await api('/notes/'+did+'/versions',payload);diff.textContent=p.diff||'内容没有变化';confirm.hidden=!p.diff;}catch(e){error.hidden=false;error.textContent=e.message;}};
    confirm.onclick=async()=>{try{confirm.disabled=true;await api('/notes/'+did+'/versions',{...payload,confirm:true});await showDraft(tid,did);}catch(e){error.hidden=false;error.textContent=e.message;confirm.disabled=false;}};
    const versions=await api('/notes/'+did+'/versions');const h=document.createElement('details');h.innerHTML='<summary>历史版本（'+versions.versions.length+'）</summary>';const list=document.createElement('pre');list.className='diff';list.textContent=versions.versions.map(v=>v.created_at+' · '+v.content_hash.slice(0,12)).join('\n');h.append(list);el.append(h);
  }catch(e){error.hidden=false;error.textContent=e.message;save.disabled=false;}};
}
