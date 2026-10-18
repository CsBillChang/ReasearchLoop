/** 个人面板：自然语言明确偏好、独立兴趣推断和真实研究统计；不发隐式模型请求。 */
import {api,key} from '../api.js';
import {escape} from '../markdown.js';
import {renderSubscriptions} from './subscriptions.js';

/** 读取画像并渲染；readOnly模式不开放改写，旧结构化偏好仍可编辑和撤销。 */
export async function renderProfile(el,onError,onSource,onConversation,{readOnly=false,notice=''}={}) {
  el.setAttribute('aria-busy','true');
  el.innerHTML='<div class="loading-state" role="status">正在读取个人研究记录…<span class="loading-line" aria-hidden="true"></span><span class="loading-line short" aria-hidden="true"></span></div>';
  let data;
  try{data=await api('/profile');}
  catch(err){el.innerHTML='<p class="warning" role="alert">研究记录暂时无法读取。请稍后重新打开个人面板。</p>';throw err;}
  finally{el.removeAttribute('aria-busy');}
  const refresh=message=>renderProfile(el,onError,onSource,onConversation,{readOnly,notice:message});
  el.innerHTML=`<div class="profile-heading"><span class="profile-portrait" aria-hidden="true">◎</span><div><p class="eyebrow">Personal research / 个人研究档案</p><h1>研究，日有所积。</h1><p class="profile-description">整理关注的方向、阅读习惯与每一次研究留下的线索。</p></div></div>
    <div class="stats">${[[data.statistics.conversations,'研究对话','从问题出发'],[data.statistics.history_tasks,'研究任务记录','留下探索足迹'],[data.statistics.deep_read_distinct_papers,'精读论文','按不同论文统计']].map(([n,label,detail])=>`<div><span>${label}</span><strong>${n}</strong><small>${detail}</small></div>`).join('')}</div>
    <p class="form-hint">精读统计指已生成完整笔记的论文，不代表你本人已读完。</p>
    <section class="profile-section preference-compose"><div class="section-heading"><span class="section-icon" aria-hidden="true">＋</span><div><h2>记录你的研究偏好</h2><p>写下关注的领域与阅读习惯，用自己的话表达即可。</p></div></div>
      <form id="preference-create" class="preference-form"><label for="preference-text" class="sr-only">描述你的长期偏好</label><textarea id="preference-text" name="content" required maxlength="1000" rows="3" placeholder="例如：我主要关注开放词汇目标检测和多模态大模型。请默认用中文回答，比较方法时先说明差异，再列实验结果。"></textarea><div class="preference-footer"><span class="form-hint">原样记住，可随时修改或撤销；当前对话要求优先。</span><button class="primary" ${readOnly?'disabled':''}>记住这条偏好</button></div></form>
      <p id="preference-status" class="success-message" role="status" ${notice?'':'hidden'}>${escape(notice)}</p></section>
    <div class="profile-columns"><section class="profile-section"><div class="section-heading"><div><h2>明确偏好 <span class="count-badge">${data.explicit_preferences.length}</span></h2><p>来自你的表达，可随时修改与撤销</p></div></div><div id="explicit"></div></section>
    <section class="profile-section"><div class="section-heading"><div><h2>兴趣线索 <span class="count-badge">${data.inferred_interests.length}</span></h2><p>从研究记录中推断，保留与否由你决定</p></div></div><div id="inferred"></div></section></div>`;
  for(const [items,id] of [[data.explicit_preferences,'explicit'],[data.inferred_interests,'inferred']]) {
    const group=el.querySelector('#'+id);
    if(!items.length)group.innerHTML='<div class="profile-empty"><span aria-hidden="true">'+(id==='explicit'?'✧':'◌')+'</span><p>'+(id==='explicit'?'还没有长期偏好':'还没有兴趣推断')+'</p><small>'+(id==='explicit'?'在上方写下你的研究习惯。':'随着研究积累，相关线索会出现在这里。')+'</small></div>';
    for(const entry of items) {
      const card=document.createElement('article');card.className='memory-card';
      const natural=entry.slot.startsWith('statement:');
      card.innerHTML='<span class="memory-kind">'+(id==='explicit'?'明确偏好':'兴趣推断')+'</span>'+(natural?'':'<strong>'+escape(entry.slot)+'</strong>')+'<p>'+escape(entry.content)+'</p>';
      const actions=document.createElement('div');actions.className='memory-actions';card.append(actions);
      if(entry.source.task_id){const b=document.createElement('button');b.textContent='查看来源';b.onclick=()=>onSource(entry.source.task_id);actions.append(b);}
      if(id==='explicit') {
        const edit=document.createElement('button');edit.textContent='修改';edit.disabled=readOnly;actions.append(edit);
        edit.onclick=()=>{
          const form=document.createElement('form');form.className='preference-form preference-edit';
          form.innerHTML='<label>修改这条偏好<textarea required maxlength="1000" rows="3" aria-label="修改偏好"></textarea></label><div class="dialog-actions"><button type="button">取消</button><button class="primary">保存修改</button></div>';
          form.querySelector('textarea').value=entry.content;card.append(form);edit.disabled=true;form.querySelector('textarea').focus();
          form.querySelector('[type=button]').onclick=()=>{form.remove();edit.disabled=false;};let request=null;
          form.onsubmit=async event=>{
            event.preventDefault();const content=form.querySelector('textarea').value.trim();if(!content)return;
            if(!request||request.content!==content)request={slot:entry.slot,content,expected_version:entry.version,idempotency_key:key()};
            const submit=form.querySelector('.primary');submit.disabled=true;
            try{await api('/profile/preferences',request);await refresh('偏好已更新，将用于后续对话。');}catch(err){onError(err);}finally{submit.disabled=false;}
          };
        };
      }
      const forget=document.createElement('button');forget.textContent='撤销';forget.disabled=readOnly;
      forget.onclick=async()=>{forget.disabled=true;try{await api('/profile/forget',{memory_id:entry.memory_id,expected_version:entry.version});await refresh('已撤销，后续对话不再使用这条记录。');}catch(err){onError(err);forget.disabled=false;}};
      actions.append(forget);group.append(card);
    }
  }
  let request=null;
  el.querySelector('#preference-create').onsubmit=async event=>{
    event.preventDefault();const form=event.target,content=form.elements.content.value.trim();if(!content)return;
    if(!request||request.content!==content)request={content,idempotency_key:key()};
    const button=form.querySelector('button');button.disabled=true;
    try{const saved=await api('/profile/preferences/statement',request);await refresh(saved.status==='active'?'已经记住，你可以在下方查看和修改。':'这条提交已被撤销，请刷新后重新添加。');}
    catch(err){onError(err);}finally{button.disabled=readOnly;}
  };
  el.querySelector('#preference-text').disabled=readOnly;
  const subscriptions=document.createElement('section');subscriptions.className='profile-section subscription-section';el.append(subscriptions);
  await renderSubscriptions(subscriptions,onError,onConversation);
  if(readOnly)subscriptions.querySelectorAll('button,input,textarea,select').forEach(control=>control.disabled=true);
}
