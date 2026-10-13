/** AG5应用协调：会话身份、幂等提交、只读流恢复与侧面板；研究决策仍由后端Agent负责。 */
import {api,upload,watch,key} from './api.js';
import {markdown,escape,download} from './markdown.js';
import {messageNode,liveNode,terminal,statusLabel,progressLabel,hydrateBriefs} from './components/chat.js';
import {showArtifacts,openPanel} from './components/artifacts.js';
import {showReadingList} from './components/reading-list.js';
import {renderProfile} from './components/profile.js';
import {renderConversations} from './components/conversations.js';
const $=s=>document.querySelector(s);
let cid=null,active=null,stream=null,generation=0,attachments=[],feedback=null,busy=false,offset=0;
let pending=null,pollTimer=null,readOnly=false;
// 接续由服务端持久化，旧浏览器暂存不能自行提交新模型任务。
localStorage.removeItem('rl-pending');
const inputs=new Map();let submitRecord=null;
const taskCache=new Map();
let conversationItems=[],renameTarget=null,renameRequest=null;
function error(e){$('#toast').textContent=e.message||String(e);$('#toast').hidden=false;setTimeout(()=>$('#toast').hidden=true,7000);}
function closeMenu(){$('#sidebar').classList.remove('open');$('#backdrop').hidden=true;}
function cacheInput(){inputs.set(cid||'new',{text:$('#prompt').value,attachments,feedback,allow:$('#save-permission').checked});}
function restoreInput(){const d=inputs.get(cid||'new')||{};$('#prompt').value=d.text||'';attachments=d.attachments||[];feedback=d.feedback||null;$('#save-permission').checked=!!d.allow;renderAttachments();renderFeedback();}
function renderAttachments(){$('#attachments').innerHTML=attachments.map((a,i)=>'<span class="chip">▤ '+escape(a.filename||a.url||a.input_ref)+' <button type="button" data-remove="'+i+'" aria-label="移除附件">×</button></span>').join('');}
function renderFeedback(){$('#feedback-target').hidden=!feedback;$('#feedback-target').innerHTML=feedback?'针对已选结果提交反馈 · '+escape(feedback.source_task_id.slice(0,8))+' <button id="clear-feedback">取消关联</button>':'';}
function controls(){const running=active&&!terminal(active.status)&&active.status!=='needs_input';$('#send').textContent=running?'停止后发送':active?.status==='needs_input'?'回答并继续':'发送 ↑';$('#send').disabled=busy||readOnly;$('#prompt').disabled=readOnly;$('#cancel').hidden=!running;$('#composer-help').textContent=running?'追加要求将停止当前任务，保留已有成果后继续。':active?.status==='needs_input'?'这次回答将恢复原任务，保留最初要求。':'Enter 发送 · Shift+Enter 换行 · 结果请结合原文使用';if(readOnly){$('#composer-help').textContent='只读演示 · 可浏览已有成果，研究任务请使用日常入口';$('#cancel').hidden=true;}}
async function list(){const data=await api('/conversations');conversationItems=data.conversations;renderConversations($('#conversations'),conversationItems,$('#profile-view').hidden?cid:null,readOnly);}
function showChat(){$('#profile-view').hidden=true;$('#chat-view').hidden=false;$('#profile-nav').classList.remove('active');$('#profile-nav').removeAttribute('aria-current');$('#view-title').textContent='科研对话';closeMenu();}
async function openConversation(id,{fresh=false,keepInput=false}={}) {
  if(!keepInput)cacheInput();clearTimeout(pollTimer);generation++;const token=generation;stream?.close();stream=null;active=null;cid=id;
  if(id)localStorage.setItem('rl-conversation',id);else localStorage.removeItem('rl-conversation');
  showChat();if(!keepInput)restoreInput();$('#live').hidden=true;$('#messages').replaceChildren();$('#older').hidden=true;$('#welcome').hidden=!!id;controls();
  await list();if(!id)return;
  let view=await api('/conversations/'+id+'?limit=1');if(token!==generation)return;
  offset=Math.max(0,view.total-(fresh?100:50));view=await api('/conversations/'+id+'?offset='+offset+'&limit=100');if(token!==generation)return;
  $('#older').hidden=offset===0;view.messages.forEach(m=>$('#messages').append(messageNode(m)));
  await hydrateBriefs($('#messages'),error,askAboutPaper);
  $('#view-title').textContent=view.title;$('#welcome').hidden=view.messages.length>0;
  const tids=[...new Set(view.messages.filter(m=>m.message_key?.startsWith('result:')).map(m=>m.task_id))];
  await Promise.all(tids.map(async tid=>{const task=await api('/tasks/'+tid);taskCache.set(tid,task);if(token!==generation)return;const node=document.querySelector('[data-task-status="'+tid+'"]');if(node){
      node.textContent=statusLabel(task.status);
      if(task.result?.gaps?.length){const warning=document.createElement('details');warning.innerHTML='<summary>未解决的问题与阅读边界</summary>';const text=document.createElement('p');text.className='warning';text.textContent=task.result.gaps.join('；');warning.append(text);node.append(warning);}
    } }));
  if(token!==generation)return;
  if(view.active_tasks.length){active=await api('/tasks/'+view.active_tasks[0].task_id);if(token!==generation)return;startWatch(active,token);}
  else if(view.pending_continuations?.length){
    $('#live').hidden=false;$('#live').textContent='新要求已保存，等待旧任务结算后继续。';
    pollTimer=setTimeout(()=>{if(token===generation)openConversation(id,{keepInput:true}).catch(error);},1000);
  }
  controls();$('#conversation-scroll').scrollTop=$('#conversation-scroll').scrollHeight;
}
function startWatch(task,token=generation) {
  active=task;$('#live').hidden=false;$('#live').innerHTML=liveNode(task);controls();
  if(task.status==='needs_input')return;
  let events=new Set(),latestRequest=0,latestKey='';
  stream=watch(task.task_id,event=>{
    if(token!==generation||events.has(event.id))return;events.add(event.id);
    if(event.type==='answer_stream'){
      const d=event.data;if(d.request_number<latestRequest)return;
      latestRequest=d.request_number;latestKey=d.stream_key;
      $('#preview-label').hidden=false;markdown($('#stream-body'),d.text);
    }else{
      if(event.type==='worker_started'){active.status='running';$('#live-status')?.classList.add('busy');controls();}
      const li=document.createElement('li');li.textContent=progressLabel(event);$('#progress-list')?.append(li);if($('#live-status'))$('#live-status').textContent=event.type==='cancel_requested'?'正在停止，等待当前请求结算…':statusLabel(active.status)+' · '+li.textContent;}
    const scroll=$('#conversation-scroll');if(scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<250)scroll.scrollTop=scroll.scrollHeight;
  },async result=>{
    if(token!==generation)return;taskCache.set(task.task_id,result);active=result;stream=null;
    try{await openConversation(cid,{keepInput:true});}catch(e){error(e);}
  },()=>{if(token===generation&&$('#live-status'))$('#live-status').textContent='连接暂时中断，正在恢复进度；任务不会重复提交。';});
}
async function task(id){if(!taskCache.has(id))taskCache.set(id,await api('/tasks/'+id));return taskCache.get(id);}
async function ensureConversation(){if(cid)return;const made=await api('/conversations',{title:$('#prompt').value.trim().slice(0,36)||'新对话',idempotency_key:key()});cid=made.conversation_id;localStorage.setItem('rl-conversation',cid);}
/** 同一输入网络失败重试沿用键；修改输入是新请求，后端忙状态仍会拒绝并发。 */
async function submit(payload,feedbackMode=false){
  const fingerprint=JSON.stringify(payload);
  if(submitRecord?.fingerprint!==fingerprint)submitRecord={fingerprint,key:key()};
  const view=await api(feedbackMode?'/feedback':'/tasks',{...payload,idempotency_key:submitRecord.key});submitRecord=null;return view;
}
$('#composer').onsubmit=async e=>{
  e.preventDefault();const text=$('#prompt').value.trim();if(!text||busy)return;busy=true;controls();
  try{
    await ensureConversation();
    if(active?.status==='needs_input'){
      await api('/tasks/'+active.task_id+'/resume',{clarification_id:active.clarification.clarification_id,answer:text,idempotency_key:(submitRecord||={key:key()}).key});submitRecord=null;
    }else if(active&&!terminal(active.status)){
      if(feedback)throw new Error('请先停止当前任务，再提交针对历史结果的反馈。');
      const assets=await api('/tasks/'+active.task_id+'/artifacts');
      const original=assets.attachments.filter(a=>a.input_ref.startsWith('upload-')||a.url);
      const nextInput={conversation_id:cid,prompt:text,attachments:[...new Map([...original,...attachments].map(a=>[a.input_ref,a])).values()],save_policy:$('#save-permission').checked?'allow':'deny',idempotency_key:key()};
      const fingerprint=JSON.stringify({...nextInput,idempotency_key:null});
      if(pending?.fingerprint!==fingerprint)pending={fingerprint,payload:nextInput};
      await api('/tasks/'+active.task_id+'/continue',pending.payload);
      pending=null;localStorage.removeItem('rl-pending');$('#prompt').value='';attachments=[];$('#save-permission').checked=false;cacheInput();renderAttachments();
      $('#composer-help').textContent='已记录后续要求。当前请求结算后继续，期间可以切换对话。';return;
    }else{
      if(feedback)await submit({conversation_id:cid,...feedback,prompt:text,save_policy:$('#save-permission').checked?'allow':'deny'},true);
      else await submit({conversation_id:cid,prompt:text,attachments,save_policy:$('#save-permission').checked?'allow':'deny'});
    }
    $('#prompt').value='';attachments=[];feedback=null;$('#save-permission').checked=false;cacheInput();renderAttachments();renderFeedback();await openConversation(cid,{keepInput:true});
  }catch(err){error(err);}finally{busy=false;controls();}
};
$('#cancel').onclick=async()=>{try{pending=null;localStorage.removeItem('rl-pending');await api('/tasks/'+active.task_id+'/cancel',{});$('#live-status').textContent='正在停止，已有成果会保留。';}catch(e){error(e);}};
$('#prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('#composer').requestSubmit();}};
$('#attach').onclick=()=>$('#pdf-input').click();
$('#pdf-input').onchange=async e=>{busy=true;controls();try{for(const f of e.target.files){const a=await upload(f);if(!attachments.some(x=>x.input_ref===a.input_ref))attachments.push({input_ref:a.input_ref,filename:a.filename});}renderAttachments();cacheInput();}catch(err){error(err);}finally{e.target.value='';busy=false;controls();}};
$('#attachments').onclick=e=>{if(e.target.dataset.remove!==undefined){attachments.splice(Number(e.target.dataset.remove),1);renderAttachments();cacheInput();}};
$('#add-link').onclick=()=>$('#link-dialog').showModal();$('#link-close').onclick=()=>$('#link-dialog').close();
$('#link-form').onsubmit=e=>{e.preventDefault();const url=$('#paper-url').value.trim();if(!/^https?:\/\//.test(url))return error(new Error('请使用 http 或 https 论文地址'));attachments.push({input_ref:'link-'+key(),url,filename:new URL(url).hostname});renderAttachments();cacheInput();$('#paper-url').value='';$('#link-dialog').close();};
$('#reading-list').onclick=()=>showReadingList(error,readOnly).catch(error);
$('#panel-close').onclick=()=>$('#artifact-panel').close();$('#menu').onclick=()=>{$('#sidebar').classList.toggle('open');$('#backdrop').hidden=!$('#sidebar').classList.contains('open');};$('#backdrop').onclick=closeMenu;
$('#new-chat').onclick=()=>openConversation(null).catch(error);
$('#conversations').onclick=e=>{
  const rename=e.target.closest('[data-rename]');
  if(rename&&!readOnly){renameTarget=conversationItems.find(c=>c.conversation_id===rename.dataset.rename);if(!renameTarget)return;renameRequest=null;$('#conversation-title').value=renameTarget.title;$('#rename-error').hidden=true;$('#rename-dialog').showModal();$('#conversation-title').focus();$('#conversation-title').select();return;}
  const button=e.target.closest('[data-cid]');if(button)openConversation(button.dataset.cid).catch(error);
};
$('#rename-cancel').onclick=()=>$('#rename-dialog').close();
/** 同一次保存失败后复用请求身份；其他窗口先改名则显示冲突，不覆盖新标题。 */
$('#rename-form').onsubmit=async e=>{
  e.preventDefault();const title=$('#conversation-title').value.trim();if(!title)return;
  if(!renameRequest||renameRequest.title!==title)renameRequest={title,expected_title:renameTarget.title,idempotency_key:key()};
  $('#rename-submit').disabled=true;$('#rename-error').hidden=true;
  try{const row=await api('/conversations/'+renameTarget.conversation_id+'/rename',renameRequest);await list();if(cid===row.conversation_id&&$('#profile-view').hidden)$('#view-title').textContent=row.title;$('#rename-dialog').close();}
  catch(err){$('#rename-error').textContent=err.message;$('#rename-error').hidden=false;}
  finally{$('#rename-submit').disabled=false;}
};
$('.brand').onclick=e=>{e.preventDefault();openConversation(null).catch(error);};
$('#welcome').onclick=e=>{const b=e.target.closest('[data-prompt]');if(b){$('#prompt').value=b.dataset.prompt;$('#prompt').focus();}};
$('#feedback-target').onclick=e=>{if(e.target.id==='clear-feedback'){feedback=null;renderFeedback();}};
$('#live').onclick=e=>{if(e.target.dataset.answer){$('#prompt').value=e.target.dataset.answer;$('#composer').requestSubmit();}};
$('#older').onclick=async()=>{try{const next=Math.max(0,offset-50);const data=await api('/conversations/'+cid+'?offset='+next+'&limit='+(offset-next));offset=next;const nodes=data.messages.map(messageNode);$('#messages').prepend(...nodes);await hydrateBriefs($('#messages'),error,askAboutPaper);$('#older').hidden=!offset;}catch(e){error(e);}};
$('#messages').onclick=async e=>{try{
  const b=e.target.closest('button');if(!b)return;
  if(b.dataset.result)await showArtifacts(b.dataset.result,error);
  if(b.dataset.download){const t=await task(b.dataset.download);download(t.result?.answer_markdown||t.error||'任务未完成','研究回答.md');}
  if(b.dataset.feedback){const t=await task(b.dataset.feedback);const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(t.result?.answer_markdown||''));feedback={source_task_id:t.task_id,content_hash:[...new Uint8Array(bytes)].map(x=>x.toString(16).padStart(2,'0')).join('')};renderFeedback();$('#prompt').focus();}
  if(b.dataset.context){const t=await task(b.dataset.context);const el=openPanel('执行与记忆来源');el.innerHTML='<p>'+escape(statusLabel(t.status))+'</p><p>模型请求：'+t.usage.requests+' · 未结算：'+t.usage.unknown+'</p>';const budget=t.usage.budget?.tokens;
      if(budget){const line=document.createElement('p');line.className='status-line';line.textContent='本任务 Token：已结算 '+budget.used_tokens.toLocaleString()+' / 上限 '+budget.token_limit.toLocaleString()+' · 预留 '+budget.reserved_tokens.toLocaleString()+' · 用量待确认 '+budget.unknown_reservations+'；所有子 Agent 共享。';el.append(line);}
      const data=await api('/tasks/'+t.task_id+'/context');const prefs=[...new Map(data.memory_usage.flatMap(m=>m.preferences||[]).map(p=>[p.memory_id,p])).values()];
      const section=document.createElement('div');section.innerHTML='<h3>本次用到的记忆</h3>';
      const list=document.createElement('ul');prefs.forEach(p=>{const li=document.createElement('li');li.textContent=(p.kind==='inferred'?'推断兴趣：':'明确偏好：')+p.content;list.append(li);});
      if(!prefs.length){const p=document.createElement('p');p.textContent='没有使用长期偏好';section.append(p);}section.append(list);
      const note=document.createElement('p');note.className='status-line';note.textContent=data.compression.snapshot_id?'已进行上下文压缩，原记录仍可追溯。':'未触发上下文压缩。';section.append(note);
      const diagnostic=document.createElement('details');diagnostic.innerHTML='<summary>来源身份与执行记录</summary>';const pre=document.createElement('pre');pre.className='diff';pre.textContent=JSON.stringify(data,null,2);diagnostic.append(pre);section.append(diagnostic);el.append(section);if(t.result?.gaps?.length){const p=document.createElement('p');p.className='warning';p.textContent=t.result.gaps.join('；');el.append(p);}}
}catch(err){error(err);}};
$('#profile-nav').onclick=async()=>{cacheInput();clearTimeout(pollTimer);generation++;stream?.close();stream=null;closeMenu();$('#chat-view').hidden=true;$('#profile-view').hidden=false;$('#profile-nav').classList.add('active');$('#profile-nav').setAttribute('aria-current','page');renderConversations($('#conversations'),conversationItems,null,readOnly);$('#view-title').textContent='个人面板';try{await renderProfile($('#profile-view'),error,async tid=>{try{const t=await api('/tasks/'+tid);const el=openPanel('偏好来源');const p=document.createElement('p');p.textContent=t.prompt;el.append(p);}catch(e){error(e);}},id=>openConversation(id).catch(error),{readOnly});}catch(e){error(e);}};
/** 将明确候选链接带入当前输入，只有用户发送后才运行研究。 */
function askAboutPaper(paper){const url=paper.paper_url||paper.pdf_url;if(url&&/^https?:\/\//.test(url)){attachments=[{input_ref:'link-'+key(),url,filename:paper.title}];renderAttachments();}$('#prompt').value='关于《'+paper.title+'》，请根据可获取的原文回答：';$('#prompt').focus();cacheInput();}
async function boot(){try{const health=await api('/health');readOnly=health.mode==='readonly';$('#connection').textContent=readOnly?'● 只读演示':'● 本地服务已连接';await openConversation(localStorage.getItem('rl-conversation'));}catch(e){$('#connection').textContent='服务暂不可用';error(e);await list().catch(()=>{});}}
boot();
