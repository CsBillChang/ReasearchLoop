/** 会话消息和公开任务状态组件；普通用户文本不经HTML解释。 */
import {markdown,escape} from '../markdown.js';
export const terminal=status=>['completed','partial','failed','cancelled'].includes(status);
export const statusLabel=status=>({queued:'已排队，等待 Worker',running:'正在研究',needs_input:'等待你的回答',completed:'任务完成',partial:'部分完成 · 已保留成果',failed:'未完成 · 查看原因',cancelled:'已停止 · 已保留成果'}[status]||status);
const tools={retrieve_evidence:'检索原文证据',search_papers:'检索论文',read_paper:'读取原文',delegate_research:'委派研究任务',review_artifact:'核验内容',reflect:'核验内容',create_note:'生成笔记',save_note:'保存笔记',set_research_plan:'整理研究目标',finish_task:'整理交付',read_memory:'补读历史',search_memory:'查找相关历史',remember_preference:'记录偏好'};
export function progressLabel(event) {
  return event.type==='observation' ? (tools[event.data.tool]||'执行研究步骤')+(event.data.ok===false?'：未完成':'') : ({task_created:'任务已受理',worker_started:'开始研究',cancel_requested:'正在停止，等待当前请求结算',task_finished:'任务已结束',task_resumed:'收到澄清，继续执行'}[event.type]||'更新任务记录');
}
/** 生成历史消息节点；结果按钮携带真实task_id，切换不会沿用另一篇论文身份。 */
export function messageNode(message) {
  const node=document.createElement('article');node.className='message '+message.role;
  if(message.message_key?.startsWith('brief:'))node.dataset.brief=message.message_key.slice(6);
  if(message.role==='user') {node.innerHTML='<div class="bubble"></div>';const parts=message.content.split('\n\n反馈对象：当前对话的任务 ');
    node.firstChild.textContent=parts[0];
    if(parts.length>1){const hint=document.createElement('small');hint.textContent='针对所选结果的反馈';node.append(hint);}return node;}
  node.innerHTML='<div class="assistant-label">ResearchLoop</div><div class="message-body"></div>';
  markdown(node.querySelector('.message-body'),message.content);
  if(message.message_key?.startsWith('result:')) {
    node.innerHTML+='<div class="message-actions"><button data-result="'+escape(message.task_id)+'">论文与笔记</button><button data-feedback="'+escape(message.task_id)+'">反馈 / 继续研究</button><button data-download="'+escape(message.task_id)+'">下载回答</button><button data-context="'+escape(message.task_id)+'">记忆与执行详情</button></div>';
    const state=document.createElement('div');state.className='status-line';state.dataset.taskStatus=message.task_id;node.append(state);
  }
  return node;
}
/** 创建运行区，进度与回答预览独立，终态不会误标成自动审校通过。 */
export function liveNode(view) {
  const label=statusLabel(view.status);
  return `<div class="status-line ${view.status==='running'?'busy':''}" id="live-status">${escape(label)}</div>
    <details><summary>查看研究进度</summary><ol class="progress-list" id="progress-list"></ol></details>
    <div class="stream-preview"><small id="preview-label" hidden>正在生成 · 以任务结束后的结果为准</small><div id="stream-body"></div></div>
    ${view.status==='needs_input'?'<div class="clarification">'+escape(view.clarification?.question||'请补充要求')+'<p>在下方输入回答，继续原任务。</p>'+ (view.clarification?.suggested_values||[]).map(s=>'<button data-answer="'+escape(s)+'">'+escape(s)+'</button>').join('')+'</div>':''}`;
}

/** 简报渲染后才记录曝光；用户已读必须通过按钮单独提交，不自动启动任何模型。 */
export async function hydrateBriefs(container,onError,onAsk) {
  const {api}=await import('../api.js');
  for(const node of container.querySelectorAll('[data-brief]:not([data-hydrated])')) {
    node.dataset.hydrated='true';
    try {
      const data=await api('/briefs/'+node.dataset.brief);
      if(!node.isConnected)continue;
      const actions=document.createElement('div');actions.className='brief-actions';
      for(const [i,item] of data.items.entries()) {
        const row=document.createElement('div');row.className='brief-item-actions';
        const label=document.createElement('span');label.textContent=(i+1)+'. '+item.paper.title;row.append(label);
        const read=document.createElement('button');let checked=data.read_identities.includes(item.identity);const update=()=>read.textContent=checked?'撤销已读标记':'标为已读';update();
        read.onclick=async()=>{read.disabled=true;try{await api('/brief-read-marks',{identity:item.identity,read:!checked});checked=!checked;update();}catch(e){onError(e);}finally{read.disabled=false;}};row.append(read);
        const ask=document.createElement('button');ask.textContent='就这篇继续提问';ask.onclick=()=>onAsk(item.paper);row.append(ask);actions.append(row);
      }
      node.append(actions);
      // IntersectionObserver只在消息实际进入视口时上报；GET读回/后台生成不算展示。
      const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)&&!document.hidden){observer.disconnect();api('/briefs/'+node.dataset.brief+'/displayed',{}).catch(onError);}});
      observer.observe(node);
    }catch(e){onError(e);}
  }
}
