/** 订阅配置组件：显式开关、日历时间与检索范围；保存表单不会立即发起研究。 */
import {api,key} from '../api.js';
import {escape} from '../markdown.js';
export async function renderSubscriptions(el,onError,onConversation) {
  const data=await api('/subscriptions');
  el.innerHTML='<h2>个性化论文简报</h2><p class="status-line">启用后每天按本地调度时间查询摘要，最多5篇，不自动全文精读。电脑休眠后只补一份合并简报；需要运行 make scheduler。</p><div class="subscription-list"></div><button type="button" class="add-subscription">新建订阅</button><div class="subscription-editor"></div>';
  const list=el.querySelector('.subscription-list');
  data.subscriptions.forEach(sub=>{
    const card=document.createElement('section');card.className='memory-card';
    card.innerHTML='<h3>'+escape(sub.config.title)+'</h3><p>'+escape(sub.config.directions.join('、'))+'</p><p>'+(sub.enabled?'已启用':'已关闭')+' · '+escape(sub.config.timezone)+' '+escape(sub.config.at)+' · 最多'+sub.config.limit+'篇</p><small>下次调度：'+escape(new Date(sub.next_due).toLocaleString('zh-CN',{timeZone:sub.config.timezone}))+'（'+escape(sub.config.timezone)+'）</small><p>发现 '+sub.statistics.found+' · 已交付 '+sub.statistics.delivered+' · 页面已展示 '+sub.statistics.displayed+' · 本人标为已读 '+sub.statistics.user_read+'</p>';
    const edit=document.createElement('button');edit.textContent='编辑订阅';edit.onclick=()=>editor(sub);card.append(edit);
    const toggle=document.createElement('button');toggle.textContent=sub.enabled?'关闭订阅':'启用订阅';toggle.onclick=async()=>{toggle.disabled=true;try{await api('/subscriptions/'+sub.subscription_id,{...sub.config,enabled:!sub.enabled,expected_revision:sub.revision,idempotency_key:key()});await renderSubscriptions(el,onError,onConversation);}catch(e){toggle.disabled=false;onError(e);}};card.append(toggle);
    const open=document.createElement('button');open.textContent='查看简报对话';open.onclick=()=>onConversation(sub.conversation_id);card.append(open);
    if(sub.latest){const status=document.createElement('p');status.className='status-line';status.textContent='最近一轮：'+({completed:'已交付',partial:'已交付，部分来源存在缺口',running:'正在检索',cancelled:'已停止'}[sub.latest.status]||sub.latest.status);card.append(status);}list.append(card);
  });
  /** 以当前revision保存，来源来自服务端目录而不是硬编码少数会议。 */
  function editor(sub) {
    const c=sub?.config||{title:'每日科研简报',directions:[],sources:data.default_sources,timezone:'Asia/Shanghai',at:'09:00',limit:5,lookback_days:7,use_profile:true};
    const box=el.querySelector('.subscription-editor');
    box.innerHTML='<form class="subscription-form"><label>简报名称<input name="title" required maxlength="80" value="'+escape(c.title)+'"></label><label>检索方向（每行一个，最多3个，建议用论文常见关键词）<textarea name="directions" required rows="3">'+escape(c.directions.join('\n'))+'</textarea></label><div class="subscription-grid"><label>时间<input type="time" name="at" value="'+escape(c.at)+'" required></label><label>时区<input name="timezone" value="'+escape(c.timezone)+'" required></label><label>最多候选数<input type="number" name="limit" min="1" max="5" value="'+c.limit+'" required></label><label>首次回溯天数<input type="number" name="lookback_days" min="1" max="365" value="'+c.lookback_days+'" required></label></div><details><summary>论文来源（17源可选）</summary><div class="source-options">'+data.sources.map(s=>'<label><input type="checkbox" name="sources" value="'+escape(s)+'" '+(c.sources.includes(s)?'checked':'')+'>'+escape(s)+'</label>').join('')+'</div></details><label class="check"><input type="checkbox" name="use_profile" '+(c.use_profile?'checked':'')+'>使用当前有效偏好与可撤销推断辅助排序</label><label class="check"><input type="checkbox" name="enabled" '+(sub?.enabled?'checked':'')+'>启用每日简报</label><p class="status-line">检索截至上一个本地日历日，后续重叠一天并去重。来源失败与数量不足会直接写在简报中。</p><button class="primary" type="submit">保存订阅</button><button type="button" class="close-editor">取消</button></form>';
    box.querySelector('.close-editor').onclick=()=>box.replaceChildren();
    const form=box.querySelector('form');let operationKey=key(),previous=null;
    form.onsubmit=async e=>{e.preventDefault();const d=new FormData(form);const payload={title:d.get('title'),directions:d.get('directions').split('\n').map(x=>x.trim()).filter(Boolean),sources:d.getAll('sources'),timezone:d.get('timezone'),at:d.get('at'),limit:Number(d.get('limit')),lookback_days:Number(d.get('lookback_days')),use_profile:d.has('use_profile'),enabled:d.has('enabled'),expected_revision:sub?.revision||0};
      if(previous&&previous!==JSON.stringify(payload))operationKey=key();previous=JSON.stringify(payload);const button=form.querySelector('[type=submit]');button.disabled=true;
      try{await api('/subscriptions'+(sub?'/'+sub.subscription_id:''),{...payload,idempotency_key:operationKey});await renderSubscriptions(el,onError,onConversation);}catch(err){button.disabled=false;onError(err);}};
    box.scrollIntoView({block:'nearest'});
  }
  el.querySelector('.add-subscription').onclick=()=>editor(null);
}
