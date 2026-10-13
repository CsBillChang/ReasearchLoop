/** 对话日期分组：按本地日历和最后活跃时间排序，不修改服务器原始时间。 */
export function groupConversations(items, now = new Date(), timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone) {
  const dateKey = date => {
    const parts = new Intl.DateTimeFormat('en-CA', {timeZone, year:'numeric', month:'2-digit', day:'2-digit'}).formatToParts(date);
    return ['year','month','day'].map(type=>parts.find(p=>p.type===type).value).join('-');
  };
  const today = dateKey(now);
  // 在日期键上减一天，避免夏令时切换时24小时不等于一个本地日历日。
  const previous = new Date(today+'T12:00:00Z');previous.setUTCDate(previous.getUTCDate()-1);
  const yesterday = previous.toISOString().slice(0,10);
  const stamp = item => Date.parse(item.last_activity_at || item.created_at) || 0;
  const groups = new Map();
  [...items].sort((a,b)=>stamp(b)-stamp(a)||a.conversation_id.localeCompare(b.conversation_id)).forEach(item=>{
    const timestamp = stamp(item), day = timestamp ? dateKey(new Date(timestamp)) : 'unknown';
    if(!groups.has(day)) {
      const label = day===today?'今天':day===yesterday?'昨天':day==='unknown'?'日期未知':day.replaceAll('-',' / ');
      groups.set(day,{date:day,label,items:[]});
    }
    groups.get(day).items.push(item);
  });
  return [...groups.values()];
}

/** 用文本节点渲染标题，重命名按钮与打开按钮分离，键盘和触屏均可操作。 */
export function renderConversations(el, items, currentId, readOnly = false) {
  el.replaceChildren();
  if(!items.length){const p=document.createElement('p');p.className='history-empty';p.textContent='从一个研究问题开始，对话会保存在这里。';el.append(p);return;}
  for(const group of groupConversations(items)) {
    const section=document.createElement('section');section.className='conversation-group';
    const heading=document.createElement('h3');heading.textContent=group.label;heading.className='conversation-date';section.append(heading);
    for(const item of group.items) {
      const row=document.createElement('div');row.className='conversation-row'+(currentId===item.conversation_id?' active':'');
      const open=document.createElement('button');open.className='conversation-open';open.dataset.cid=item.conversation_id;open.title=item.title;open.textContent=item.title;
      if(currentId===item.conversation_id)open.setAttribute('aria-current','page');
      const rename=document.createElement('button');rename.className='conversation-rename';rename.dataset.rename=item.conversation_id;rename.textContent='✎';rename.title='修改对话名称';rename.setAttribute('aria-label','修改对话名称：'+item.title);rename.disabled=readOnly;
      row.append(open,rename);section.append(row);
    }
    el.append(section);
  }
}
