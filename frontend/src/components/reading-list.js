/** 聊天侧面板中的最小清单：用户点击才写入，不隐式发送研究任务。 */
import {api,key} from '../api.js';
import {openPanel} from './artifacts.js';

/** 给论文卡添加幂等的加入按钮；重复点击或网络错误重试复用同一请求键。 */
export function readingButton(paper, taskId, item, onError) {
  const button=document.createElement('button');
  const active=item&&item.status!=='removed';
  button.textContent=active?'已加入阅读清单':'加入待读';button.disabled=Boolean(active);
  const input={paper_id:paper.paper_id,task_id:taskId,action:'add',expected_version:item?.version||0,idempotency_key:key()};
  button.onclick=async()=>{button.disabled=true;try{const result=await api('/reading-list',input);button.textContent=result.status==='removed'?'已移出，请重新打开面板添加':'已加入阅读清单';}catch(e){button.disabled=false;onError(e);}};
  return button;
}

/** 展示全部活跃清单；使用textContent处理元数据，页内不执行生成或下载。 */
export async function showReadingList(onError, readOnly=false) {
  const body=openPanel('阅读清单');body.textContent='正在读取…';
  const data=await api('/reading-list');
  if(!body.isConnected||document.querySelector('#panel-title').textContent!=='阅读清单')return;
  body.replaceChildren();
  const hint=document.createElement('p');hint.className='status-line';hint.textContent='留住想读的论文。已读是你的阅读标记，不代表精读或核验通过。';body.append(hint);
  const items=data.items.filter(item=>item.status!=='removed');
  if(!items.length){const empty=document.createElement('p');empty.textContent='暂无待读论文。打开对话中的“论文与研究笔记”，从论文卡加入。';body.append(empty);}
  for(const item of items){
    const card=document.createElement('article');card.className='paper-card';
    const title=document.createElement('h3');title.textContent=item.paper.title||item.paper_id;card.append(title);
    const status=document.createElement('p');status.textContent=(item.status==='read'?'已读':'待读')+' · '+(item.paper.source||'来源未标注');card.append(status);
    const url=item.paper.paper_url||item.paper.pdf_url;
    if(/^https?:\/\//.test(url||'')){const link=document.createElement('a');link.href=url;link.rel='noopener';link.target='_blank';link.textContent='打开论文来源 ↗';card.append(link);}
    const controls=document.createElement('div');controls.className='message-actions';card.append(controls);
    for(const [action,label] of [[item.status==='read'?'unread':'read',item.status==='read'?'标记待读':'标记已读'],['remove','移出清单']]){
      const button=document.createElement('button');button.textContent=label;button.disabled=readOnly;
      const input={paper_id:item.paper_id,action,expected_version:item.version,idempotency_key:key()};
      button.onclick=async()=>{button.disabled=true;try{await api('/reading-list',input);await showReadingList(onError,readOnly);}catch(e){button.disabled=false;onError(e);}};
      controls.append(button);
    }
    body.append(card);
  }
}
