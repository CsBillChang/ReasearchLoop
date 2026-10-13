/** 同源API客户端；显式错误保留输入，网络失败不自动重发有副作用请求。 */
export async function api(path, body, options={}) {
  const response=await fetch('/api'+path, {method:body===undefined?'GET':'POST',
    headers:body===undefined?{}:{'Content-Type':'application/json'},
    body:body===undefined?undefined:JSON.stringify(body), ...options});
  const data=await response.json().catch(()=>({detail:'服务返回无法读取的响应'}));
  if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  return data;
}
/** 上传PDF原始字节，使用服务器生成的身份；本地文件路径不会进入请求。 */
export async function upload(file) {
  return api('/uploads?filename='+encodeURIComponent(file.name),undefined,
    {method:'POST',headers:{'Content-Type':'application/pdf'},body:file});
}
/** 建立只读事件流；进度断线由EventSource按游标重连，不重新提交任务。 */
export function watch(taskId,onProgress,onSettled,onError) {
  const stream=new EventSource('/api/tasks/'+encodeURIComponent(taskId)+'/stream');
  stream.addEventListener('progress',e=>onProgress(JSON.parse(e.data)));
  stream.addEventListener('settled',e=>{stream.close();onSettled(JSON.parse(e.data));});
  stream.onerror=()=>onError?.();
  return stream;
}
export const key=()=>crypto.randomUUID();
