/** 矩阵只展示持久产物；所有文本转义，原文定位复用现有任务证据入口。 */
const labels={question:'研究问题',method:'方法',dataset:'数据集',metric:'指标与实验条件',limitation:'限制'};
const states={evidenced:'有原文引句',uncertain:'尚不确定',conflict:'存在冲突',missing:'当前材料缺证'};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/** 生成无脚本标记；来源hash与Agent状态分列，不显示“核验通过”。 */
export function matrixMarkup(matrices) {
  if(!matrices?.length)return '<p class="status-line">本任务没有整理结构化证据矩阵；原回答与笔记仍可阅读。</p>';
  return matrices.map((m,i)=>`<details class="evidence-matrix" ${i===0?'open':''}>
    <summary>${i===0?'最新矩阵':'历史矩阵'} · ${esc(m.matrix_hash.slice(0,12))} · ${esc(m.created_at)}</summary>
    <h3>${esc(m.question)}</h3>
    <p class="${m.integrity==='intact'?'status-line':'warning'}">${m.integrity==='intact'?'来源身份与引句已检查；未进行矩阵语义审校。':'来源或产物已变化，此版本仅作历史参考。'}</p>
    <p class="status-line">各行保留各自实验条件，不代表可以直接排名。矩阵与笔记分别版本化。</p>
    ${m.rows.map(r=>`<article class="matrix-row"><h3>${esc(r.title)}</h3>
      <details class="matrix-source"><summary>来源版本与查阅范围</summary>
        <p>原文 SHA-256：${esc(r.content_sha256)}<br>文档：${esc(r.document_id)}<br>解析器：${esc(r.parser_name)} ${esc(r.parser_version)}</p>
        <p>实际范围：${[...r.scope].sort((a,b)=>(a.page_number??0)-(b.page_number??0)).map(s=>s.page_number?'第 '+s.page_number+' 页':'正文（无物理页码）').join('、')}</p>
        ${r.scope.filter(s=>s.gaps?.length).map(s=>`<p class="warning">${esc(s.gaps.join('；'))}</p>`).join('')}
      </details>
      <div class="matrix-cells">${r.cells.map(c=>`<section class="matrix-cell" data-cell-status="${esc(c.status)}">
        <h4>${labels[c.dimension]||esc(c.dimension)} <span>${states[c.status]||esc(c.status)}</span></h4>
        <p>${esc(c.value||'—')}</p>
        ${c.conditions.length?'<p><strong>适用条件：</strong>'+c.conditions.map(esc).join('；')+'</p>':''}
        ${c.reason?'<p class="matrix-reason">'+esc(c.reason)+'</p>':''}
        ${c.references.map(ref=>`<details><summary>原文引句 · ${ref.page_number?'第 '+ref.page_number+' 页':'无物理页码'}</summary>
          <blockquote>${esc(ref.quote)}</blockquote><button data-matrix-evidence="${esc(ref.evidence_id)}" ${m.integrity!=='intact'?'disabled':''}>查看该处原文</button></details>`).join('')}
      </section>`).join('')}</div></article>`).join('')}
  </details>`).join('');
}

/** 绑定原文按钮，保持当前任务权限；不提交模型或写请求。 */
export function renderMatrices(element, matrices, onEvidence) {
  element.innerHTML=matrixMarkup(matrices);
  element.querySelectorAll('[data-matrix-evidence]').forEach(b=>{b.onclick=()=>onEvidence(b.dataset.matrixEvidence);});
}
