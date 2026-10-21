/** 包装marked的裸URL解析，在中文句读处停止，尾注仍交回普通Markdown解析。
 * @param {Function} originalUrl 锁定marked版本的默认URL tokenizer。
 * @returns {Function} 保留默认解析上下文和非裸链接行为的tokenizer。
 */
export function withChineseUrlBoundary(originalUrl) {
  return function url(src) {
    const token=originalUrl.call(this,src);
    if(!token) return false;
    const boundary=token.raw.search(/[，。；！？、（）【】《》“”‘’]/u);
    if(boundary<0) return token;
    // 只重读裸URL前缀；显式链接、尖括号链接与代码由marked其他tokenizer处理。
    // 不在href生成后裁字符串，以免丢掉原本应显示的中文尾注。
    return originalUrl.call(this,src.slice(0,boundary)) || false;
  };
}
