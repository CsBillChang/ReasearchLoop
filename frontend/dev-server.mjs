// 本地开发服务器：提供有限静态文件，并把 /api/ 原样代理给 Python 后端。
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('.', import.meta.url));
const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.woff2':'font/woff2', '.woff':'font/woff', '.ttf':'font/ttf' };

/** 输入本地 API 地址，返回尚未监听的 HTTP server，便于测试使用独立端口。 */
export function createServer(apiBase = 'http://127.0.0.1:18082') {
  const target = new URL(apiBase);
  if (target.protocol !== 'http:') throw new Error('The local development API must use http:');
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname.startsWith('/api/')) {
      const proxy = http.request({ hostname: target.hostname, port: target.port || 80,
        path: url.pathname + url.search, method: req.method,
        headers: { ...req.headers, host: target.host } }, (upstream) => {
        res.writeHead(upstream.statusCode, upstream.headers);
        upstream.pipe(res);
      });
      proxy.on('error', () => { if (!res.headersSent) res.writeHead(502); res.end('API unavailable'); });
      req.pipe(proxy);
      return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405); res.end(); return;
    }
    let pathname;
    try { pathname = decodeURIComponent(url.pathname); } catch { res.writeHead(400); res.end(); return; }
    // 只公开首页和 src；配置、服务端代码、依赖描述文件不应经静态路由泄露。
    let filename = pathname === '/' ? '/index.html' : pathname;
    const vendor = {
      '/vendor/marked.js':'/node_modules/marked/lib/marked.esm.js',
      '/vendor/purify.js':'/node_modules/dompurify/dist/purify.es.mjs',
      '/vendor/katex.js':'/node_modules/katex/dist/katex.mjs',
      '/vendor/katex.css':'/node_modules/katex/dist/katex.min.css'
    };
    if (vendor[filename] || /^\/vendor\/fonts\/[A-Za-z0-9_-]+\.(woff2?|ttf)$/.test(filename)) {
      filename = vendor[filename] || filename.replace('/vendor/fonts/','/node_modules/katex/dist/fonts/');
      try {
        const body=await readFile(path.join(root,filename));
        res.writeHead(200,{'Content-Type':mime[path.extname(filename)] || 'text/javascript; charset=utf-8'});
        res.end(req.method === 'HEAD' ? undefined : body);
      } catch {res.writeHead(404);res.end();}
      return;
    }
    if (!(filename === '/index.html' || filename.startsWith('/src/')) || filename.includes('..') || filename.includes('\\') || filename.includes('\0')) {
      res.writeHead(404); res.end(); return;
    }
    const absolute = path.resolve(root, '.' + filename);
    if (!absolute.startsWith(root)) { res.writeHead(404); res.end(); return; }
    try {
      const body = await readFile(absolute);
      res.writeHead(200, { 'Content-Type': mime[path.extname(absolute)] || 'application/octet-stream', 'Cache-Control':'no-store' });
      res.end(req.method === 'HEAD' ? undefined : body);
    } catch { res.writeHead(404); res.end(); }
  });
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number(process.env.PORT || 4173);
  createServer(process.env.RESEARCHLOOP_API_URL).listen(port, '127.0.0.1', () => console.log(`ResearchLoop frontend: http://127.0.0.1:${port}`));
}
