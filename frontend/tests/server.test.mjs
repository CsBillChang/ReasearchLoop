// 开发服务器测试：检查静态文件白名单和 API 代理，不等同于浏览器点击验收。
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { createServer } from '../dev-server.mjs';

async function start(server) {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${server.address().port}`;
}
function close(server) { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); }

test('public page loads and private development files are not served', async () => {
  const server = createServer(); const base = await start(server);
  try {
    const response = await fetch(base);
    assert.equal(response.status, 200);
    assert.match(await response.text(), /你的个人科研助手/);
    for (const pathname of ['/package.json','/dev-server.mjs','/.env','/src/%2e%2e%2fpackage.json']) {
      assert.equal((await fetch(base+pathname)).status, 404, pathname);
    }
  } finally { await close(server); }
});

test('same-origin API proxy preserves request and upstream status', async () => {
  const upstream = http.createServer((req,res) => { res.writeHead(202, {'content-type':'application/json'}); res.end(JSON.stringify({path:req.url,method:req.method})); });
  const target = await start(upstream); const server = createServer(target); const base = await start(server);
  try {
    const response = await fetch(base+'/api/health?check=1');
    assert.equal(response.status, 202);
    assert.deepEqual(await response.json(), {path:'/api/health?check=1',method:'GET'});
  } finally { await close(server); await close(upstream); }
});


test('vendor whitelist serves locked renderers but not arbitrary node_modules', async()=>{
  const server=createServer(); const base=await start(server);
  try {
    for (const file of ['/vendor/marked.js','/vendor/purify.js','/vendor/katex.js','/vendor/katex.css']) assert.equal((await fetch(base+file)).status,200);
    assert.equal((await fetch(base+'/node_modules/marked/package.json')).status,404);
    assert.equal((await fetch(base+'/vendor/../package.json')).status,404);
  } finally {await close(server);}
});
