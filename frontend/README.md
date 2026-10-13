# ResearchLoop 前端

原生JavaScript/HTML/CSS，聊天与个人面板两个入口，主题色 #fb7299。支持文字/链接/PDF、真实进度/SSE、追问/取消接续、Markdown/KaTeX、原文依据与成果侧栏；个人偏好、对话日期分组与改名已实现。

```bash
npm ci
npm test
npm run check
```

从仓库根 `make start` 统一启动，或单独 `make frontend`（需API/Worker）。默认前端4173、API18082，开发代理仅监听本机，不是公网服务器。Markdown经DOMPurify净化；只公开白名单依赖和字体。

components/evidence-matrix.js是保留的实验UI，不能以渲染夹具宣称模型矩阵交付通过。当前模块说明见 [文件地图](../docs/file-map.md)，后续计划见 [improvement-plan](../docs/improvement-plan.md)。
