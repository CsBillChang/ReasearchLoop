# ResearchLoop 前端

原生JavaScript ES Modules / HTML / CSS，聊天与个人面板两个入口。白色阅读主题以近白侧栏、暖灰文字、浅粉选择态和森林绿按钮（#52664f）组织内容，标题使用本机衬线字体，正文使用系统字体，不依赖远端字体。支持文字/链接/PDF、真实进度/SSE、追问/取消接续、Markdown/KaTeX、原文依据与成果侧栏；个人偏好、对话日期分组与改名已实现。窄屏侧栏采用抽屉，支持键盘焦点隔离、Escape退出与焦点返回。

```bash
npm ci
npm test
npm run check
```

从仓库根 `make start` 统一启动，或单独 `make frontend`（需API/Worker）。默认前端4173、API18082，开发代理仅监听本机，不是公网服务器。Markdown经DOMPurify净化；只公开白名单依赖和字体。

components/evidence-matrix.js是保留的实验UI，不能以渲染夹具宣称模型矩阵交付通过。功能与架构见 [项目说明](../README.md)，接口与后端分层见 [后端说明](../backend/README.md)。
