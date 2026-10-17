# 第三方代码与模型归属

## daily-paper-reader

上游作者 ziwenhahaha，MIT License。项目复用/适配其来源RPC映射、正文获取思路和精读函数；主要落点：

- backend/research_loop/providers/dpr_summary.py：五个精读输入/生成函数及提示，保留DPR三轮重生成语义。
- providers/dpr_sources.py、discovery.py、dpr_acquisition.py、pdf.py：来源和获取适配；ResearchLoop增加任务持久化、授权与观察。
- RRF与图表提取路径参考其实现；不是复制整套上游应用。

完整许可证原文保留于下节。来源配置和访问凭据不随源码分发。

### DPR MIT 许可证原文

```text
MIT License

Copyright (c) 2026 ziwenhahaha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

文档精简后，原 `docs/licenses/daily-paper-reader.txt` 的版权与许可声明已完整迁入本节；源码中的旧路径注释按此对应。此次未改动业务源码。

## 运行依赖与设计参考

Python/前端依赖以 backend/uv.lock、frontend/package-lock.json 为准，各自许可证独立适用；公开源码不把所有依赖重新许可。PyMuPDF及其他可选组件请按其发行许可证使用。下载的参考仓库、PDF论文和模型权重未打包。

E5模型固定版本和重排模型固定版本见docs/references.md；仓库只提供清单模板，下载者保留模型卡和许可证。论文CRITIC/CoVe/RARR及CRAG/PaperQA/Hello-Agents等是方法/设计参考，实际复制与仅参考明确区分。

ResearchLoop整体尚未另行指定开源再分发许可证。本文件保留第三方归属，不替用户选择全仓库许可，也不表示第三方MIT覆盖所有项目内容。
