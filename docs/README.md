# STREAM research presentation

Interactive static presentation of STREAM: Sequential Transformer for Risk Estimation from Asynchronous Medical Events for Dynamic Postoperative AKI Prediction.

## 发布到 GitHub Pages

1. 新建 GitHub 公开仓库，例如 `stream-presentation`。
2. 将本压缩包解压，把其中的文件和 `assets` 文件夹上传到仓库根目录。`index.html` 应直接位于根目录，不要再包一层文件夹；不要直接上传 ZIP。
3. 提交到 `main` 分支。
4. 打开仓库 **Settings → Pages**。
5. 在 **Build and deployment** 中选择 **Deploy from a branch**，分支选择 **main**，目录选择 **/(root)**，点击 **Save**。
6. 部署完成后，Pages 设置页会提供访问地址。后续提交到同一分支会触发更新。

官方说明：https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## 与模型代码放在同一个仓库

如果复用已有研究代码仓库，可把本包内容放到 `main` 分支的 `docs/` 目录，确保入口是 `docs/index.html`。Pages 中选择 `main` 和 `/docs`。保留原有代码文件；如已有 `docs` 内容，请合并而非覆盖整个目录。

## 包含内容

- `index.html`：入口
- `style.css`：样式与响应式布局
- `app.js`：18章内容和交互
- `evidence.json`：表格、算法、参考文献等展示数据
- `assets/`：原始图表与机构标识
- `.nojekyll`：静态文件发布标记

已移除 manuscript 和 supplementary 的 Word 文件及两个下载入口。图表、表格、参考文献、英文讲者备注和交互均保留；此包不包含预测模型权重或推理服务。

## 本地预览

由于网页需要读取 `evidence.json`，请通过 HTTP 服务预览，不要直接双击 HTML。
在解压目录运行：

```bash
python -m http.server 8000
```

浏览器打开 http://localhost:8000/ 。本地命令只用于预览，GitHub Pages 不需要 Python 或构建步骤。

## 使用方式

左右方向键切换章节；G 打开章节目录；N 切换讲者备注。页面提供阅读模式、全屏、图表放大与表格筛选。

Google Fonts 字体通过网络加载；不可用时使用系统字体。外部文献和研究代码链接需要联网。
