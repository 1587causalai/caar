# 文档部署指南

本文档站点使用 [Docsify](https://docsify.js.org/) 构建，支持多种部署方式。

## 🚀 本地预览

### 方法一：使用 Python 脚本（推荐）

```bash
# 在项目根目录运行
python serve_docs.py

# 或指定端口
python serve_docs.py --port 3001

# 不自动打开浏览器
python serve_docs.py --no-browser
```

### 方法二：使用 docsify-cli

```bash
# 安装 docsify-cli
npm install -g docsify-cli

# 在 docs 目录启动服务
cd docs
docsify serve

# 或在项目根目录
docsify serve docs
```

### 方法三：使用 Python 内置服务器

```bash
cd docs
python -m http.server 3000
```

## 🌐 在线部署

### GitHub Pages（推荐）

1. **自动部署**：
   - 推送代码到 `main` 或 `master` 分支
   - GitHub Actions 会自动构建和部署
   - 访问 `https://1587causalai.github.io/caar`

2. **GitHub Pages 设置**：
   - 进入仓库 Settings → Pages
   - Source 选择 "Deploy from a branch"
   - Branch 选择 `gh-pages`
   - 保存设置

3. **故障排除**：
   - 如果遇到权限错误，进入 Settings → Actions → General，选择 "Read and write permissions"
   - 查看 Actions 标签页的部署日志

### Netlify

1. 连接 GitHub 仓库
2. 设置构建命令：`echo "Build complete"`
3. 设置发布目录：`docs`
4. 部署

### Vercel

1. 导入 GitHub 仓库
2. Framework Preset 选择 "Other"
3. Output Directory 设置为 `docs`
4. 部署

## 📝 文档编写

### 文件结构

```
docs/
├── index.html          # Docsify 配置
├── README.md           # 首页
├── _sidebar.md         # 侧边栏
├── _navbar.md          # 导航栏
├── .nojekyll          # 禁用 Jekyll
├── images/            # 图片资源
└── *.md               # 文档页面
```

### 添加新页面

1. 在 `docs/` 目录创建 `.md` 文件
2. 在 `_sidebar.md` 中添加链接
3. 提交并推送代码

### 图片使用

- 将图片放在 `docs/images/` 目录
- 使用相对路径引用：`![描述](images/图片名.png)`
- 支持点击放大功能

## 🎨 自定义配置

### 主题颜色

在 `index.html` 中修改：

```css
:root {
  --theme-color: #42b883;  /* 主题色 */
}
```

### 插件配置

当前已启用的插件：
- 🔍 全文搜索
- 📋 代码复制
- 📄 分页导航
- 📊 字数统计
- 🔍 图片缩放
- 🎨 代码高亮

### 添加新插件

在 `index.html` 中添加插件脚本：

```html
<script src="//cdn.jsdelivr.net/npm/plugin-name"></script>
```

## 🔧 故障排除

### 常见问题

1. **图片不显示**：
   - 检查图片路径是否正确
   - 确保图片文件存在于 `docs/images/` 目录

2. **侧边栏不显示**：
   - 确保 `_sidebar.md` 文件存在
   - 检查 `index.html` 中 `loadSidebar: true` 配置

3. **搜索不工作**：
   - 确保搜索插件已加载
   - 检查网络连接（CDN 资源）

4. **本地服务器启动失败**：
   - 检查端口是否被占用
   - 尝试使用其他端口

### 获取帮助

- [Docsify 官方文档](https://docsify.js.org/)
- [GitHub Issues](https://github.com/1587causalai/caar/issues)
- [社区讨论](https://github.com/docsifyjs/docsify/discussions)

---

*最后更新：2024年* 