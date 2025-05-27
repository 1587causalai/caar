# 图片部署指南

本文档说明如何正确处理项目文档中的图片路径，以确保在本地开发和GitHub Pages部署时都能正常显示。

## 问题背景

- 实验结果图片存储在 `results/` 目录下
- 文档存储在 `docs/` 目录下
- 文档中引用了大量的实验结果图片
- GitHub Pages 从 `docs/` 目录提供服务

## 解决方案

### 方案1：使用符号链接（仅本地开发）

```bash
cd docs
ln -s ../results results
```

这样可以在本地开发时正常查看图片，但GitHub Pages不支持符号链接。

### 方案2：复制图片到docs目录（推荐用于部署）

运行部署脚本：

```bash
./deploy_docs.sh
```

这会将 `results/` 目录复制到 `docs/` 下，确保GitHub Pages能够访问图片。

### 方案3：使用GitHub仓库的raw链接

如果不想复制大量图片，可以修改图片路径为GitHub raw链接：

```markdown
![图片描述](https://raw.githubusercontent.com/1587causalai/robust-regression-experiment/main/results/xxx.png)
```

但这种方式需要修改所有图片链接，且依赖网络连接。

## 当前采用的方案

1. **本地开发**：使用符号链接（已创建）
2. **GitHub Pages部署**：运行 `deploy_docs.sh` 脚本复制图片

## 注意事项

- `results/` 目录已在 `.gitignore` 中，不会被提交
- 如果要部署到GitHub Pages，需要将 `docs/results/` 提交到仓库
- 可以考虑使用Git LFS来管理大量图片文件 