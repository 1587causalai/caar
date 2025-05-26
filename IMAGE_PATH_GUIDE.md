# 图片路径修复指南

## 问题描述

在 Markdown 文档中引用图片时，本地和 GitHub 部署环境对路径的解析方式不同：

- **本地环境**：相对路径基于当前 Markdown 文件的位置
- **GitHub 环境**：路径基于项目根目录

这导致使用相对路径 `../results/image.png` 的图片在本地能正常显示，但在 GitHub 上无法显示。

## 解决方案

### ✅ 推荐做法：使用项目根目录的绝对路径

```markdown
<!-- ❌ 错误：相对路径 -->
![图片描述](../results/folder/image.png)

<!-- ✅ 正确：从项目根目录开始的路径 -->
![图片描述](results/folder/image.png)
```

### 🔧 批量修复工具

项目中提供了 `fix_image_paths.py` 脚本来批量修复所有图片路径：

```bash
python fix_image_paths.py
```

该脚本会：
1. 扫描 `docs/` 目录下的所有 `.md` 文件
2. 将 `../results/` 路径替换为 `results/`
3. 报告修复的文件数量和路径数量

## 最佳实践

### 1. 图片组织结构
```
project-root/
├── docs/           # 文档目录
│   ├── report1.md
│   └── report2.md
├── results/        # 图片目录
│   ├── experiment1/
│   │   ├── chart1.png
│   │   └── chart2.png
│   └── experiment2/
└── README.md
```

### 2. 路径引用规则
- **在 `docs/` 目录下的文档**：使用 `results/subfolder/image.png`
- **在根目录的文档**：使用 `results/subfolder/image.png`
- **避免使用**：`../`, `./`, 绝对系统路径

### 3. 验证方法
在提交到 GitHub 前，可以通过以下方式验证：
1. 在项目根目录启动本地服务器查看
2. 使用 GitHub Pages 预览功能
3. 检查 GitHub 仓库中的图片链接

## 技术原理

### GitHub Markdown 渲染
GitHub 在渲染 Markdown 时：
- 将所有路径解析为相对于仓库根目录
- 不支持 `../` 这样的父目录引用
- 图片路径直接映射到仓库文件结构

### 本地 Markdown 查看器
大多数本地 Markdown 查看器：
- 基于当前文件位置解析相对路径
- 支持 `../` 父目录引用
- 路径解析更加灵活

## 自动化建议

### Git Pre-commit Hook
可以设置 Git 钩子在提交前自动检查和修复路径：

```bash
#!/bin/sh
# .git/hooks/pre-commit
python fix_image_paths.py
git add docs/
```

### CI/CD 集成
在持续集成流程中添加路径检查：

```yaml
# .github/workflows/check-images.yml
- name: Check image paths
  run: |
    python fix_image_paths.py
    if [ -n "$(git diff --name-only)" ]; then
      echo "Image paths were fixed. Please commit the changes."
      exit 1
    fi
```

## 故障排除

### 图片仍然无法显示
1. 检查图片文件是否存在
2. 验证文件名大小写是否正确
3. 确认图片格式是否支持（PNG, JPG, GIF, SVG）
4. 检查文件权限和 Git 跟踪状态

### 路径修复后本地无法显示
这是正常现象，因为：
- 修复后的路径是为 GitHub 优化的
- 本地查看建议使用支持项目根目录的 Markdown 编辑器
- 或者在项目根目录启动 HTTP 服务器查看

## 相关工具

- **VS Code**：支持项目根目录路径预览
- **Typora**：可配置图片根目录
- **GitHub Pages**：自动支持正确的路径解析
- **GitBook**：原生支持项目根目录路径

---

*最后更新：2024年* 