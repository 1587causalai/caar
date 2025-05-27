#!/bin/bash

# 部署文档到GitHub Pages的脚本
# 该脚本会将results目录复制到docs目录下，确保图片能在GitHub Pages中正常显示

echo "准备部署文档到GitHub Pages..."

# 检查是否在项目根目录
if [ ! -d "docs" ] || [ ! -d "results" ]; then
    echo "错误：请在项目根目录运行此脚本"
    exit 1
fi

# 删除docs目录下的旧results（如果是目录而非符号链接）
if [ -d "docs/results" ] && [ ! -L "docs/results" ]; then
    echo "删除旧的results目录..."
    rm -rf docs/results
fi

# 如果存在符号链接，先删除
if [ -L "docs/results" ]; then
    echo "删除符号链接..."
    rm docs/results
fi

# 复制results目录到docs目录
echo "复制results目录到docs目录..."
cp -r results docs/

echo "部署准备完成！"
echo "现在可以提交并推送到GitHub："
echo "  git add docs/results"
echo "  git commit -m 'Add results images for GitHub Pages'"
echo "  git push" 