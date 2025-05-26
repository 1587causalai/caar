#!/usr/bin/env python3
"""
图片格式还原脚本

将 Markdown 文档中的 HTML <img> 标签转换回 Markdown 图片语法
使用 ../results/ 路径格式以确保本地显示正常。
"""

import os
import re
from pathlib import Path

def convert_html_images_to_markdown(file_path):
    """将单个文件中的 HTML img 标签转换回 Markdown 图片语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换 HTML img 标签
        # 匹配模式：<img src="path" alt="alt_text" style="...">
        pattern = r'<img src="([^"]+)" alt="([^"]*)"[^>]*>'
        
        def replace_func(match):
            img_path = match.group(1)
            alt_text = match.group(2)
            
            # 确保使用 ../results/ 路径格式（本地显示友好）
            if img_path.startswith('results/'):
                img_path = '../' + img_path
            
            # 创建 Markdown 图片语法
            return f'![{alt_text}]({img_path})'
        
        new_content = re.sub(pattern, replace_func, content)
        
        # 检查是否有修改
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数：扫描并转换所有 Markdown 文件中的图片回 Markdown 格式"""
    
    # 要扫描的目录
    scan_dirs = ['docs']  # 只扫描 docs 目录
    
    converted_files = []
    total_files = 0
    
    for scan_dir in scan_dirs:
        if not os.path.exists(scan_dir):
            continue
            
        # 查找所有 .md 文件
        for root, dirs, files in os.walk(scan_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    
                    total_files += 1
                    print(f"检查文件: {file_path}")
                    
                    if convert_html_images_to_markdown(file_path):
                        converted_files.append(file_path)
                        print(f"  ✅ 已还原为 Markdown 格式: {file_path}")
                    else:
                        print(f"  ⏭️  无需还原: {file_path}")
    
    # 报告结果
    print(f"\nMarkdown 格式还原完成!")
    print(f"共检查了 {total_files} 个 Markdown 文件")
    print(f"还原了 {len(converted_files)} 个文件:")
    
    for file_path in converted_files:
        print(f"  - {file_path}")
    
    if len(converted_files) == 0:
        print("所有文件都没有需要还原的图片！")
    else:
        print(f"\n所有图片已还原为 Markdown 语法格式")
        print("使用 ../results/ 路径格式，确保本地显示正常。")
        print("\n示例还原:")
        print('  原格式: <img src="results/folder/image.png" alt="图片描述" style="max-width: 100%; height: auto;">')
        print("  新格式: ![图片描述](../results/folder/image.png)")

if __name__ == "__main__":
    main() 