#!/usr/bin/env python3
"""
图片格式转换脚本

将 Markdown 文档中的图片从 Markdown 语法转换为 HTML <img> 标签
这样可能解决本地和远程显示的兼容性问题。
"""

import os
import re
from pathlib import Path

def convert_markdown_images_to_html(file_path):
    """将单个文件中的 Markdown 图片转换为 HTML img 标签"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换图片语法
        # 匹配模式：![alt text](path)
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        
        def replace_func(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 统一使用 results/ 路径（不带 ../）
            if img_path.startswith('../results/'):
                img_path = img_path[3:]  # 去掉 ../
            
            # 创建 HTML img 标签
            return f'<img src="{img_path}" alt="{alt_text}" style="max-width: 100%; height: auto;">'
        
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
    """主函数：扫描并转换所有 Markdown 文件中的图片为 HTML 格式"""
    
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
                    
                    if convert_markdown_images_to_html(file_path):
                        converted_files.append(file_path)
                        print(f"  ✅ 已转换为 HTML 格式: {file_path}")
                    else:
                        print(f"  ⏭️  无需转换: {file_path}")
    
    # 报告结果
    print(f"\nHTML 格式转换完成!")
    print(f"共检查了 {total_files} 个 Markdown 文件")
    print(f"转换了 {len(converted_files)} 个文件:")
    
    for file_path in converted_files:
        print(f"  - {file_path}")
    
    if len(converted_files) == 0:
        print("所有文件都没有需要转换的图片！")
    else:
        print(f"\n所有图片已转换为 HTML <img> 标签格式")
        print("现在图片应该在本地和远程环境中都能正确显示。")
        print("\n示例转换:")
        print("  原格式: ![图片描述](../results/folder/image.png)")
        print("  新格式: <img src=\"results/folder/image.png\" alt=\"图片描述\" style=\"max-width: 100%; height: auto;\">")

if __name__ == "__main__":
    main() 