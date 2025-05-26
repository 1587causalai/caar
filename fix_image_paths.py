#!/usr/bin/env python3
"""
图片路径修复脚本

修复 Markdown 文档中的图片路径，将相对路径 ../results/ 替换为项目根目录路径 results/
这样可以确保图片在本地和 GitHub 部署环境中都能正确显示。
"""

import os
import re
from pathlib import Path

def fix_image_paths_in_file(file_path):
    """修复单个文件中的图片路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换图片路径
        # 匹配模式：![任意文本](../results/任意路径)
        pattern = r'!\[([^\]]*)\]\(\.\./results/([^)]+)\)'
        replacement = r'![\1](results/\2)'
        
        new_content = re.sub(pattern, replacement, content)
        
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
    """主函数：扫描并修复所有 Markdown 文件中的图片路径"""
    
    # 要扫描的目录
    scan_dirs = ['docs', '.']  # docs 目录和根目录
    
    fixed_files = []
    total_files = 0
    
    for scan_dir in scan_dirs:
        if not os.path.exists(scan_dir):
            continue
            
        # 查找所有 .md 文件
        for root, dirs, files in os.walk(scan_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    
                    # 跳过当前脚本可能生成的临时文件
                    if 'fix_image_paths' in file:
                        continue
                    
                    total_files += 1
                    print(f"检查文件: {file_path}")
                    
                    if fix_image_paths_in_file(file_path):
                        fixed_files.append(file_path)
                        print(f"  ✅ 已修复: {file_path}")
                    else:
                        print(f"  ⏭️  无需修复: {file_path}")
    
    # 报告结果
    print(f"\n修复完成!")
    print(f"共检查了 {total_files} 个 Markdown 文件")
    print(f"修复了 {len(fixed_files)} 个文件:")
    
    for file_path in fixed_files:
        print(f"  - {file_path}")
    
    if len(fixed_files) == 0:
        print("所有文件的图片路径都是正确的！")
    else:
        print(f"\n所有 '../results/' 路径已替换为 'results/' 路径")
        print("现在图片应该在本地和 GitHub 环境中都能正确显示。")

if __name__ == "__main__":
    main() 