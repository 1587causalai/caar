#!/usr/bin/env python3
"""
Fix image paths in Markdown files for GitHub compatibility.

This script converts relative image paths (../results/...) to absolute paths
from the project root (results/...) so they work both locally and on GitHub.
"""

import os
import re
import glob
from pathlib import Path

def fix_image_paths_in_file(file_path):
    """Fix image paths in a single Markdown file."""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match image references with relative paths
    # Matches: ![alt text](../results/path/to/image.png)
    pattern = r'!\[([^\]]*)\]\(\.\./results/([^)]+)\)'
    
    # Replace with absolute path from project root
    # Replace: ![alt text](../results/path/to/image.png)
    # With:    ![alt text](results/path/to/image.png)
    def replacement(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        return f'![{alt_text}](results/{image_path})'
    
    # Count replacements for reporting
    original_content = content
    content = re.sub(pattern, replacement, content)
    
    # Count how many replacements were made
    replacements = len(re.findall(pattern, original_content))
    
    if replacements > 0:
        # Write back the modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Fixed {replacements} image paths")
    else:
        print(f"  - No image paths to fix")
    
    return replacements

def main():
    """Main function to process all Markdown files."""
    print("🔧 Fixing image paths in Markdown files for GitHub compatibility...")
    print("=" * 60)
    
    # Find all Markdown files in the docs directory
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("❌ Error: docs/ directory not found!")
        return
    
    md_files = list(docs_dir.glob("*.md"))
    
    if not md_files:
        print("❌ No Markdown files found in docs/ directory!")
        return
    
    total_replacements = 0
    
    for md_file in md_files:
        replacements = fix_image_paths_in_file(md_file)
        total_replacements += replacements
    
    print("=" * 60)
    print(f"✅ Processing complete!")
    print(f"📊 Total files processed: {len(md_files)}")
    print(f"🔄 Total image paths fixed: {total_replacements}")
    
    if total_replacements > 0:
        print("\n📝 Changes made:")
        print("   Before: ![alt](../results/path/image.png)")
        print("   After:  ![alt](results/path/image.png)")
        print("\n🚀 Your images should now work on both local and GitHub!")
    else:
        print("\n✨ No changes needed - all paths are already correct!")

if __name__ == "__main__":
    main() 