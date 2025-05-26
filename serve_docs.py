#!/usr/bin/env python3
"""
本地文档服务器启动脚本
用于预览 docsify 文档站点
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

def serve_docs(port=3000, open_browser=True):
    """启动文档服务器"""
    
    # 切换到 docs 目录
    docs_dir = Path(__file__).parent / "docs"
    if not docs_dir.exists():
        print("❌ docs 目录不存在！")
        sys.exit(1)
    
    os.chdir(docs_dir)
    
    # 创建服务器
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}"
            print(f"🚀 文档服务器已启动: {url}")
            print(f"📁 服务目录: {docs_dir.absolute()}")
            print("按 Ctrl+C 停止服务器")
            
            # 自动打开浏览器
            if open_browser:
                print("🌐 正在打开浏览器...")
                webbrowser.open(url)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ 端口 {port} 已被占用，请尝试其他端口")
            print(f"例如: python {sys.argv[0]} --port 3001")
        else:
            print(f"❌ 启动服务器失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动 docsify 文档服务器")
    parser.add_argument("--port", "-p", type=int, default=3000, 
                       help="服务器端口 (默认: 3000)")
    parser.add_argument("--no-browser", action="store_true", 
                       help="不自动打开浏览器")
    
    args = parser.parse_args()
    
    serve_docs(port=args.port, open_browser=not args.no_browser) 