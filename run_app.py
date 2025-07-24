#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集生成器启动脚本
用于启动Streamlit可视化界面
"""

import subprocess
import sys
import os
import signal
import time
import atexit
from pathlib import Path

# 全局变量用于存储子进程
streamlit_process = None

def cleanup_processes():
    """清理所有相关进程"""
    global streamlit_process
    
    if streamlit_process:
        try:
            # 尝试优雅地终止进程
            streamlit_process.terminate()
            print("⏳ 等待进程结束...")
            
            # 等待最多3秒
            try:
                streamlit_process.wait(timeout=3)
                print("✅ 应用已正常关闭")
            except subprocess.TimeoutExpired:
                print("⚠️ 进程未在3秒内结束，强制终止...")
                streamlit_process.kill()
                streamlit_process.wait()
                print("✅ 应用已强制关闭")
        except Exception as e:
            print(f"❌ 关闭过程中出现错误: {e}")
    
    # Windows系统额外清理
    if sys.platform == "win32":
        try:
            print("🧹 清理残留进程...")
            # 终止端口8501上的进程
            subprocess.run([
                "powershell", "-Command", 
                "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
            ], capture_output=True)
            
            # 终止包含streamlit的python进程
            subprocess.run([
                "taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *streamlit*"
            ], capture_output=True)
            
            print("✅ 清理完成")
        except Exception as e:
            print(f"⚠️ 清理过程中出现警告: {e}")

def signal_handler(signum, frame):
    """信号处理函数，用于优雅地关闭应用"""
    print("\n🛑 接收到中断信号，正在关闭应用...")
    cleanup_processes()
    sys.exit(0)

def main():
    """
    启动Streamlit应用
    """
    global streamlit_process
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    # 注册退出时的清理函数
    atexit.register(cleanup_processes)
    
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    app_file = current_dir / "app.py"
    
    # 检查app.py是否存在
    if not app_file.exists():
        print("❌ 错误: 找不到app.py文件")
        return
    
    # 检查是否安装了streamlit
    try:
        import streamlit
        print("✅ Streamlit已安装")
    except ImportError:
        print("❌ 错误: 未安装Streamlit")
        print("请运行: pip install -r requirements.txt")
        return
    
    print("🚀 正在启动数据集生成器可视化界面...")
    print("📱 界面将在浏览器中自动打开")
    print("🔗 如果没有自动打开，请访问: http://localhost:8501")
    print("⏹️  停止服务方式:")
    print("   1. 在终端中按 Ctrl+C (推荐)")
    print("   2. 运行关闭脚本: python stop_app.py")
    print("   3. Windows用户: 双击 stop_app.bat")
    print("💡 提示: 程序现在支持更好的进程清理机制")
    print("-" * 50)
    
    # 启动Streamlit应用
    try:
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.headless", "false",
            "--global.developmentMode", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
        
        # 等待进程结束
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        if streamlit_process:
            streamlit_process.terminate()

if __name__ == "__main__":
    main()