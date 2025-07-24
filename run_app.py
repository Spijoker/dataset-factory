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
from pathlib import Path

# 全局变量用于存储子进程
streamlit_process = None

def signal_handler(signum, frame):
    """信号处理函数，用于优雅地关闭应用"""
    global streamlit_process
    print("\n🛑 接收到中断信号，正在关闭应用...")
    
    if streamlit_process:
        try:
            # 尝试优雅地终止进程
            streamlit_process.terminate()
            print("⏳ 等待进程结束...")
            
            # 等待最多5秒
            try:
                streamlit_process.wait(timeout=5)
                print("✅ 应用已正常关闭")
            except subprocess.TimeoutExpired:
                print("⚠️ 进程未在5秒内结束，强制终止...")
                streamlit_process.kill()
                streamlit_process.wait()
                print("✅ 应用已强制关闭")
        except Exception as e:
            print(f"❌ 关闭过程中出现错误: {e}")
    
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
    print("⏹️  停止服务: 在终端中按 Ctrl+C")
    print("💡 提示: 现在可以随时按 Ctrl+C 强制中止程序")
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