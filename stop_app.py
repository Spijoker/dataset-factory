#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集生成器关闭脚本
用于强制关闭所有相关的Streamlit进程
"""

import subprocess
import sys
import os
import psutil
import time

def find_streamlit_processes():
    """查找所有Streamlit相关进程"""
    streamlit_processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('streamlit' in str(cmd).lower() for cmd in cmdline):
                    streamlit_processes.append(proc)
                elif cmdline and any('app.py' in str(cmd) for cmd in cmdline):
                    streamlit_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        print(f"⚠️ 查找进程时出错: {e}")
    
    return streamlit_processes

def kill_processes_by_port(port=8501):
    """通过端口号查找并终止进程"""
    try:
        if sys.platform == "win32":
            # Windows系统
            result = subprocess.run(
                ["netstat", "-ano"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
                            print(f"✅ 已终止端口 {port} 上的进程 (PID: {pid})")
                        except subprocess.CalledProcessError:
                            print(f"⚠️ 无法终止进程 PID: {pid}")
        else:
            # Unix/Linux/macOS系统
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], 
                capture_output=True, 
                text=True
            )
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"✅ 已终止端口 {port} 上的进程 (PID: {pid})")
                    except subprocess.CalledProcessError:
                        print(f"⚠️ 无法终止进程 PID: {pid}")
                        
    except subprocess.CalledProcessError:
        print(f"⚠️ 无法查找端口 {port} 上的进程")
    except Exception as e:
        print(f"❌ 终止端口进程时出错: {e}")

def main():
    """主函数"""
    print("🛑 正在关闭数据集生成器...")
    print("-" * 40)
    
    # 方法1: 查找并终止Streamlit进程
    print("📋 查找Streamlit进程...")
    processes = find_streamlit_processes()
    
    if processes:
        print(f"🔍 找到 {len(processes)} 个相关进程")
        for proc in processes:
            try:
                print(f"⏹️  终止进程: {proc.info['name']} (PID: {proc.info['pid']})")
                proc.terminate()
                
                # 等待进程结束
                try:
                    proc.wait(timeout=3)
                    print(f"✅ 进程 {proc.info['pid']} 已正常终止")
                except psutil.TimeoutExpired:
                    print(f"⚠️ 进程 {proc.info['pid']} 未响应，强制终止...")
                    proc.kill()
                    proc.wait()
                    print(f"✅ 进程 {proc.info['pid']} 已强制终止")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"⚠️ 无法终止进程 {proc.info['pid']}: {e}")
    else:
        print("ℹ️  未找到Streamlit进程")
    
    # 方法2: 通过端口终止进程
    print("\n🔌 检查端口 8501...")
    kill_processes_by_port(8501)
    
    # 方法3: Windows特定的taskkill命令
    if sys.platform == "win32":
        print("\n🪟 使用Windows taskkill命令...")
        try:
            # 终止所有python进程中包含streamlit的
            subprocess.run([
                "taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *streamlit*"
            ], capture_output=True)
            
            # 终止所有包含app.py的python进程
            subprocess.run([
                "wmic", "process", "where", "name='python.exe' and CommandLine like '%app.py%'", "delete"
            ], capture_output=True)
            
            print("✅ Windows进程清理完成")
        except Exception as e:
            print(f"⚠️ Windows进程清理出错: {e}")
    
    print("\n" + "="*40)
    print("🎉 关闭操作完成！")
    print("💡 如果浏览器页面仍然显示，请手动刷新或关闭")
    print("💡 现在可以安全地重新启动应用: python run_app.py")
    
    # 等待用户确认
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()