#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集生成器快速启动脚本
简化的启动入口，直接从根目录启动应用
"""

import subprocess
import sys
from pathlib import Path

def main():
    """启动应用"""
    # 获取scripts目录下的run_app.py
    script_path = Path(__file__).parent / "scripts" / "run_app.py"
    
    if not script_path.exists():
        print("❌ 错误: 找不到启动脚本")
        return
    
    # 执行启动脚本
    subprocess.run([sys.executable, str(script_path)])

if __name__ == "__main__":
    main()