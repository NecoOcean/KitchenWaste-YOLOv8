# run.py
# -*- coding: utf-8 -*-
"""
启动脚本 - 从项目根目录运行主程序
"""
import sys
import os

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if __name__ == '__main__':
    from core.main import main
    main()
