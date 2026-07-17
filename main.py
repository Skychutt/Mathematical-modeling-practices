"""
数学建模 Python 练习 —— 项目入口

目录：
  01_python基础/     语法、NumPy、Matplotlib、SciPy
  02_启发式算法/     贪心、爬山、模拟退火、遗传、粒子群、禁忌、蚁群

推荐用法：
  1) pip install -r requirements.txt
  2) 直接运行某个练习文件，例如：
       python 01_python基础/01_变量与类型.py
       python 02_启发式算法/02_模拟退火.py
  3) 或运行本文件，快速抽查启发式算法是否可用：
       python main.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(rel: str):
    path = ROOT / rel
    print("\n" + "=" * 60)
    print(f"运行: {rel}")
    print("=" * 60)
    runpy.run_path(str(path), run_name="__main__")


def main():
    demos = [
        "01_python基础/01_变量与类型.py",
        "02_启发式算法/01_贪心与爬山法.py",
        "02_启发式算法/04_粒子群优化.py",
    ]
    # 允许：python main.py all  跑更多演示（较慢）
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        demos = [
            "01_python基础/01_变量与类型.py",
            "01_python基础/06_numpy基础.py",
            "02_启发式算法/01_贪心与爬山法.py",
            "02_启发式算法/02_模拟退火.py",
            "02_启发式算法/03_遗传算法.py",
            "02_启发式算法/04_粒子群优化.py",
            "02_启发式算法/05_禁忌搜索.py",
            "02_启发式算法/06_蚁群算法.py",
        ]

    for rel in demos:
        run_script(rel)

    print("\n完成。逐个打开对应 .py 文件阅读注释并修改参数练习即可。")


if __name__ == "__main__":
    main()
