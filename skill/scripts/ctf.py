#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf.py — ctf-toolkit 统一入口（规范性 CLI 见 E:\\gpu_crack\\CLI_REFERENCE.md）

职责
----
1. 自动挑选解释器：CTF_PYTHON（若设置且存在）→ Python315（torch+CUDA）→ 当前解释器。
2. 把 <module> 映射到工具箱里的实际脚本（默认 E:\\gpu_crack，可用 CTF_TOOLKIT_DIR 覆盖）。
3. 全局参数 --backend/--chunk 放在模块前后均可（内部自动挪到子命令之前）。
4. `selftest` 依次跑 gpu bench + ctf_tools/ctf_ext/ctf_misc 三个模块自测，退出码聚合。

用法
----
    python ctf.py <module> [args...]        # module 即子命令名，如 rsa / mtscan / lsb
    python ctf.py <module> --help           # 查看该模块的参数
    python ctf.py selftest                  # 四模块组合自测（全 PASS 退出码 0）
    python ctf.py help

示例
----
    python ctf.py rsa --n 0x... --e 65537 --c 0x...
    python ctf.py mtscan --key-hex <hex> --seed-count 1000000
    python ctf.py --backend numpy xorbrute --data-hex <hex> --max-len 1

退出码
------
    0  成功（未命中/未攻破同样为 0，看输出判断）
    1  自测失败 / 无参数（打印帮助）
    2  未知模块 / 参数或输入错误（转发子进程退出码）
"""

from __future__ import annotations

import os
import subprocess
import sys

TOOLKIT = os.environ.get('CTF_TOOLKIT_DIR', r'E:\gpu_crack')

# module -> script
SCRIPTS = {
    'bench': 'gpu_crack.py', 'mtscan': 'gpu_crack.py', 'sboxscan': 'gpu_crack.py',
    'ksscore': 'gpu_crack.py', 'lfsrscan': 'gpu_crack.py',
    'chacha': 'ctf_tools.py', 'poly1305': 'ctf_tools.py', 'rs63': 'ctf_tools.py',
    'crc32': 'ctf_tools.py', 'xorbrute': 'ctf_tools.py', 'aesctr': 'ctf_tools.py',
    'tea': 'ctf_ext.py', 'speck': 'ctf_ext.py', 'rc4': 'ctf_ext.py', 'lcg': 'ctf_ext.py',
    'md5': 'ctf_ext.py', 'rsa': 'ctf_ext.py', 'encode': 'ctf_ext.py',
    'lsb': 'ctf_misc.py', 'stegscan': 'ctf_misc.py', 'imgdiff': 'ctf_misc.py',
    'bitplane': 'ctf_misc.py', 'cyclic': 'ctf_misc.py', 'cycfind': 'ctf_misc.py',
    'gadgets': 'ctf_misc.py', 'elfinfo': 'ctf_misc.py',
}

# 优先使用的 GPU 解释器候选
PY_CANDIDATES = [
    os.environ.get('CTF_PYTHON', ''),
    r'C:\Users\<user>\AppData\Local\Programs\Python\Python315\python.exe',
    sys.executable,
]


def pick_python() -> str:
    """挑一个 import torch 且 CUDA 可用的解释器；都不行就返回当前解释器。

    显式设置了 CTF_PYTHON 时优先尊重（哪怕没有 CUDA，也能用 numpy 后端跑）。
    """
    env_py = os.environ.get('CTF_PYTHON', '')
    if env_py and os.path.exists(env_py):
        return env_py
    for p in PY_CANDIDATES:
        if not p or not os.path.exists(p):
            continue
        try:
            r = subprocess.run([p, '-c', 'import torch;assert torch.cuda.is_available()'],
                               capture_output=True, timeout=60)
            if r.returncode == 0:
                return p
        except Exception:
            continue
    return sys.executable


GLOBAL_OPTS = ('--backend', '--chunk')

# 每个脚本支持的全局选项（ctf_misc.py 是纯 CPU 的 PIL/capstone 工具，没有这些参数）
GLOBALS_OF = {
    'gpu_crack.py': GLOBAL_OPTS,
    'ctf_tools.py': GLOBAL_OPTS,
    'ctf_ext.py': GLOBAL_OPTS,
    'ctf_misc.py': (),
}


def filter_globals(gl, allowed):
    """按目标脚本支持的选项过滤全局参数，返回 (保留, 丢弃)。"""
    keep, drop = [], []
    i = 0
    while i < len(gl):
        opt = gl[i].split('=', 1)[0]
        tgt = keep if opt in allowed else drop
        if '=' in gl[i] or i + 1 >= len(gl):
            tgt.append(gl[i]); i += 1
        else:
            tgt.extend(gl[i:i + 2]); i += 2
    return keep, drop



def split_globals(args):
    """
    把模块级全局参数（--backend / --chunk）提到子命令之前。
    argparse 要求全局选项出现在子命令前面，否则子命令会报 unrecognized arguments。
    """
    head, tail = [], []
    i = 0
    while i < len(args):
        if args[i] in GLOBAL_OPTS:
            head.append(args[i])
            if i + 1 < len(args):
                head.append(args[i + 1])
                i += 2
                continue
        elif any(args[i].startswith(o + '=') for o in GLOBAL_OPTS):
            head.append(args[i])
        else:
            tail.append(args[i])
        i += 1
    return head, tail


def run(py: str, script: str, args: list) -> int:
    path = os.path.join(TOOLKIT, script)
    if not os.path.exists(path):
        print(f'[错误] 找不到 {path}；用 CTF_TOOLKIT_DIR 指定 toolkit 目录', file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    cmd = [py, '-X', 'utf8', path] + args
    return subprocess.call(cmd, env=env)


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        print('可用模块:', ', '.join(sorted(SCRIPTS)))
        return 1
    py = pick_python()

    # 允许 `ctf.py --backend numpy <module>` 这种写法：先吃掉最前面的全局选项
    gl_front, i = [], 0
    while i < len(argv):
        if argv[i] in GLOBAL_OPTS:
            if i + 1 >= len(argv):
                print(f'[错误] {argv[i]} 缺少取值', file=sys.stderr)
                return 2
            gl_front += [argv[i], argv[i + 1]]
            i += 2
        elif any(argv[i].startswith(o + '=') for o in GLOBAL_OPTS):
            gl_front.append(argv[i])
            i += 1
        else:
            break
    argv = argv[i:]
    if not argv:
        print(__doc__)
        print('可用模块:', ', '.join(sorted(SCRIPTS)))
        return 1
    mod, rest = argv[0], argv[1:]

    if mod in ('selftest', 'test', 'all'):
        rc = 0
        for script, args in (('gpu_crack.py', ['bench']),
                             ('ctf_tools.py', ['selftest']),
                             ('ctf_ext.py', ['selftest']),
                             ('ctf_misc.py', ['selftest'])):
            g_ok, _ = filter_globals(gl_front, GLOBALS_OF[script])
            print(f'\n{"="*18} {script} {" ".join(args)} {"="*18}', flush=True)
            rc |= run(py, script, g_ok + args)
        return rc

    if mod in ('help', '-h', '--help'):
        print(__doc__)
        print('可用模块:', ', '.join(sorted(SCRIPTS)))
        return 0

    if mod not in SCRIPTS:
        print(f'[错误] 未知模块 {mod}；可用: {", ".join(sorted(SCRIPTS))}', file=sys.stderr)
        return 2
    gl, rest = split_globals(rest)
    g_ok, dropped = filter_globals(gl_front + gl, GLOBALS_OF[SCRIPTS[mod]])
    if dropped:
        print(f'[提示] {" ".join(dropped)} 对 {SCRIPTS[mod]} 无效，已忽略', file=sys.stderr)
    # 全局选项必须放在子命令之前
    return run(py, SCRIPTS[mod], g_ok + [mod] + rest)


if __name__ == '__main__':
    sys.exit(main())
