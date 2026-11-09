#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf.py — ctf-toolkit 统一入口（规范性 CLI 见仓库根目录的 CLI_REFERENCE.md）

职责
----
1. 自动挑选解释器：CTF_PYTHON（若设置且存在）→ PATH 上的 python/python3（torch+CUDA）→ 当前解释器。
2. 把 <module> 映射到工具箱里的实际脚本（默认本仓库根目录，可用 CTF_TOOLKIT_DIR 覆盖）。
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
import re
import shutil
import subprocess
import sys

# ---- 输出编码 ----
# 被管道/重定向时（harness、Codex、subprocess.call）强制 UTF-8，保证调用方按 UTF-8 解码不会乱码；
# 交互式终端仍沿用系统默认编码（Windows 控制台 cp936 存中文更友好）。
def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and not stream.isatty():
                stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


_force_utf8_stdio()

# 仓库根目录 = 自动探测（见 _locate_toolkit）；脱离仓库单独部署时请设 CTF_TOOLKIT_DIR
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TOOLKIT = os.path.dirname(os.path.dirname(_HERE))
MARKER = 'gpu_crack.py'


def _locate_toolkit() -> str:
    """确定工具箱目录：CTF_TOOLKIT_DIR（有效时）-> 从本文件向上寻找 gpu_crack.py -> 向上两级。

    向上查找让 skill 目录嵌在仓库任意层级都能自洽（本地 skill/ 副本、skills/<name>/ 布局都适用）；
    仅当 skill 目录被单独拷到仓库之外时，才需要显式设置 CTF_TOOLKIT_DIR。
    """
    env = os.environ.get('CTF_TOOLKIT_DIR', '').strip()
    if env:
        if os.path.isfile(os.path.join(env, MARKER)):
            return os.path.abspath(env)
        print(f'[提示] CTF_TOOLKIT_DIR={env} 下没有 {MARKER}，忽略该变量并自动探测', file=sys.stderr)
    d = _HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, MARKER)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return _DEFAULT_TOOLKIT


TOOLKIT = _locate_toolkit()


def _toolkit_ready() -> bool:
    """工具箱目录里必须能找到 gpu_crack.py，否则一次性报清楚，而不是静默跑出四段空输出。"""
    if os.path.isfile(os.path.join(TOOLKIT, MARKER)):
        return True
    print(f'[错误] 工具箱目录 {TOOLKIT} 中找不到 {MARKER}；'
          f'请设置环境变量 CTF_TOOLKIT_DIR 指向仓库根目录', file=sys.stderr)
    return False

# module -> script
# 真源在仓库根的 ctf_registry.py；skill 目录被单独部署时（脱离仓库）回退到内置表。
_EMBEDDED_SCRIPTS = {
    'bench': 'gpu_crack.py', 'mtscan': 'gpu_crack.py', 'sboxscan': 'gpu_crack.py',
    'ksscore': 'gpu_crack.py', 'lfsrscan': 'gpu_crack.py',
    'chacha': 'ctf_tools.py', 'poly1305': 'ctf_tools.py', 'rs63': 'ctf_tools.py',
    'crc32': 'ctf_tools.py', 'xorbrute': 'ctf_tools.py', 'aesctr': 'ctf_tools.py',
    'tea': 'ctf_ext.py', 'speck': 'ctf_ext.py', 'rc4': 'ctf_ext.py', 'lcg': 'ctf_ext.py',
    'md5': 'ctf_ext.py', 'rsa': 'ctf_ext.py', 'encode': 'ctf_ext.py',
    'mdext': 'ctf_ext.py', 'bm': 'ctf_ext.py', 'zipbrute': 'ctf_ext.py',
    'eccscan': 'ctf_ext.py',
    'lsb': 'ctf_misc.py', 'stegscan': 'ctf_misc.py', 'imgdiff': 'ctf_misc.py',
    'bitplane': 'ctf_misc.py', 'cyclic': 'ctf_misc.py', 'cycfind': 'ctf_misc.py',
    'gadgets': 'ctf_misc.py', 'elfinfo': 'ctf_misc.py',
}

SCRIPTS = dict(_EMBEDDED_SCRIPTS)
PLUGINS = {}
GROUPS = {}
try:                                     # 插件/注册表只在本仓库内可用，部署副本会自动跳过
    if TOOLKIT not in sys.path:
        sys.path.insert(0, TOOLKIT)
    import ctf_registry as _REG          # noqa: E402

    SCRIPTS = dict(_REG.SCRIPTS)
    GROUPS = _REG.groups()
    PLUGINS = {p.name: p.path for p in _REG.discover_plugins(TOOLKIT)}
except Exception:                        # noqa: BLE001  —— 任何异常都退回内置表，保证可用
    GROUPS = {}

# 解释器探测结果缓存（按 CTF_PYTHON 取值区分，避免每轮重复拉起子进程）
_PICKED = {}


def _candidate_interpreters():
    """按优先级产出候选解释器路径，全部自动发现，不含任何硬编码本机路径。

    顺序：CTF_PYTHON -> 当前解释器 -> PATH 上的 python3/python
          -> Windows `py -0p` 登记的每个版本 -> 仓库内 .venv/venv
    """
    seen = set()

    def ok(p):
        if not p:
            return None
        try:
            p = os.path.abspath(p)
        except Exception:
            return None
        if p in seen or not os.path.isfile(p):
            return None
        seen.add(p)
        return p

    yield from filter(None, [ok(os.environ.get('CTF_PYTHON', ''))])

    for name in (sys.executable, shutil.which('python3'), shutil.which('python')):
        p = ok(name)
        if p:
            yield p

    # Windows：py 启动器登记的所有已安装版本（含各版本共存的场景）
    if os.name == 'nt':
        launcher = shutil.which('py')
        if launcher:
            try:
                r = subprocess.run([launcher, '-0p'], capture_output=True, text=True,
                                   timeout=20, encoding='utf-8', errors='replace')
                for line in (r.stdout or '').splitlines():
                    m = re.search(r'([A-Za-z]:\\.*?\.exe)', line)
                    if m:
                        p = ok(m.group(1))
                        if p:
                            yield p
            except Exception:
                pass

    # 仓库根下的虚拟环境（用户克隆后常直接建在这里）
    for rel in ('.venv/Scripts/python.exe', '.venv/bin/python',
                'venv/Scripts/python.exe', 'venv/bin/python'):
        p = ok(os.path.join(TOOLKIT, rel.replace('/', os.sep)))
        if p:
            yield p


def pick_python() -> str:
    """自动挑一个 import torch 且 CUDA 可用的解释器；都不行就返回当前解释器。

    显式设置了 CTF_PYTHON 时优先尊重（哪怕没有 CUDA，也能用 numpy 后端跑）。
    """
    key = os.environ.get('CTF_PYTHON', '')
    if key in _PICKED:
        return _PICKED[key]
    forced = key.strip()
    if forced and os.path.isfile(forced):
        # 显式指定即强制采纳：不问 CUDA，GPU 内核可退回 numpy 后端
        _PICKED[key] = os.path.abspath(forced)
        return _PICKED[key]
    for p in _candidate_interpreters():
        try:
            r = subprocess.run([p, '-c', 'import torch;assert torch.cuda.is_available()'],
                               capture_output=True, timeout=60)
            if r.returncode == 0:
                _PICKED[key] = p
                return p
        except Exception:
            continue
    _PICKED[key] = sys.executable
    return sys.executable


# 需要带值的全局选项（--opt VALUE 或 --opt=VALUE）
GLOBAL_VALUED = ('--backend', '--chunk', '--checkpoint')
# 开关型全局选项（不带值）
GLOBAL_FLAGS = ('--json', '--quiet', '--no-progress', '--reset-checkpoint')
GLOBAL_OPTS = GLOBAL_VALUED + GLOBAL_FLAGS

# 每个脚本支持的全局选项（ctf_misc.py 是纯 CPU 的 PIL/capstone 工具，没有这些参数）
GLOBALS_OF = {
    'gpu_crack.py': GLOBAL_OPTS,
    'ctf_tools.py': GLOBAL_OPTS,
    'ctf_ext.py': GLOBAL_OPTS,
    'ctf_misc.py': (),
}


def filter_globals(gl, allowed):
    """按目标脚本支持的选项过滤全局参数，返回 (保留, 丢弃)。开关型选项不吞下一个参数。"""
    keep, drop = [], []
    i = 0
    while i < len(gl):
        opt = gl[i].split('=', 1)[0]
        tgt = keep if opt in allowed else drop
        if opt in GLOBAL_FLAGS or '=' in gl[i] or i + 1 >= len(gl):
            tgt.append(gl[i]); i += 1
        else:
            tgt.extend(gl[i:i + 2]); i += 2
    return keep, drop



def split_globals(args):
    """
    把模块级全局参数（--backend / --chunk / --json / --quiet / ...）提到子命令之前。
    argparse 要求全局选项出现在子命令前面，否则子命令会报 unrecognized arguments。
    开关型选项（不带值）只搬它自己，避免吞掉后面的子命令名。
    """
    head, tail = [], []
    i = 0
    while i < len(args):
        opt = args[i].split('=', 1)[0]
        if opt in GLOBAL_FLAGS:
            head.append(args[i])
            i += 1
            continue
        if opt in GLOBAL_VALUED:
            head.append(args[i])
            if '=' not in args[i] and i + 1 < len(args):
                head.append(args[i + 1])
                i += 2
                continue
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
        _print_modules()
        return 1
    py = pick_python()

    # 允许 `ctf.py --backend numpy <module>` 这种写法：先吃掉最前面的全局选项
    gl_front, i = [], 0
    while i < len(argv):
        opt = argv[i].split('=', 1)[0]
        if opt in GLOBAL_FLAGS:                  # 开关型：只搬自己，别吞掉后面的模块名
            gl_front.append(argv[i])
            i += 1
        elif opt in GLOBAL_VALUED:
            if '=' in argv[i]:
                gl_front.append(argv[i])
                i += 1
            elif i + 1 >= len(argv):
                print(f'[错误] {argv[i]} 缺少取值', file=sys.stderr)
                return 2
            else:
                gl_front += [argv[i], argv[i + 1]]
                i += 2
        else:
            break
    argv = argv[i:]
    if not argv:
        print(__doc__)
        _print_modules()
        return 1
    mod, rest = argv[0], argv[1:]

    if mod in ('selftest', 'test', 'all'):
        if not _toolkit_ready():
            return 2
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
        _print_modules()
        return 0

    if mod not in SCRIPTS and mod not in PLUGINS:
        print(f'[错误] 未知模块 {mod}；可用: {", ".join(sorted(list(SCRIPTS) + list(PLUGINS)))}',
              file=sys.stderr)
        return 2
    if not _toolkit_ready():
        return 2
    if mod in PLUGINS:                       # 插件：整条命令交给插件脚本自己解析
        return run(py, PLUGINS[mod], rest)
    gl, rest = split_globals(rest)
    g_ok, dropped = filter_globals(gl_front + gl, GLOBALS_OF[SCRIPTS[mod]])
    if dropped:
        print(f'[提示] {" ".join(dropped)} 对 {SCRIPTS[mod]} 无效，已忽略', file=sys.stderr)
    # 全局选项必须放在子命令之前
    return run(py, SCRIPTS[mod], g_ok + [mod] + rest)


def _print_modules() -> None:
    """按分组打印内置内核，再单列插件（有注册表时分组，否则平铺）。"""
    if GROUPS:
        for group, names in GROUPS.items():
            print(f'  {group}（{len(names)}）: {" ".join(names)}')
    else:
        print('可用模块:', ', '.join(sorted(SCRIPTS)))
    if PLUGINS:
        print(f'  插件（{len(PLUGINS)}）: {" ".join(sorted(PLUGINS))}')


if __name__ == '__main__':
    sys.exit(main())
