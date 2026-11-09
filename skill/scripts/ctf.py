#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf.py — ctf-toolkit 统一入口（规范性 CLI 见仓库根目录的 CLI_REFERENCE.md）

职责
----
1. 自动挑选解释器：CTF_PYTHON（若设置且存在）→ PATH 上的 python/python3（torch+CUDA）→ 当前解释器。
   探测结果带缓存（用户缓存目录 JSON，键含候选解释器路径 + mtime）；CTF_NO_PY_CACHE=1 可关闭，
   CTF_PY_CACHE=文件路径 可改缓存位置；CTF_DEBUG_TIMING=1 打印探测/缓存耗时。
2. 把 <module> 映射到工具箱里的实际脚本（默认本仓库根目录，可用 CTF_TOOLKIT_DIR 覆盖）。
3. 全局参数 --backend/--chunk 放在模块前后均可（内部自动挪到子命令之前）。
4. `selftest` 依次跑 gpu bench + ctf_tools/ctf_ext/ctf_misc 三个模块自测，退出码聚合；
   `selfcheck` 跑统一入口自身的单元自检（参数搬运/退出码/缓存/回退表，不需要 GPU）。

用法
----
    python ctf.py <module> [args...]        # module 即子命令名，如 rsa / mtscan / lsb
    python ctf.py <module> --help           # 查看该模块的参数
    python ctf.py help [module]             # 打印总帮助 / 指定模块的帮助
    python ctf.py selftest [全局参数...]    # 四模块组合自测（全 PASS 退出码 0）
    python ctf.py selfcheck                 # 统一入口自身自检（不跑内核）
    python ctf.py -- <module> [args...]     # 显式结束全局参数解析

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

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

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
# doc_lint.py 会校验这张回退表与注册表逐项一致，防止部署副本漂移。
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

# 每个脚本支持的全局参数（注册表缺失时的回退表；同样被 doc_lint 校验与注册表一致）
_EMBEDDED_GLOBALS = {
    'gpu_crack.py': ('--backend', '--chunk', '--json', '--quiet',
                     '--checkpoint', '--reset-checkpoint', '--no-progress'),
    'ctf_tools.py': ('--backend', '--chunk', '--json', '--quiet',
                     '--checkpoint', '--reset-checkpoint', '--no-progress'),
    'ctf_ext.py': ('--backend', '--json', '--quiet',
                   '--checkpoint', '--reset-checkpoint', '--no-progress'),
    'ctf_misc.py': (),
}

SCRIPTS = dict(_EMBEDDED_SCRIPTS)
GLOBALS_OF = dict(_EMBEDDED_GLOBALS)
PLUGINS = {}
GROUPS = {}
try:                                     # 插件/注册表只在本仓库内可用，部署副本会自动跳过
    if TOOLKIT not in sys.path:
        sys.path.insert(0, TOOLKIT)
    import ctf_registry as _REG          # noqa: E402

    SCRIPTS = dict(_REG.SCRIPTS)
    GLOBALS_OF = dict(_REG.SCRIPT_GLOBALS)
    GROUPS = _REG.groups()
    try:
        PLUGINS = {p.name: p.path for p in _REG.discover_plugins(TOOLKIT)}
    except Exception:                    # noqa: BLE001  —— 插件坏了不影响内置内核
        PLUGINS = {}
except Exception:                        # noqa: BLE001  —— 任何异常都退回内置表，保证可用
    GROUPS = {}

# 解释器探测结果缓存（进程内，按 CTF_PYTHON 取值区分）
_PICKED = {}

PROBE_CMD = 'import torch;assert torch.cuda.is_available()'
# 校验「这个可执行文件确实能当 Python 用」：要求子进程输出标记，避免 cmd.exe 之类
# 的壳程序把 -c 参数吞掉后返回 0 造成误判。
IS_PYTHON_MARKER = 'CTF_PY_PROBE_OK'
IS_PYTHON_CMD = f"import sys;sys.stdout.write('{IS_PYTHON_MARKER}')"
CACHE_VERSION = 2


def _debug(msg: str) -> None:
    if os.environ.get('CTF_DEBUG_TIMING', '').strip().lower() not in ('', '0', 'false', 'no', 'off'):
        print(msg, file=sys.stderr, flush=True)


def _env_on(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() not in ('', '0', 'false', 'no', 'off')


# ---------------------------------------------------------------- 解释器缓存

def cache_file() -> str:
    """解释器探测缓存文件路径：CTF_PY_CACHE（文件路径）> 用户缓存目录。"""
    override = os.environ.get('CTF_PY_CACHE', '').strip()
    if override:
        return os.path.abspath(override)
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.join(os.path.expanduser('~'), 'AppData', 'Local')
    else:
        base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'gpu_crack', 'ctf_python_cache.json')


def _exe_record(path: str):
    """解释器身份记录：路径 + mtime + size（任一变化都视为需要重新探测）。"""
    if not path:
        return None
    try:
        ap = os.path.abspath(path)
        st = os.stat(ap)
    except OSError:
        return None
    if not os.path.isfile(ap):
        return None
    return {'path': ap, 'mtime_ns': int(st.st_mtime_ns), 'size': int(st.st_size)}


def _record_matches(rec, stored) -> bool:
    if not isinstance(rec, dict) or not isinstance(stored, dict):
        return False
    return all(rec.get(k) == stored.get(k) for k in ('path', 'mtime_ns', 'size'))


def _cache_key(records, tag: str = 'cuda') -> str:
    blob = json.dumps({'version': CACHE_VERSION, 'probe': PROBE_CMD, 'tag': tag,
                       'candidates': [{'path': r['path'], 'mtime_ns': r['mtime_ns'],
                                       'size': r['size']} for r in records]},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _cache_read(path: str) -> dict:
    """读整个缓存结构 {'version','entries','aux'}；坏了/没权限就当空缓存。"""
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            if not isinstance(data.get('entries'), dict):
                data['entries'] = {}
            if not isinstance(data.get('aux'), dict):
                data['aux'] = {}
            return data
    except Exception:                                  # 缓存坏了/没权限都当作没有
        pass
    return {'version': CACHE_VERSION, 'entries': {}, 'aux': {}}


def _cache_load(path: str) -> dict:
    return _cache_read(path)['entries']


def _cache_write(path: str, data: dict) -> None:
    """同目录临时文件 + os.replace 原子替换；并发写最后者胜，绝不写坏 JSON。"""
    tmp = ''
    try:
        d = os.path.dirname(path) or '.'
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.ctf_py_cache.', suffix='.tmp', dir=d)
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(',', ':'))
        os.replace(tmp, path)
        tmp = ''
    except Exception:                                  # 缓存失败绝不影响主流程
        pass
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _cache_store(path: str, key: str, value: dict) -> None:
    """读-改-写：只更新 entries，保留 aux（py -0p 候选列表等）。"""
    data = _cache_read(path)
    data['version'] = CACHE_VERSION
    data['entries'][key] = value
    _cache_write(path, data)


def _aux_get(path: str, name: str):
    return _cache_read(path)['aux'].get(name)


def _aux_set(path: str, name: str, value) -> None:
    data = _cache_read(path)
    data['version'] = CACHE_VERSION
    data['aux'][name] = value
    _cache_write(path, data)


# 负缓存（所有候选都探测失败 → 回落当前解释器）的复检周期：环境可能后装了 CUDA，
# 正缓存由解释器身份（mtime/size）失效，负缓存额外用时间兜底。
_NEGATIVE_TTL = 6 * 3600
# Windows `py -0p` 候选列表的缓存时长（该命令约 0.2s；列表变化不频繁，1 小时足够新）。
_PY_LIST_TTL = 3600


def _pick_with_cache(records, probe, cache_path: str, use_cache: bool = True,
                     cache_tag: str = 'cuda'):
    """可注入 probe/cache 的核心选择逻辑，返回 (选中路径, info)。自检直接驱动它。

    info = {'cached': bool, 'probed': int, 'ok': bool}
    """
    key = _cache_key(records, cache_tag)
    cached = _cache_load(cache_path) if (use_cache and records) else {}
    ent = cached.get(key)
    if isinstance(ent, dict):
        chosen = ent.get('chosen')
        rec = _exe_record(chosen.get('path', '')) if isinstance(chosen, dict) else None
        if rec is not None and _record_matches(rec, chosen):
            fresh = bool(ent.get('probe_ok')) or (time.time() - float(ent.get('ts') or 0) < _NEGATIVE_TTL)
            if fresh:
                return rec['path'], {'cached': True, 'probed': 0, 'ok': bool(ent.get('probe_ok'))}
    chosen, ok, probed = None, False, 0
    for rec in records:
        probed += 1
        try:
            if probe(rec['path']):
                chosen, ok = rec, True
                break
        except Exception:
            continue
    if chosen is None:                                 # 全都不行：回落当前解释器
        cur = _exe_record(sys.executable)
        if cur is None and records:
            cur = records[0]
        chosen = cur
    if chosen is not None and use_cache:
        _cache_store(cache_path, key,
                     {'chosen': chosen, 'probe_ok': ok, 'ts': int(time.time())})
    return (chosen['path'] if chosen else sys.executable), {'cached': False, 'probed': probed, 'ok': ok}


def _py_registered_paths(use_cache: bool = True, cache_path: str = ''):
    """Windows `py -0p` 登记的 python.exe 列表；解析结果按启动器身份 + TTL 缓存。

    该子进程约 0.2s，而真正昂贵的 CUDA 探测是 1~3s，所以列表本身也值得缓存。
    """
    if os.name != 'nt':
        return []
    launcher = shutil.which('py')
    if not launcher:
        return []
    lrec = _exe_record(launcher)
    if use_cache and cache_path and lrec is not None:
        memo = _aux_get(cache_path, 'py_registered')
        if (isinstance(memo, dict) and _record_matches(lrec, memo.get('launcher') or {})
                and time.time() - float(memo.get('ts') or 0) < _PY_LIST_TTL):
            paths = memo.get('paths')
            if isinstance(paths, list):
                return [p for p in paths if isinstance(p, str)]
    paths = []
    try:
        r = subprocess.run([launcher, '-0p'], capture_output=True, text=True,
                           timeout=20, encoding='utf-8', errors='replace')
        for line in (r.stdout or '').splitlines():
            m = re.search(r'([A-Za-z]:\\.*?\.exe)', line)
            if m:
                paths.append(m.group(1))
    except Exception:
        pass
    if use_cache and cache_path and lrec is not None:
        _aux_set(cache_path, 'py_registered',
                 {'launcher': lrec, 'paths': paths, 'ts': int(time.time())})
    return paths


def _candidate_records(use_cache: bool = True, cache_path: str = ''):
    """按优先级产出候选解释器记录，全部自动发现，不含任何硬编码本机路径。

    顺序：当前解释器 -> PATH 上的 python3/python -> Windows `py -0p` 登记的每个版本
          -> 仓库内 .venv/venv（CTF_PYTHON 已在上层单独处理）。
    """
    recs, seen = [], set()

    def add(p):
        if not p:
            return
        rec = _exe_record(p)
        if rec is None:
            return
        key = rec['path'].lower() if os.name == 'nt' else rec['path']
        if key in seen:
            return
        seen.add(key)
        recs.append(rec)

    for name in (sys.executable, shutil.which('python3'), shutil.which('python')):
        add(name)

    # Windows：py 启动器登记的所有已安装版本（含各版本共存的场景）
    for p in _py_registered_paths(use_cache=use_cache, cache_path=cache_path):
        add(p)

    # 仓库根下的虚拟环境（用户克隆后常直接建在这里）
    for rel in ('.venv/Scripts/python.exe', '.venv/bin/python',
                'venv/Scripts/python.exe', 'venv/bin/python'):
        add(os.path.join(TOOLKIT, rel.replace('/', os.sep)))
    return recs


def _run_probe(py: str) -> bool:
    """真正的 CUDA 探测：`python -c "import torch;assert torch.cuda.is_available()"`。"""
    try:
        r = subprocess.run([py, '-c', PROBE_CMD], capture_output=True, timeout=90)
        return r.returncode == 0
    except Exception:
        return False


def _run_is_python(py: str) -> bool:
    """轻量校验：这个可执行文件能否以 Python 方式启动（用于 CTF_PYTHON 强制路径）。

    故意检查子进程 stdout 含标记：某些壳程序（cmd.exe 等）会把 -c 吞掉并返回 0，
    只按退出码判断会误采纳，导致后续整条命令静默跑空。
    """
    try:
        r = subprocess.run([py, '-c', IS_PYTHON_CMD], capture_output=True, timeout=30)
        return r.returncode == 0 and IS_PYTHON_MARKER.encode() in (r.stdout or b'')
    except Exception:
        return False


def pick_python() -> str:
    """自动挑一个 import torch 且 CUDA 可用的解释器；都不行就返回当前解释器。

    CTF_PYTHON 显式指定时优先采纳（不做 CUDA 检查，允许 numpy 后端）；若它根本不能以
    Python 启动则不硬顶，告警后回到自动探测，避免后续出现无法解释的失败。
    """
    env_key = os.environ.get('CTF_PYTHON', '')
    if env_key in _PICKED:
        return _PICKED[env_key]
    forced = env_key.strip()
    use_cache = not _env_on('CTF_NO_PY_CACHE')
    cpath = cache_file()
    t0 = time.time()
    if forced:
        rec = _exe_record(forced)
        if rec is None:
            print(f'[提示] CTF_PYTHON={forced} 不存在或不可读，忽略该变量并自动探测', file=sys.stderr)
        else:
            chosen, info = _pick_with_cache([rec], _run_is_python, cpath, use_cache=use_cache,
                                            cache_tag='is-python-v1')
            if info['ok'] and chosen == rec['path']:
                _PICKED[env_key] = rec['path']
                _debug(f'[py ] CTF_PYTHON 采纳 {rec["path"]}（缓存{"命中" if info["cached"] else "未命中"}，'
                       f'{time.time() - t0:.3f}s）')
                return rec['path']
            print(f'[提示] CTF_PYTHON={forced} 无法作为 Python 解释器启动，忽略并自动探测',
                  file=sys.stderr)
    records = _candidate_records(use_cache=use_cache, cache_path=cpath)
    chosen, info = _pick_with_cache(records, _run_probe, cpath, use_cache=use_cache)
    _PICKED[env_key] = chosen
    _debug(f'[py ] 自动探测选中 {chosen}（缓存{"命中" if info["cached"] else "未命中"}，'
           f'本次探测 {info["probed"]} 个候选，{time.time() - t0:.3f}s）')
    return chosen


# ---------------------------------------------------------------- 全局参数搬运

# 需要带值的全局选项（--opt VALUE 或 --opt=VALUE）
GLOBAL_VALUED = ('--backend', '--chunk', '--checkpoint')
# 开关型全局选项（不带值）
GLOBAL_FLAGS = ('--json', '--quiet', '--no-progress', '--reset-checkpoint')
GLOBAL_OPTS = GLOBAL_VALUED + GLOBAL_FLAGS


def _looks_like_option(tok: str) -> bool:
    """下一个 token 是否像选项（用于判断带值全局参数是否缺值）。负数视为取值。"""
    if not tok or tok == '-' or not tok.startswith('-'):
        return False
    return re.match(r'^-\d', tok) is None


def _take_front_globals(argv):
    """吃掉最前面的全局参数（模块名之前），返回 (globals, 剩余 argv, 错误列表)。"""
    head, i, errs = [], 0, []
    while i < len(argv):
        tok = argv[i]
        if tok == '--':            # 显式结束全局参数解析：丢掉标记，后面按模块处理
            i += 1
            break
        opt = tok.split('=', 1)[0]
        if opt in GLOBAL_FLAGS:
            head.append(tok)
            i += 1
            continue
        if opt in GLOBAL_VALUED:
            if '=' in tok:
                head.append(tok)
                i += 1
            elif i + 1 < len(argv) and not _looks_like_option(argv[i + 1]):
                head.extend([tok, argv[i + 1]])
                i += 2
            else:
                errs.append(f'{tok} 缺少取值')
                i += 1
            continue
        break
    return head, argv[i:], errs


def filter_globals(gl, allowed):
    """按目标脚本支持的选项过滤全局参数，返回 (保留, 丢弃)。开关型选项不吞下一个参数。"""
    keep, drop = [], []
    i = 0
    while i < len(gl):
        tok = gl[i]
        opt = tok.split('=', 1)[0]
        tgt = keep if opt in allowed else drop
        if opt in GLOBAL_FLAGS or '=' in tok or i + 1 >= len(gl):
            tgt.append(tok)
            i += 1
        else:
            tgt.extend(gl[i:i + 2])
            i += 2
    return keep, drop


def split_globals(args):
    """把模块名之后的全局参数提到子命令之前，返回 (globals, rest, errors)。

    argparse 要求全局选项出现在子命令前面，否则子命令会报 unrecognized arguments。
    - 开关型选项（不带值）只搬它自己，避免吞掉后面的参数；
    - 带值选项缺值时记入 errors（由 main 统一报错），不会把下一个选项当值；
    - `--` 之后一律原样保留（不再搬运），语义与常规 CLI 的 `--` 一致。
    """
    head, tail, errs = [], [], []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == '--':
            tail.extend(args[i:])
            break
        opt = tok.split('=', 1)[0]
        if opt in GLOBAL_FLAGS:
            head.append(tok)
            i += 1
            continue
        if opt in GLOBAL_VALUED:
            if '=' in tok:
                head.append(tok)
                i += 1
            elif i + 1 < len(args) and not _looks_like_option(args[i + 1]):
                head.extend([tok, args[i + 1]])
                i += 2
            else:
                errs.append(f'{tok} 缺少取值')
                i += 1
            continue
        tail.append(tok)
        i += 1
    return head, tail, errs


# ---------------------------------------------------------------- 执行

def run(py: str, script: str, args: list) -> int:
    path = os.path.join(TOOLKIT, script)
    if not os.path.exists(path):
        print(f'[错误] 找不到 {path}；用 CTF_TOOLKIT_DIR 指定 toolkit 目录', file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    cmd = [py, '-X', 'utf8', path] + args
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, env=env)
    except OSError as exc:
        print(f'[错误] 无法启动解释器 {py}：{exc}', file=sys.stderr)
        return 2
    _debug(f'[run] {os.path.basename(script)} rc={rc} {time.time() - t0:.2f}s')
    return rc


def _print_usage() -> None:
    print(__doc__)
    _print_modules()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _print_usage()
        return 1

    gl_front, rest, errs = _take_front_globals(argv)
    if errs:
        for e in errs:
            print(f'[错误] {e}', file=sys.stderr)
        return 2
    if not rest:
        print('[提示] 缺少模块名，下面打印用法与模块列表', file=sys.stderr)
        _print_usage()
        return 1
    mod, rest = rest[0], rest[1:]

    # ---- 统一入口自身的命令 ----
    if mod in ('selftest', 'test', 'all'):
        return _cmd_selftest(gl_front, rest)
    if mod == 'selfcheck':
        return _selfcheck()
    if mod in ('help', '-h', '--help'):
        return _cmd_help(rest)

    if mod not in SCRIPTS and mod not in PLUGINS:
        if mod.startswith('-'):
            print(f'[错误] 未知全局参数 {mod}；全局参数: {" ".join(GLOBAL_OPTS)}', file=sys.stderr)
        else:
            print(f'[错误] 未知模块 {mod}；可用: '
                  f'{", ".join(sorted(list(SCRIPTS) + list(PLUGINS)))}', file=sys.stderr)
        return 2
    if not _toolkit_ready():
        return 2

    py = pick_python()
    if mod in PLUGINS:                       # 插件：插件自己解析参数，全局参数不适用
        if gl_front:
            print(f'[提示] {" ".join(gl_front)} 对插件 {mod} 无效，已忽略', file=sys.stderr)
        return run(py, PLUGINS[mod], rest)
    gl, rest2, errs = split_globals(rest)
    if errs:
        for e in errs:
            print(f'[错误] {e}', file=sys.stderr)
        return 2
    g_ok, dropped = filter_globals(gl_front + gl, GLOBALS_OF.get(SCRIPTS[mod], ()))
    if dropped:
        print(f'[提示] {" ".join(dropped)} 对 {SCRIPTS[mod]} 无效，已忽略', file=sys.stderr)
    # 全局选项必须放在子命令之前
    return run(py, SCRIPTS[mod], g_ok + [mod] + rest2)


def _cmd_help(rest) -> int:
    """`ctf.py help` 打印总帮助；`ctf.py help <module>` 转发到该模块的 --help。"""
    if not rest:
        _print_usage()
        return 0
    target = rest[0]
    if target in ('selftest', 'selfcheck'):
        _print_usage()
        return 0
    if target in PLUGINS:
        if not _toolkit_ready():
            return 2
        return run(pick_python(), PLUGINS[target], ['--help'])
    if target in SCRIPTS:
        if not _toolkit_ready():
            return 2
        return run(pick_python(), SCRIPTS[target], [target, '--help'])
    print(f'[错误] 未知模块 {target}；可用: '
          f'{", ".join(sorted(list(SCRIPTS) + list(PLUGINS)))}', file=sys.stderr)
    return 2


def _cmd_selftest(gl_front, rest) -> int:
    """四模块组合自测：gpu bench + 3×selftest，退出码按位 OR 聚合。"""
    gl_tail, extra, errs = split_globals(rest)
    if errs:
        for e in errs:
            print(f'[错误] {e}', file=sys.stderr)
        return 2
    if extra:
        print(f'[错误] selftest 不支持模块参数：{" ".join(extra)}', file=sys.stderr)
        return 2
    if not _toolkit_ready():
        return 2
    py = pick_python()
    gl = gl_front + gl_tail
    rc = 0
    for script, args in (('gpu_crack.py', ['bench']),
                         ('ctf_tools.py', ['selftest']),
                         ('ctf_ext.py', ['selftest']),
                         ('ctf_misc.py', ['selftest'])):
        g_ok, _dropped = filter_globals(gl, GLOBALS_OF.get(script, ()))
        print(f'\n{"=" * 18} {script} {" ".join(args)} {"=" * 18}', flush=True)
        rc |= run(py, script, g_ok + args)
    return rc


def _selfcheck() -> int:
    """统一入口自检：参数搬运/退出码/解释器缓存/回退表，全部无 GPU 依赖。"""
    import contextlib
    import io
    import threading

    global TOOLKIT, SCRIPTS, GLOBALS_OF, PLUGINS
    fails = []

    def check(name, cond, detail=''):
        print(('PASS ' if cond else 'FAIL ') + name
              + (('  |  ' + str(detail)[:300]) if detail and not cond else ''))
        if not cond:
            fails.append(name)

    # ---- 1. 纯函数：split_globals ----
    cases = [
        (['--backend', 'numpy', '--json'], ['--backend', 'numpy', '--json'], []),
        (['--backend=numpy', 'xorbrute'], ['--backend=numpy'], ['xorbrute']),
        (['mtscan', '--key-hex', 'ff', '--backend', 'numpy'],
         ['--backend', 'numpy'], ['mtscan', '--key-hex', 'ff']),
        (['--json', '--quiet'], ['--json', '--quiet'], []),
        (['-h'], [], ['-h']),
        (['--', '--json'], [], ['--', '--json']),
    ]
    ok = True
    detail = ''
    for src, want_head, want_tail in cases:
        head, tail, errs = split_globals(list(src))
        if head != want_head or tail != want_tail or errs:
            ok = False
            detail = (src, head, tail, errs)
            break
    check('split_globals: value/flag/-- boundary', ok, detail)

    head, tail, errs = split_globals(['--backend'])
    check('split_globals: missing value reported', not head and errs, (head, tail, errs))
    head, tail, errs = split_globals(['--backend', '--json'])
    check('split_globals: option-looking value not swallowed', head == ['--json'] and errs,
          (head, tail, errs))

    # ---- 2. 纯函数：filter_globals ----
    keep, drop = filter_globals(['--backend', 'numpy', '--json'],
                                GLOBAL_OPTS)
    check('filter_globals: all supported kept', keep == ['--backend', 'numpy', '--json'] and not drop,
          (keep, drop))
    keep, drop = filter_globals(['--json', '--backend', 'numpy'], ('--json',))
    check('filter_globals: unsupported valued dropped with its value',
          keep == ['--json'] and drop == ['--backend', 'numpy'], (keep, drop))
    keep, drop = filter_globals(['--json', '--quiet'], ())
    check('filter_globals: flags dropped singly', not keep and drop == ['--json', '--quiet'],
          (keep, drop))

    # ---- 3. 前置全局参数 ----
    head, rest2, errs = _take_front_globals(['--backend', 'numpy', 'rsa', '--n', '1'])
    check('front globals: module separated', head == ['--backend', 'numpy']
          and rest2 == ['rsa', '--n', '1'] and not errs, (head, rest2, errs))
    head, rest2, errs = _take_front_globals(['--', 'rsa', '--json'])
    check('front globals: -- terminator', not head and rest2 == ['rsa', '--json'] and not errs,
          (head, rest2, errs))
    head, rest2, errs = _take_front_globals(['--backend', '--json', 'rsa'])
    check('front globals: missing value reported', bool(errs), (head, rest2, errs))

    # ---- 4. 部署回退表与注册表一致 ----
    check('embedded SCRIPTS == SCRIPTS', _EMBEDDED_SCRIPTS == SCRIPTS,
          set(_EMBEDDED_SCRIPTS.items()) ^ set(SCRIPTS.items()))
    check('embedded GLOBALS == GLOBALS_OF',
          {k: tuple(v) for k, v in _EMBEDDED_GLOBALS.items()}
          == {k: tuple(v) for k, v in GLOBALS_OF.items()},
          (list(_EMBEDDED_GLOBALS), list(GLOBALS_OF)))
    check('GLOBALS_OF covers SCRIPTS', set(GLOBALS_OF) >= set(SCRIPTS.values()), GLOBALS_OF)
    check('globals subset of GLOBAL_OPTS',
          all(set(v) <= set(GLOBAL_OPTS) for v in GLOBALS_OF.values()), GLOBALS_OF)
    check('ctf_ext has no --chunk', '--chunk' not in GLOBALS_OF.get('ctf_ext.py', ()),
          GLOBALS_OF.get('ctf_ext.py'))
    check('ctf_misc has no globals', not GLOBALS_OF.get('ctf_misc.py', ()),
          GLOBALS_OF.get('ctf_misc.py'))

    # ---- 5. 解释器缓存（注入 fake probe，不真的探测 torch）----
    tmp = tempfile.mkdtemp(prefix='ctf_dispatcher_check_')
    try:
        fake1 = os.path.join(tmp, 'py_a.exe')
        fake2 = os.path.join(tmp, 'py_b.exe')
        for p in (fake1, fake2):
            with open(p, 'w', encoding='utf-8') as fh:
                fh.write('x')
        cpath = os.path.join(tmp, 'cache', 'pycache.json')
        records = [_exe_record(fake1), _exe_record(fake2)]
        calls = {'n': 0, 'paths': []}

        def fake_probe(path):
            calls['n'] += 1
            calls['paths'].append(path)
            return path == records[1]['path']

        chosen, info = _pick_with_cache(records, fake_probe, cpath)
        check('cache: first call probes', chosen == records[1]['path'] and not info['cached']
              and calls['n'] == 2, (chosen, info))
        check('cache: file written valid json', _json_ok(cpath), cpath)
        calls['n'] = 0
        chosen, info = _pick_with_cache(records, fake_probe, cpath)
        check('cache: second call hits without probe', chosen == records[1]['path']
              and info['cached'] and calls['n'] == 0, (chosen, info, calls['n']))
        # 解释器 mtime 变化 -> 身份失效 -> 重新探测
        os.utime(records[1]['path'], None)
        time.sleep(0.01)
        os.utime(records[1]['path'], None)
        calls['n'] = 0
        chosen, info = _pick_with_cache(records, fake_probe, cpath)
        check('cache: interpreter identity change re-probes',
              not info['cached'] and calls['n'] >= 1, (info, calls['n']))
        # CTF_NO_PY_CACHE 等价于 use_cache=False
        calls['n'] = 0
        chosen, info = _pick_with_cache(records, fake_probe, cpath, use_cache=False)
        check('cache: disabled still probes', calls['n'] == 2 and not info['cached'], calls['n'])
        # 损坏缓存被容错
        with open(cpath, 'w', encoding='utf-8') as fh:
            fh.write('{broken json')
        calls['n'] = 0
        chosen, info = _pick_with_cache(records, fake_probe, cpath)
        check('cache: corrupted file tolerated', chosen == records[1]['path'] and calls['n'] == 2,
              (chosen, calls['n']))
        # 并发写：多线程同时写同一缓存，最终必须是合法 JSON
        errs2 = []

        def writer(i):
            try:
                _cache_store(cpath, f'k{i}', {'chosen': records[0], 'probe_ok': True, 'ts': i})
            except Exception as exc:            # noqa: BLE001
                errs2.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        entries = _cache_load(cpath)
        check('cache: concurrent writers leave valid json', not errs2 and len(entries) >= 1,
              (errs2, len(entries)))
        # aux（py -0p 候选列表）读改写保留
        _aux_set(cpath, 'py_registered', {'paths': ['C:/fake/python.exe'], 'ts': 1})
        _cache_store(cpath, 'extra_key', {'chosen': records[0], 'probe_ok': True, 'ts': 2})
        aux = _aux_get(cpath, 'py_registered')
        check('cache: aux survives entries write',
              isinstance(aux, dict) and aux.get('paths') == ['C:/fake/python.exe']
              and _json_ok(cpath), aux)
        # 不同 probe 语义的缓存键必须隔离（同一候选 + 不同 cache_tag 不能互相命中）
        calls['n'] = 0
        chosen, info = _pick_with_cache(records, fake_probe, cpath, cache_tag='other-probe')
        check('cache: cache_tag isolates different probe semantics',
              not info['cached'] and calls['n'] == 2, (info, calls['n']))
        # 缓存写入失败不应影响结果（父路径是文件）
        blocker = os.path.join(tmp, 'blockfile')
        with open(blocker, 'w', encoding='utf-8') as fh:
            fh.write('x')
        chosen, info = _pick_with_cache(records, fake_probe, os.path.join(blocker, 'c.json'))
        check('cache: unwritable cache path tolerated', bool(chosen), chosen)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 6. 退出码 / 参数搬运端到端（临时 toolkit + 假模块，不依赖 GPU）----
    tk = tempfile.mkdtemp(prefix='ctf_dispatch_tk_')
    saved = {'TOOLKIT': TOOLKIT, 'SCRIPTS': SCRIPTS, 'GLOBALS_OF': GLOBALS_OF,
             'PLUGINS': PLUGINS, 'ARGV': sys.argv, 'PICKED': dict(_PICKED),
             'ENV': {k: os.environ.get(k) for k in ('CTF_PYTHON', 'CTF_PY_CACHE',
                                                    'CTF_NO_PY_CACHE', 'PYTHONIOENCODING')}}
    try:
        argv_log = os.path.join(tk, 'argv.log')
        # 假模块：把收到的 argv 追加写入 FAKE_ARGV_FILE，并按 FAKE_<SCRIPT> 环境变量退出，
        # 这样自检能同时验证「参数搬运结果」和「退出码聚合」，不依赖真实内核/GPU。
        fake_src = (
            'import json, os, sys\n'
            'log = os.environ.get("FAKE_ARGV_FILE")\n'
            'if log:\n'
            '    with open(log, "a", encoding="utf-8") as fh:\n'
            '        fh.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")\n'
            'key = "FAKE_" + os.path.basename(os.path.abspath(sys.argv[0]))[:-3].upper().replace(".", "_")\n'
            'sys.exit(int(os.environ.get(key, "0")))\n'
        )
        for s in ('gpu_crack.py', 'ctf_tools.py', 'ctf_ext.py', 'ctf_misc.py'):
            with open(os.path.join(tk, s), 'w', encoding='utf-8') as fh:
                fh.write(fake_src)
        os.makedirs(os.path.join(tk, 'plugins'))
        with open(os.path.join(tk, 'plugins', 'okplug.py'), 'w', encoding='utf-8') as fh:
            fh.write('import sys\nKERNEL = dict(name="okplug", help="selfcheck")\n'
                     'if __name__ == "__main__":\n'
                     '    sys.exit(2 if "--fail" in sys.argv else 0)\n')
        with open(os.path.join(tk, 'plugins', 'broken.py'), 'w', encoding='utf-8') as fh:
            fh.write('def broken(:\n')
        if '_REG' in globals():
            warn_buf = io.StringIO()
            with contextlib.redirect_stderr(warn_buf):
                PLUGINS = {p.name: p.path for p in _REG.discover_plugins(tk)}   # noqa: F821
            check('plugin discovery tolerates broken plugin',
                  'broken.py' in warn_buf.getvalue() and 'okplug' in PLUGINS,
                  (warn_buf.getvalue(), list(PLUGINS)))
        else:
            PLUGINS = {'okplug': os.path.join(tk, 'plugins', 'okplug.py')}
        TOOLKIT = tk
        os.environ['CTF_PYTHON'] = sys.executable
        os.environ['CTF_PY_CACHE'] = os.path.join(tk, 'pycache.json')
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        def run_main(args):
            out, err = io.StringIO(), io.StringIO()
            old_argv = sys.argv
            sys.argv = ['ctf.py'] + args
            os.environ['FAKE_ARGV_FILE'] = argv_log
            if os.path.exists(argv_log):
                os.remove(argv_log)
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = main()
            finally:
                sys.argv = old_argv
            calls = []
            if os.path.exists(argv_log):
                with open(argv_log, encoding='utf-8') as fh:
                    calls = [json.loads(line) for line in fh if line.strip()]
            return rc, out.getvalue(), err.getvalue(), calls

        rc, out, err, calls = run_main([])
        check('main: no args -> rc 1 + help', rc == 1 and '退出码' in out, rc)
        rc, out, err, calls = run_main(['nonsense'])
        check('main: unknown module -> rc 2', rc == 2 and '未知模块' in err, (rc, err[:120]))
        rc, out, err, calls = run_main(['--frobnicate', 'bench'])
        check('main: unknown global -> rc 2', rc == 2 and '未知全局参数' in err, (rc, err[:120]))
        rc, out, err, calls = run_main(['bench'])
        check('main: module dispatch rc0 + argv', rc == 0 and calls == [['bench']], (rc, calls))
        # 全局参数前后位置等价
        rc1, _o1, _e1, calls1 = run_main(['mtscan', '--key-hex', 'ff', '--json'])
        rc2, _o2, _e2, calls2 = run_main(['--json', 'mtscan', '--key-hex', 'ff'])
        check('main: globals before/after module equivalent',
              rc1 == rc2 == 0 and calls1 == calls2 == [['--json', 'mtscan', '--key-hex', 'ff']],
              (calls1, calls2))
        # ctf_ext 不支持 --chunk -> 丢弃并提示，不能把 8192 当子命令
        rc, out, err, calls = run_main(['rsa', '--n', '3', '--chunk', '8192'])
        check('main: unsupported --chunk dropped for ctf_ext',
              rc == 0 and '无效' in err and calls == [['rsa', '--n', '3']], (rc, calls, err[:160]))
        # 缺值全局参数（模块之后）必须报错，而不是把模块名当值
        rc, out, err, calls = run_main(['xorbrute', '--data-hex', 'ff', '--backend'])
        check('main: missing global value after module -> rc2',
              rc == 2 and '缺少取值' in err and not calls, (rc, err[:160]))
        rc, out, err, calls = run_main(['--backend'])
        check('main: only globals -> rc2', rc == 2 and not calls, rc)
        # -- 分隔符
        rc, out, err, calls = run_main(['--', 'bench', '--json'])
        check('main: leading -- works', rc == 0 and calls == [['--json', 'bench']], (rc, calls))
        # help [module]
        rc, out, err, calls = run_main(['help'])
        check('main: help rc0', rc == 0 and '批量爆破' in out, rc)
        rc, out, err, calls = run_main(['help', 'mtscan'])
        check('main: help <module> rc0', rc == 0 and calls == [['mtscan', '--help']], (rc, calls))
        rc, out, err, calls = run_main(['help', 'nope'])
        check('main: help unknown -> rc2', rc == 2, rc)
        # 插件：rc 透传 + 全局参数被忽略
        rc, out, err, calls = run_main(['okplug', '--fail'])
        check('main: plugin rc passthrough', rc == 2, rc)
        rc, out, err, calls = run_main(['--json', 'okplug'])
        check('main: plugin front globals warned+ignored', rc == 0 and '已忽略' in err, (rc, err[:120]))
        # selftest：四模块 OR 聚合 + 尾随全局参数
        def fake_rc(rcs, extra_args=()):
            for script, code in zip(('gpu_crack.py', 'ctf_tools.py', 'ctf_ext.py', 'ctf_misc.py'), rcs):
                os.environ['FAKE_' + os.path.basename(script)[:-3].upper().replace('.', '_')] = str(code)
            try:
                return run_main(['selftest'] + list(extra_args))
            finally:
                for script in ('gpu_crack.py', 'ctf_tools.py', 'ctf_ext.py', 'ctf_misc.py'):
                    os.environ.pop('FAKE_' + os.path.basename(script)[:-3].upper().replace('.', '_'), None)

        for rcs, expected in (([0, 0, 0, 0], 0), ([1, 0, 0, 0], 1), ([0, 2, 0, 0], 2),
                              ([0, 0, 4, 0], 4), ([2, 1, 0, 0], 3), ([0, 0, 0, 1], 1)):
            rc, out, err, calls = fake_rc(rcs)
            check(f'main: selftest OR {rcs} -> {expected}', rc == expected, rc)
        rc, out, err, calls = fake_rc([0, 0, 0, 0], ['--backend', 'numpy'])
        check('main: selftest accepts trailing globals',
              rc == 0 and ['--backend', 'numpy', 'selftest'] in calls, (rc, calls))
        rc, out, err, calls = fake_rc([0, 0, 0, 0], ['garbage'])
        check('main: selftest rejects module args', rc == 2 and '不支持' in err and not calls,
              (rc, err[:120], calls))
        # run() 找不到脚本
        rc = run(sys.executable, 'no_such_script.py', [])
        check('run: missing script -> rc2', rc == 2, rc)
    finally:
        TOOLKIT = saved['TOOLKIT']
        SCRIPTS = saved['SCRIPTS']
        GLOBALS_OF = saved['GLOBALS_OF']
        PLUGINS = saved['PLUGINS']
        sys.argv = saved['ARGV']
        _PICKED.clear()
        _PICKED.update(saved['PICKED'])
        for k, v in saved['ENV'].items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tk, ignore_errors=True)

    # ---- 7. 工具箱目录解析 ----
    tmp = tempfile.mkdtemp(prefix='ctf_locate_check_')
    try:
        skill_scripts = os.path.join(tmp, 'repo', 'skill', 'scripts')
        os.makedirs(skill_scripts)
        with open(os.path.join(tmp, 'repo', MARKER), 'w', encoding='utf-8') as fh:
            fh.write('')
        saved_here = globals()['_HERE']
        saved_env = os.environ.get('CTF_TOOLKIT_DIR')
        try:
            globals()['_HERE'] = skill_scripts
            os.environ.pop('CTF_TOOLKIT_DIR', None)
            got = _locate_toolkit()
            check('locate: upward search finds repo root',
                  os.path.abspath(got) == os.path.abspath(os.path.join(tmp, 'repo')), got)
            os.environ['CTF_TOOLKIT_DIR'] = os.path.join(tmp, 'repo')
            check('locate: CTF_TOOLKIT_DIR wins',
                  os.path.abspath(_locate_toolkit()) == os.path.abspath(os.path.join(tmp, 'repo')),
                  _locate_toolkit())
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                os.environ['CTF_TOOLKIT_DIR'] = os.path.join(tmp, 'bogus')
                got = _locate_toolkit()
            check('locate: invalid env hinted + ignored',
                  'CTF_TOOLKIT_DIR' in err.getvalue()
                  and os.path.abspath(got) == os.path.abspath(os.path.join(tmp, 'repo')), got)
        finally:
            globals()['_HERE'] = saved_here
            if saved_env is None:
                os.environ.pop('CTF_TOOLKIT_DIR', None)
            else:
                os.environ['CTF_TOOLKIT_DIR'] = saved_env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print('DISPATCHER:', 'ALL PASS' if not fails else f'{len(fails)} FAIL')
    return 0 if not fails else 1


def _json_ok(path: str) -> bool:
    try:
        with open(path, encoding='utf-8') as fh:
            return isinstance(json.load(fh), dict)
    except Exception:
        return False


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
