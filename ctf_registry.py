#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ctf_registry.py — 内核注册表（唯一真源）。

为什么要它：原先「模块名 → 脚本」的映射被抄在 4 个地方（skill/scripts/ctf.py、
doc_lint.py、README、CLI_REFERENCE），加一个内核要同时改 4 处，必然漂移。
现在所有内核元数据集中在这里，其余位置读它。

两种注册方式：

1. 内置内核（仓库自带，写在 BUILTIN 里）——
       Kernel('mtscan', 'gpu_crack.py', '按已知密钥流反查 MT19937 种子', group='批量爆破')

2. 插件内核（放 plugins/*.py，不碰本文件）——
       KERNEL = dict(name='hello', help='示例插件内核')      # 模块级元数据
       def main(argv): ...                                   # 可选：插件自处理参数
   插件脚本由统一入口以子进程方式调用，与内置内核行为一致。

插件元数据优先做**静态读取**（AST 解析模块级 `KERNEL = dict(...)` / 字面量 dict），
不执行插件顶层代码；仅当静态读取失败时才退回 `exec_module` 兜底（并告警），这样
`ctf.py help` / `--help` / `doc_lint` 不会因为插件顶层代码而产生副作用或变慢。

约定：插件名不得与内置内核重名，也不得占用调度器的保留命令名（selftest/help/...）；
重名时忽略并告警。
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tokenize
from typing import Dict, Iterator, List, Optional, Tuple

# 单一版本真源：README / CLI_REFERENCE / SKILL.md / kernels.md 的「文档版本」必须等于它。
# doc_lint.py 读这个常量做校验，其他位置不要再硬编码版本号。
VERSION = '3.5'

PLUGIN_DIRNAME = 'plugins'

# 调度器（skill/scripts/ctf.py）自己的命令名；插件不得占用，否则会被静默遮蔽。
RESERVED_NAMES = frozenset({'selftest', 'selfcheck', 'test', 'all', 'help'})


class Kernel:
    __slots__ = ('name', 'script', 'help', 'group', 'chunk', 'auto_json', 'builtin')

    def __init__(self, name: str, script: str, help: str, group: str = '',
                 chunk: bool = False, auto_json: bool = False, builtin: bool = True):
        self.name = name
        self.script = script
        self.help = help
        self.group = group
        self.chunk = chunk          # 是否支持 --chunk
        self.auto_json = auto_json  # 是否支持 --json
        self.builtin = builtin

    def __repr__(self) -> str:      # 便于调试
        return f'<Kernel {self.name} -> {self.script}>'


BL = '批量爆破'
CS = '密码学'
AL = '算法族'
FK = '取证 & Pwn'

BUILTIN: List[Kernel] = [
    # gpu_crack.py —— 支持 --chunk / --json
    Kernel('bench', 'gpu_crack.py', '吞吐基准（正式搜索前先跑）', BL, True, True),
    Kernel('mtscan', 'gpu_crack.py', '按已知密钥流反查 MT19937 种子', BL, True, True),
    Kernel('sboxscan', 'gpu_crack.py', '枚举 AES-128/192/256 S-box 破坏 (a,b)', BL, True, True),
    Kernel('ksscore', 'gpu_crack.py', '比特一致度打分选密钥流候选', BL, True, True),
    Kernel('lfsrscan', 'gpu_crack.py', 'LFSR 初始状态批量扫描', BL, True, True),
    # ctf_tools.py
    Kernel('chacha', 'ctf_tools.py', 'ChaCha20 密钥流/加解密/口令爆破', CS, True, True),
    Kernel('poly1305', 'ctf_tools.py', 'Poly1305 MAC 计算与校验', CS, False, True),
    Kernel('rs63', 'ctf_tools.py', 'RS(63,31) GF(256) 纠错译码', CS, False, True),
    Kernel('crc32', 'ctf_tools.py', 'CRC32 批量计算/校验', CS, False, True),
    Kernel('xorbrute', 'ctf_tools.py', '重复 XOR 与单字节 XOR 爆破', CS, True, True),
    Kernel('aesctr', 'ctf_tools.py', 'AES-128/192/256-CTR 密钥流/加解密', CS, False, True),
    # ctf_ext.py
    Kernel('tea', 'ctf_ext.py', 'TEA / XTEA / XXTEA 加解密', AL, False, True),
    Kernel('speck', 'ctf_ext.py', 'Speck32/64、Speck64/128 加解密', AL, False, True),
    Kernel('rc4', 'ctf_ext.py', 'RC4 加解密', AL, False, True),
    Kernel('lcg', 'ctf_ext.py', 'LCG 参数恢复与状态预测', AL, False, True),
    Kernel('md5', 'ctf_ext.py', 'MD5 哈希 / 口令爆破', AL, False, True),
    Kernel('rsa', 'ctf_ext.py', 'RSA 多攻击自动尝试', AL, False, True),
    Kernel('encode', 'ctf_ext.py', '多层编码 BFS 自动解码', AL, False, True),
    Kernel('mdext', 'ctf_ext.py', 'MD5 / SHA-1 哈希长度扩展', AL, False, True),
    Kernel('bm', 'ctf_ext.py', 'Berlekamp-Massey 反推 LFSR 多项式', AL, False, True),
    Kernel('zipbrute', 'ctf_ext.py', 'ZIP 口令爆破（字典/掩码）', AL, False, True),
    Kernel('eccscan', 'ctf_ext.py', 'ECC 小私钥区间搜索', AL, False, True),
    # ctf_misc.py —— 纯 CPU，不支持 chunk/json
    Kernel('lsb', 'ctf_misc.py', '图像 LSB 位平面穷举提取', FK, False, False),
    Kernel('stegscan', 'ctf_misc.py', '文件级隐写/附加数据扫描', FK, False, False),
    Kernel('imgdiff', 'ctf_misc.py', '两图差异包围盒与可视化', FK, False, False),
    Kernel('bitplane', 'ctf_misc.py', '单 bit 平面导出 PNG', FK, False, False),
    Kernel('cyclic', 'ctf_misc.py', 'pwntools 兼容 de Bruijn 模式生成', FK, False, False),
    Kernel('cycfind', 'ctf_misc.py', '溢出偏移定位', FK, False, False),
    Kernel('gadgets', 'ctf_misc.py', 'ROP gadget 搜索', FK, False, False),
    Kernel('elfinfo', 'ctf_misc.py', 'ELF 保护/导入/PLT/GOT 解析', FK, False, False),
]

# 三份「模块 → 脚本」派生视图，供不同调用方使用
SCRIPTS: Dict[str, str] = {k.name: k.script for k in BUILTIN}
BY_SCRIPT: Dict[str, List[str]] = {}
for _k in BUILTIN:
    BY_SCRIPT.setdefault(_k.script, []).append(_k.name)

# 脚本 → 支持的全局参数（统一入口 ctf.py 搬运参数时用；skill 目录单独部署时用它做内置回退表）。
# 说明：--chunk 由各内核能力位决定；--checkpoint/--reset-checkpoint/--no-progress 是
# add_sched_args() 提供的调度参数，ctf_ext.py 虽然没有 --chunk 但同样定义了这三个参数
# （历史接口，argparse 接受），因此调度参数按脚本单独登记，而不是从 --chunk 派生。
BACKEND_SCRIPTS: Tuple[str, ...] = ('gpu_crack.py', 'ctf_tools.py', 'ctf_ext.py')
SCHED_SCRIPTS: Tuple[str, ...] = ('gpu_crack.py', 'ctf_tools.py', 'ctf_ext.py')


def script_globals() -> Dict[str, Tuple[str, ...]]:
    """返回 {脚本: 支持的全局参数元组}，由 BUILTIN 的能力位派生（唯一真源）。"""
    out: Dict[str, List[str]] = {script: [] for script in BY_SCRIPT}
    for k in BUILTIN:
        g = out[k.script]
        if k.script in BACKEND_SCRIPTS and '--backend' not in g:
            g.append('--backend')
        if k.chunk and '--chunk' not in g:
            g.append('--chunk')
        if k.auto_json:
            for flag in ('--json', '--quiet'):
                if flag not in g:
                    g.append(flag)
        if k.script in SCHED_SCRIPTS:
            for flag in ('--checkpoint', '--reset-checkpoint', '--no-progress'):
                if flag not in g:
                    g.append(flag)
    return {script: tuple(g) for script, g in out.items()}


SCRIPT_GLOBALS: Dict[str, Tuple[str, ...]] = script_globals()


def functional_count() -> int:
    """功能子命令数（不含各模块的 selftest）。"""
    return sum(1 for k in BUILTIN if k.name != 'selftest')


def groups() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k in BUILTIN:
        out.setdefault(k.group or '其它', []).append(k.name)
    return out


# ---------------------------------------------------------------- 插件

class Plugin:
    __slots__ = ('name', 'path', 'help')

    def __init__(self, name: str, path: str, help: str):
        self.name, self.path, self.help = name, path, help

    def __repr__(self) -> str:
        return f'<Plugin {self.name} @ {self.path}>'


_MISSING = object()
# (path, mtime_ns, size) -> (Plugin | None, error_text)，避免同一进程内重复解析
_META_CACHE: Dict[Tuple[str, int, int], Tuple[Optional['Plugin'], str]] = {}


def _warn(msg: str) -> None:
    print(f'[warn] {msg}', file=sys.stderr)


def _literal_kernel(node: ast.AST) -> object:
    """尝试把模块级 KERNEL 的赋值静态求值成 dict；失败返回 _MISSING。"""
    if isinstance(node, ast.Dict):
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            return _MISSING
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        merged: Dict[str, object] = {}
        for arg in node.args:
            got = _literal_kernel(arg)
            if not isinstance(got, dict):
                return _MISSING
            merged.update(got)
        for kw in node.keywords:
            if kw.arg is None or not isinstance(kw.arg, str):
                return _MISSING
            try:
                merged[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError, TypeError):
                return _MISSING
        return merged
    # 其它字面量（字符串/数字/列表…）：能求值就返回，交给上层判定「不是合法 KERNEL dict」
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return _MISSING


def _read_meta_static(path: str) -> object:
    """不执行插件代码，静态解析模块级 `KERNEL = ...`；返回 dict 或 _MISSING。

    只接受顶层赋值；同一文件里出现多个 KERNEL 赋值时要求它们全部可静态求值，否则
    返回 _MISSING 交给 exec 兜底（避免读到被后续语句覆盖的旧值）。
    """
    with tokenize.open(path) as fh:          # 尊重 PEP 263 编码声明
        src = fh.read()
    tree = ast.parse(src, filename=path)
    found = _MISSING
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == 'KERNEL' for t in targets):
            continue
        got = _literal_kernel(value)
        if got is _MISSING:
            return _MISSING
        found = got
    return found


def _plugin_from_meta(meta: object, path: str) -> Optional[Plugin]:
    if not isinstance(meta, dict):
        return None
    name = meta.get('name')
    if not isinstance(name, str) or not name.strip():
        return None
    return Plugin(name.strip(), path, str(meta.get('help', '（插件未提供说明）')))


def _load_meta(path: str) -> Optional[Plugin]:
    """读取插件元数据：优先静态 AST 解析，失败才退回 exec_module 兜底。"""
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    if key in _META_CACHE:
        return _META_CACHE[key][0]

    base = os.path.basename(path)
    try:
        meta = _read_meta_static(path)
    except SyntaxError as exc:
        _warn(f'插件语法错误 {base}: {exc}')
        _META_CACHE[key] = (None, 'syntax')
        return None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        _warn(f'插件无法读取 {base}: {exc}')
        _META_CACHE[key] = (None, 'read')
        return None

    plug = _plugin_from_meta(meta, path) if meta is not _MISSING else None
    if plug is not None:
        _META_CACHE[key] = (plug, 'static')
        return plug
    if meta is not _MISSING:
        # 静态读到了 KERNEL，但不是合法的 dict/name —— 不必执行插件代码，直接忽略
        _warn(f'插件 {base} 的 KERNEL 必须是含非空字符串 name 的 dict，已忽略')
        _META_CACHE[key] = (None, 'badkernel')
        return None

    # ---- exec 兜底（只有静态读不出 KERNEL 时才执行插件顶层代码）----
    _warn(f'插件 {base} 的 KERNEL 不是可静态解析的字面量，退回 exec 读取（会执行顶层代码）')
    spec = importlib.util.spec_from_file_location(
        'ctf_plugin_' + os.path.splitext(base)[0] + '_' + str(st.st_mtime_ns), path)
    if spec is None or spec.loader is None:
        _META_CACHE[key] = (None, 'nospec')
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # 插件坏了不能拖垮主程序
        _warn(f'插件加载失败 {base}: {exc}')
        _META_CACHE[key] = (None, 'exec')
        return None
    plug = _plugin_from_meta(getattr(mod, 'KERNEL', None), path)
    if plug is None:
        _warn(f'插件 {base} 的 KERNEL 必须是含非空字符串 name 的 dict，已忽略')
        _META_CACHE[key] = (None, 'badkernel')
        return None
    _META_CACHE[key] = (plug, 'exec')
    return plug


def discover_plugins(root: str) -> List[Plugin]:
    """扫描 <root>/plugins/*.py，返回可用插件（跳过 _ 开头、内置/保留重名与坏插件）。"""
    pdir = os.path.join(root, PLUGIN_DIRNAME)
    if not os.path.isdir(pdir):
        return []
    found: List[Plugin] = []
    seen = set(SCRIPTS) | set(RESERVED_NAMES)
    for name in sorted(os.listdir(pdir)):
        if not name.endswith('.py') or name.startswith('_'):
            continue
        plug = _load_meta(os.path.join(pdir, name))
        if plug is None:
            continue
        if not _valid_plugin_name(plug.name):
            _warn(f'插件 {name} 的名字 {plug.name!r} 非法（不能含空白/以 - 开头），已忽略')
            continue
        if plug.name in seen:
            if plug.name in RESERVED_NAMES:
                _warn(f'插件 {name} 使用了调度器保留命令名 {plug.name!r}，已忽略')
            else:
                _warn(f'插件 {name} 与已有内核重名（{plug.name}），已忽略')
            continue
        seen.add(plug.name)
        found.append(plug)
    return found


def _valid_plugin_name(name: str) -> bool:
    return (bool(name) and name == name.strip() and not name.startswith('-')
            and not any(ch.isspace() for ch in name))


def all_names(root: Optional[str] = None) -> Iterator[str]:
    for name in SCRIPTS:
        yield name
    if root:
        for plug in discover_plugins(root):
            yield plug.name


# ---------------------------------------------------------------- 自检

def _selfcheck() -> int:
    """`python ctf_registry.py`：注册表/插件读取/静态元数据的安全与容错自检。"""
    import shutil
    import tempfile

    fails: List[str] = []

    def check(name: str, cond: bool, detail: object = '') -> None:
        print(('PASS ' if cond else 'FAIL ') + name + (('  |  ' + str(detail)[:300]) if detail and not cond else ''))
        if not cond:
            fails.append(name)

    # 1. 版本真源格式
    check('VERSION format', bool(VERSION) and VERSION[0].isdigit() and VERSION.count('.') == 1, VERSION)

    # 2. 内置内核视图自洽
    names = [k.name for k in BUILTIN]
    check('no duplicate kernel names', len(names) == len(set(names)), names)
    check('30 functional kernels', functional_count() == 30, functional_count())
    check('SCRIPTS covers all kernels', len(SCRIPTS) == len(BUILTIN), (len(SCRIPTS), len(BUILTIN)))
    check('no selftest in SCRIPTS', 'selftest' not in SCRIPTS, list(SCRIPTS)[:5])
    g = groups()
    flat = [n for v in g.values() for n in v]
    check('groups partition kernels', sorted(flat) == sorted(names) and len(g) == 4, (len(flat), len(g)))
    check('BY_SCRIPT partitions kernels', sorted(n for v in BY_SCRIPT.values() for n in v) == sorted(names), BY_SCRIPT)
    check('script_globals covers scripts', set(SCRIPT_GLOBALS) == set(BY_SCRIPT), set(SCRIPT_GLOBALS) ^ set(BY_SCRIPT))

    # 3. 全局参数派生正确性
    sg = SCRIPT_GLOBALS
    check('gpu_crack globals', sg.get('gpu_crack.py') == ('--backend', '--chunk', '--json', '--quiet',
                                                         '--checkpoint', '--reset-checkpoint', '--no-progress'),
          sg.get('gpu_crack.py'))
    check('ctf_misc has no globals', sg.get('ctf_misc.py') == (), sg.get('ctf_misc.py'))
    check('ctf_ext has no --chunk', '--chunk' not in sg.get('ctf_ext.py', ()), sg.get('ctf_ext.py'))
    check('every script except misc supports --json',
          all('--json' in v for s, v in sg.items() if s != 'ctf_misc.py'), sg)

    # 4. 真实插件目录：静态读取结果与真实仓库插件一致
    here = os.path.dirname(os.path.abspath(__file__))
    real = discover_plugins(here)
    real_names = [p.name for p in real]
    check('repo plugins discovered', 'morse' in real_names, real_names)

    # 5. 临时目录容错矩阵（不写仓库）
    tmp = tempfile.mkdtemp(prefix='ctf_registry_check_')
    try:
        pdir = os.path.join(tmp, PLUGIN_DIRNAME)
        os.makedirs(pdir)
        marker = os.path.join(tmp, 'side_effect.txt')

        def w(name: str, text: str) -> None:
            with open(os.path.join(pdir, name), 'w', encoding='utf-8') as fh:
                fh.write(text)

        w('sideeffect.py', f'open(r"{marker}", "a", encoding="utf-8").write("x")\n'
                           'KERNEL = dict(name="sideeffect", help="静态读取不应执行顶层代码")\n')
        w('bad_syntax.py', 'def broken(:\n    pass\n')
        w('bad_kernel.py', 'KERNEL = "not a dict"\n')
        w('_underscore.py', 'KERNEL = dict(name="hidden", help="应跳过")\n')
        w('dup_builtin.py', 'KERNEL = dict(name="mtscan", help="重名")\n')
        w('reserved.py', 'KERNEL = dict(name="selftest", help="保留名")\n')
        w('dynamic.py', 'KERNEL = dict(name="dynamic" + "_plug", help="需 exec 兜底")\n')
        plugs = discover_plugins(tmp)
        got = sorted(p.name for p in plugs)
        check('temp plugins: static read no side effect', not os.path.exists(marker),
              '插件顶层代码被执行（静态读取失效）')
        check('temp plugins: bad files skipped', 'bad_syntax' not in got and 'bad_kernel' not in got, got)
        check('temp plugins: underscore skipped', 'hidden' not in got, got)
        check('temp plugins: duplicate builtin skipped', 'mtscan' not in got, got)
        check('temp plugins: reserved name skipped', 'selftest' not in got, got)
        check('temp plugins: valid found', 'sideeffect' in got, got)
        check('temp plugins: dynamic exec fallback', 'dynamic_plug' in got, got)

        # 空目录 / 不存在的目录
        empty = os.path.join(tmp, 'empty_root')
        os.makedirs(os.path.join(empty, PLUGIN_DIRNAME))
        check('empty plugins dir', discover_plugins(empty) == [], discover_plugins(empty))
        check('missing plugins dir', discover_plugins(os.path.join(tmp, 'nope')) == [], '')

        # all_names 去重且包含内置 + 插件
        alln = list(all_names(tmp))
        check('all_names unique', len(alln) == len(set(alln)), len(alln))
        check('all_names includes builtin+plugin', 'mtscan' in alln and 'sideeffect' in alln, len(alln))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 6. 静态读取与 exec 读取对真实插件结果一致（接口不变）
    for p in real:
        try:
            exec_plug = _exec_meta_for_compare(p.path)
        except Exception:
            exec_plug = None
        same = exec_plug is not None and exec_plug.name == p.name and exec_plug.help == p.help
        check(f'plugin meta static==exec: {p.name}', same, (p.name, getattr(exec_plug, 'name', None)))

    print()
    print('REGISTRY:', 'ALL PASS' if not fails else f'{len(fails)} FAIL')
    return 0 if not fails else 1


def _exec_meta_for_compare(path: str) -> Optional[Plugin]:
    """自检用：强制走 exec 路径读一次元数据，与静态结果对比。"""
    spec = importlib.util.spec_from_file_location(
        'ctf_plugin_cmp_' + os.path.splitext(os.path.basename(path))[0]
        + '_' + str(abs(hash(path))), path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return _plugin_from_meta(getattr(mod, 'KERNEL', None), path)


if __name__ == '__main__':                          # 自检：python ctf_registry.py
    print(f'VERSION = {VERSION}')
    print(f'内置内核 {functional_count()} 个，分 {len(groups())} 组，覆盖 {len(BY_SCRIPT)} 个脚本')
    for g, names in groups().items():
        print(f'  {g}（{len(names)}）: {" ".join(names)}')
    here = os.path.dirname(os.path.abspath(__file__))
    plugs = discover_plugins(here)
    print(f'插件 {len(plugs)} 个: ' + (', '.join(p.name for p in plugs) or '（无）'))
    print()
    sys.exit(_selfcheck())
