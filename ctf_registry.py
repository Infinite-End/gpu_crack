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

约定：插件名不得与内置内核重名；重名时以内置为准并在 stderr 告警。
"""
from __future__ import annotations

import importlib.util
import os
from typing import Dict, Iterator, List, Optional

PLUGIN_DIRNAME = 'plugins'


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


def _load_meta(path: str) -> Optional[Plugin]:
    """把插件文件当模块 import 一次，读它的 KERNEL 字典。"""
    spec = importlib.util.spec_from_file_location(
        'ctf_plugin_' + os.path.splitext(os.path.basename(path))[0], path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # 插件坏了不能拖垮主程序
        print(f'[warn] 插件加载失败 {os.path.basename(path)}: {exc}', file=os.sys.stderr)
        return None
    meta = getattr(mod, 'KERNEL', None)
    if not isinstance(meta, dict) or 'name' not in meta:
        return None
    return Plugin(str(meta['name']), path, str(meta.get('help', '（插件未提供说明）')))


def discover_plugins(root: str) -> List[Plugin]:
    """扫描 <root>/plugins/*.py，返回可用插件（跳过 _ 开头与内置重名项）。"""
    pdir = os.path.join(root, PLUGIN_DIRNAME)
    if not os.path.isdir(pdir):
        return []
    found: List[Plugin] = []
    seen = set(SCRIPTS)
    for name in sorted(os.listdir(pdir)):
        if not name.endswith('.py') or name.startswith('_'):
            continue
        plug = _load_meta(os.path.join(pdir, name))
        if plug is None:
            continue
        if plug.name in seen:
            print(f'[warn] 插件 {name} 与已有内核重名（{plug.name}），已忽略',
                  file=os.sys.stderr)
            continue
        seen.add(plug.name)
        found.append(plug)
    return found


def all_names(root: Optional[str] = None) -> Iterator[str]:
    for name in SCRIPTS:
        yield name
    if root:
        for plug in discover_plugins(root):
            yield plug.name


if __name__ == '__main__':                          # 自检：python ctf_registry.py
    print(f'内置内核 {functional_count()} 个，分 {len(groups())} 组，覆盖 {len(BY_SCRIPT)} 个脚本')
    for g, names in groups().items():
        print(f'  {g}（{len(names)}）: {" ".join(names)}')
    here = os.path.dirname(os.path.abspath(__file__))
    plugs = discover_plugins(here)
    print(f'插件 {len(plugs)} 个: ' + (', '.join(p.name for p in plugs) or '（无）'))
