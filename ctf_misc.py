#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf_misc.py — 取证（图片/文件层）与 Pwn 辅助工具集（纯 CPU，无 --backend/--chunk）

子命令
------
  图片 / 取证
  lsb         LSB 位平面提取（通道模板/位深/行列序/像素交织），按 magic/flag/可打印率打分
  stegscan    文件级隐写扫描：尾附加数据、21 条内嵌 magic、PNG chunk 异常、zlib 流、可疑字符串
  imgdiff     两图差异（XOR/相减），输出包围盒、LSB 差异与可视化图
  bitplane    导出单个位平面为 PNG（--plane 0..7 --channel 0..2 --out 文件路径）

  Pwn 辅助
  cyclic      生成 pwntools 兼容的 de Bruijn 循环模式（小写字母，子序列 n=4）
  cycfind     在数据中定位循环模式的偏移
  gadgets     ROP gadget 搜索（capstone；ret/retf/ret imm16/syscall 结尾）
  elfinfo     ELF 概览：架构/位数/入口/NX/PIE/RELRO/PLT/GOT/危险导入
  selftest    自测（全 PASS 退出码 0）

退出码
------
  0  成功（没找到/无可疑数据也是 0，看输出）
  1  运行期错误（文件不存在、图像非法、非 ELF、参数越界等；stderr 一行 "[cmd] 错误: ..."，无 traceback）

用法
----
  python ctf_misc.py selftest
  python ctf_misc.py lsb      --img a.png
  python ctf_misc.py stegscan --file a.png
  python ctf_misc.py imgdiff  --a a.png --b b.png --out diff.png
  python ctf_misc.py bitplane --img a.png --plane 0 --channel 1 --out p0.png
  python ctf_misc.py cyclic   --length 200
  python ctf_misc.py cycfind  --data-hex 61616162... --length 200
  python ctf_misc.py gadgets  --file vuln --filter "pop rdi"
  python ctf_misc.py elfinfo  --file vuln

完整参数/输出/退出码见 CLI_REFERENCE.md。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import struct
import sys
import zlib

import numpy as np
from PIL import Image

try:
    from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_MODE_32
    from capstone.x86 import X86_INS_RET, X86_INS_RETF, X86_INS_SYSCALL
    HAS_CS = True
except Exception:
    HAS_CS = False

try:
    from elftools.elf.elffile import ELFFile
    from elftools.common.exceptions import ELFError
    HAS_ELF = True
except Exception:
    HAS_ELF = False


# =====================================================================
# 通用
# =====================================================================
MAGICS = {
    b'\x89PNG\r\n\x1a\n': 'PNG',
    b'PK\x03\x04': 'ZIP',
    b'PK\x05\x06': 'ZIP(empty)',
    b'\x7fELF': 'ELF',
    b'%PDF': 'PDF',
    b'\x1f\x8b\x08': 'GZIP',
    b'\xff\xd8\xff': 'JPEG',
    b'BZh': 'BZIP2',
    b'\xfd7zXZ\x00': 'XZ',
    b'7z\xbc\xaf\x27\x1c': '7Z',
    b'SQLite format 3': 'SQLite',
    b'Rar!\x1a\x07': 'RAR',
    b'\x78\x9c': 'zlib(78 9c)',
    b'\x78\xda': 'zlib(78 da)',
    b'\x78\x01': 'zlib(78 01)',
    b'OggS': 'OGG',
    b'RIFF': 'RIFF/WAV',
    b'\x00\x00\x01\x00': 'ICO',
    b'BM': 'BMP',
    b'GIF87a': 'GIF',
    b'GIF89a': 'GIF',
}

PNG_SIG = b'\x89PNG\r\n\x1a\n'


def find_magics(data: bytes, min_gap=0):
    """扫描数据里的已知 magic，返回 [(offset, name)]。"""
    hits = []
    for m, name in MAGICS.items():
        start = 0
        while True:
            i = data.find(m, start)
            if i < 0:
                break
            hits.append((i, name))
            start = i + 1
    hits.sort()
    return hits


def printable_runs(data: bytes, minlen=6):
    """提取连续可打印字符串。"""
    out = []
    cur = bytearray()
    pos = 0
    for i, b in enumerate(data):
        if 32 <= b < 127 or b in (9,):
            if not cur:
                pos = i
            cur.append(b)
        else:
            if len(cur) >= minlen:
                out.append((pos, bytes(cur)))
            cur = bytearray()
    if len(cur) >= minlen:
        out.append((pos, bytes(cur)))
    return out


def _ensure_parent(path):
    """为输出文件创建父目录（--out 指向不存在的目录时不再抛 FileNotFoundError）。"""
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


# =====================================================================
# LSB 提取
# =====================================================================
def lsb_bits(img, channels=(0, 1, 2), bit=0, col_major=False, interleave=False):
    """从图像按通道/位平面取 bit，返回 np.ndarray[uint8]（0/1）。

    - interleave=False：按通道顺序整平面读取（ch0 全部 → ch1 全部 …）
    - interleave=True ：按像素交织读取（每像素依次取 ch0, ch1, ch2 …）
    - col_major=True  ：像素按列优先（x 外层、y 内层）
    """
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[:, :, None]
    c = a.shape[2]
    order = [ch for ch in channels if 0 <= ch < c]
    if not order:
        return np.zeros(0, dtype=np.uint8)
    sub = a[:, :, order]
    if interleave:
        # 行优先：y, x, ch ；列优先：x, y, ch
        q = sub.transpose(1, 0, 2) if col_major else sub
    else:
        # 行优先：ch, y, x ；列优先：ch, x, y
        q = sub.transpose(2, 1, 0) if col_major else sub.transpose(2, 0, 1)
    return ((q >> bit) & 1).astype(np.uint8).ravel()


def bits_to_bytes(bits, msb_first=True):
    n = (len(bits) // 8) * 8
    if n <= 0:
        return b''
    b = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, 8)
    if msb_first:
        w = (1 << np.arange(7, -1, -1))
    else:
        w = (1 << np.arange(0, 8))
    return (b * w).sum(1).astype(np.uint8).tobytes()


def _grade(raw: bytes):
    """给一段提取数据打分：magic > flag 特征 > 内嵌 magic > 可打印率。

    只把 >=3 字节的 magic 用于"内嵌偏移"打分：BMP('BM')、zlib('78 9c') 这类
    2 字节 magic 在随机数据里命中率太高，会把真正的 flag 挤出结果。
    """
    for m, name in MAGICS.items():
        if raw.startswith(m):
            return 100, f'MAGIC {name}'
    # flag 特征优先于"偏移处 magic"，否则随机噪声里的 78 9c 会压过真 flag
    if re.search(rb'(flag|ctf|gylike|EAI|DASCTF|NSSCTF)\{', raw[:256], re.I):
        return 90, f'含 flag 特征 {raw[:48]!r}'
    for m, name in MAGICS.items():
        if len(m) < 3:
            continue
        off = raw[:4096].find(m)
        if 0 <= off < 256:
            return 80, f'MAGIC {name} @+{off}'
    head = raw[:64]
    printable = bytes(b for b in head if 32 <= b < 127)
    if len(printable) >= 8 and len(printable) / len(head) > 0.6:
        return 40, f'前缀可打印 {printable[:24]!r}'
    # 整体可打印率
    txt = bytes(b for b in raw if 32 <= b < 127)
    ratio = len(txt) / max(1, len(raw))
    if ratio > 0.9 and len(txt) > 16:
        if re.search(rb'(flag|ctf|gylike|\{|\})', raw, re.I):
            return 60, f'可打印(含 flag 特征) {raw[:32]!r}'
        return 20, f'可打印 {raw[:24]!r}'
    return 0, ''


def lsb_scan(path, max_report=8):
    """穷举 (通道组合 × 位平面 × 行/列序 × 像素交织 × 位序)，找可识别内容。"""
    img = Image.open(path)
    img.load()
    if img.mode not in ('RGB', 'RGBA', 'L', 'P'):
        img = img.convert('RGB')
    a = np.asarray(img)
    c = a.shape[2] if a.ndim == 3 else 1
    chan_sets = [(0, 1, 2), (0,), (1,), (2,), (2, 1, 0), (0, 1, 2, 3), (3,)]
    results = []
    seen_keys = set()
    for chs in chan_sets:
        eff = tuple(ch for ch in chs if ch < c)
        if not eff or eff in seen_keys:
            continue
        seen_keys.add(eff)
        for bit in range(3):
            for col_major in (False, True):
                for interleave in (False, True):
                    bits = lsb_bits(img, eff, bit, col_major, interleave)
                    for msb in (True, False):
                        raw = bits_to_bytes(bits, msb)
                        if len(raw) < 8:
                            continue
                        score, info = _grade(raw)
                        if score > 0:
                            results.append((score, eff, bit, col_major, msb, info,
                                            raw[:64], interleave))
    results.sort(key=lambda r: -r[0])
    print(f'[lsb] {path}  mode={img.mode} size={img.size}  命中 {len(results)} 组')
    for score, chs, bit, cm, msb, info, head, il in results[:max_report]:
        order = 'col' if cm else 'row'
        mode = 'pix' if il else 'seq'
        bitorder = 'msb' if msb else 'lsb'
        print(f'  score={score:3d} ch={chs} bit={bit} {order}/{mode} {bitorder} → {info}')
        print(f'        head={head[:48]!r}')
    return results


# =====================================================================
# 文件级隐写扫描
# =====================================================================
PNG_STANDARD_CHUNKS = ('IHDR', 'IDAT', 'IEND', 'PLTE', 'tRNS', 'gAMA', 'pHYs',
                       'tEXt', 'iTXt', 'zTXt', 'sRGB', 'bKGD', 'tIME',
                       'cHRM', 'sBIT', 'hIST', 'cICP', 'mDCV', 'cLLI', 'eXIf')


def _png_walk(data: bytes):
    """遍历 PNG chunk。

    返回 (chunks, iend_end, warnings)：
      chunks   = [(type_name, data_len, truncated)]
      iend_end = IEND chunk 结束后的文件偏移（找不到为 None）
      warnings = 结构异常提示
    """
    chunks, warnings = [], []
    pos, iend_end = 8, None
    n = len(data)
    while pos + 8 <= n:
        ln = struct.unpack('>I', data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        end = pos + 12 + ln
        typname = typ.decode('latin1', 'ignore')
        if end > n:
            chunks.append((typname, ln, True))
            warnings.append(f'chunk @{pos:#x} {typname!r} 声明长度 {ln}，'
                            f'实际仅剩 {n - pos - 8} 字节（文件被截断/长度异常）')
            break
        chunks.append((typname, ln, False))
        if typ == b'IEND':
            iend_end = end
            break
        pos = end
    return chunks, iend_end, warnings


def stegscan(path):
    """文件级隐写扫描，返回解析出的关键信息 dict（便于库调用/自测断言）。"""
    data = open(path, 'rb').read()
    info = {'size': len(data), 'png_tail': None, 'jpeg_tail': None,
            'chunks': None, 'zlib_hits': []}
    print(f'[stegscan] {path}  {len(data):,} 字节')
    mags = find_magics(data)
    print(f'  内嵌 magic: {[(o, n) for o, n in mags][:12]}')

    # 1) PNG：IEND 之后的数据（按 chunk 结构定位真正的 IEND，不再用 rfind）
    if data.startswith(b'\x89PNG'):
        chunks, iend_end, warns = _png_walk(data)
        if iend_end is None:
            # 兜底：找最后一个完整的空 IEND chunk 签名
            j = data.rfind(b'\x00\x00\x00\x00IEND')
            if j >= 0:
                iend_end = j + 12
        if iend_end is not None:
            tail = data[iend_end:]
            info['png_tail'] = len(tail)
            tag = '  ★ 可疑' if tail else ''
            print(f'  PNG: IEND 之后还有 {len(tail)} 字节{tag}')
            if tail:
                print(f'    尾部 head={tail[:48]!r}')
                for m, n in MAGICS.items():
                    j = tail.find(m)
                    if j >= 0:
                        print(f'    ★ 尾部 +{j} 处含 {n}')
        else:
            print('  PNG: 未找到 IEND（文件可能被截断）')
        odd = [(t, l) for t, l, _ in chunks if t not in PNG_STANDARD_CHUNKS]
        info['chunks'] = [(t, l) for t, l, _ in chunks]
        print(f'  PNG chunks={len(chunks)}  非常规 chunk: {odd[:6]}')
        for w in warns:
            print(f'  ⚠ {w}')

    # 2) JPEG：FFD9 之后
    if data.startswith(b'\xff\xd8'):
        idx = data.rfind(b'\xff\xd9')
        if idx >= 0:
            tail_len = len(data) - idx - 2
            info['jpeg_tail'] = tail_len
            if tail_len > 0:
                print(f'  JPEG: FFD9 之后还有 {tail_len} 字节  ★ 可疑')
                print(f'    尾部 head={data[idx + 2:idx + 2 + 48]!r}')

    # 3) zlib 流尝试解压
    found = 0
    for m in (b'\x78\x9c', b'\x78\xda', b'\x78\x01'):
        start = 0
        while found < 3:
            i = data.find(m, start)
            if i < 0:
                break
            start = i + 1
            try:
                out = zlib.decompressobj().decompress(data[i:i + 100000])
            except Exception:
                continue
            if len(out) > 32:
                print(f'  ★ zlib@{i:#x} 解出 {len(out)} 字节: {out[:48]!r}')
                info['zlib_hits'].append((i, len(out), out[:64]))
                found += 1

    # 4) 可疑长字符串
    runs = printable_runs(data)
    sus = [r for r in runs if re.search(rb'(flag|ctf|gylike|passw|secret|key)', r[1], re.I)]
    print(f'  可打印串 {len(runs)} 段，可疑 {len(sus)} 段')
    for off, s in sus[:6]:
        print(f'    @{off:#x} {s[:64]!r}')
    info['suspect_strings'] = sus[:6]
    return info


# =====================================================================
# 图像差异
# =====================================================================
def imgdiff(pa, pb, out_path=None):
    a = np.asarray(Image.open(pa).convert('RGB')).astype(np.int16)
    b = np.asarray(Image.open(pb).convert('RGB')).astype(np.int16)
    if a.shape != b.shape:
        print(f'[imgdiff] 尺寸不同 {a.shape} vs {b.shape}，裁剪到较小者')
        h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    d = np.abs(a - b)
    diff = d.sum(2)
    ys, xs = np.nonzero(diff > 0)
    print(f'[imgdiff] 差异像素 {len(ys)} / {diff.size}  最大差 {int(diff.max()) if diff.size else 0}')
    if len(ys):
        print(f'  包围盒 x[{xs.min()}..{xs.max()}] y[{ys.min()}..{ys.max()}]')
    # 通道级 LSB 差异（隐写常见）
    x = (a ^ b)
    bitdiff = x & 1
    print(f'  LSB 层差异像素 {int(bitdiff.sum())}')
    if out_path:
        _ensure_parent(out_path)
        vis = np.clip(diff * 8, 0, 255).astype(np.uint8)
        Image.fromarray(vis).save(out_path)
        print(f'  差异图已保存 {out_path}')
    return diff


def bitplane(path, plane=0, channel=0, out_path=None):
    if not 0 <= plane <= 7:
        raise ValueError(f'--plane 必须在 0..7 之间（得到 {plane}）')
    if not 0 <= channel <= 2:
        raise ValueError(f'--channel 必须在 0..2 之间（RGB 转换后只有 3 个通道，得到 {channel}）')
    img = Image.open(path).convert('RGB')
    a = np.asarray(img)
    p = ((a[:, :, channel] >> plane) & 1) * 255
    im = Image.fromarray(p.astype(np.uint8))
    nonzero = int((p > 0).sum())
    if out_path:
        _ensure_parent(out_path)
        im.save(out_path)
        print(f'[bitplane] plane={plane} ch={channel} → {out_path}  非零像素 {nonzero}')
    else:
        print(f'[bitplane] plane={plane} ch={channel}  非零像素 {nonzero}')
    return im


# =====================================================================
# Pwn：cyclic pattern（与 pwntools pwnlib.util.cyclic 的 de Bruijn 一致）
# =====================================================================
CYCLIC_ALPHABET = b'abcdefghijklmnopqrstuvwxyz'
_CYCLIC_CACHE = {}


def _de_bruijn(alphabet: bytes, n: int = 4) -> bytes:
    """FKM de Bruijn 序列生成器（与 pwnlib.util.cyclic.de_bruijn 相同）。"""
    k = len(alphabet)
    a = [0] * (k * n)
    seq = bytearray()

    def db(t, p):
        if t > n:
            if n % p == 0:
                for j in range(1, p + 1):
                    seq.append(alphabet[a[j]])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return bytes(seq)


def cyclic(length, alphabet=None, n=4):
    """De Bruijn 序列，与 pwntools `cyclic` 兼容（默认小写字母，子序列长度 n=4）。

    已知向量：cyclic(20) == b'aaaabaaacaaadaaaeaaa'
              cyclic(32) == b'aaaabaaacaaadaaaeaaafaaagaaahaaa'
    """
    if length is None:
        length = 0
    if length < 0:
        raise ValueError(f'--length 不能为负（得到 {length}）')
    alpha = bytes(alphabet) if alphabet is not None else CYCLIC_ALPHABET
    if len(alpha) < 2:
        raise ValueError('字母表至少需要 2 个字符')
    key = (alpha, n)
    seq = _CYCLIC_CACHE.get(key)
    if seq is None:
        seq = _de_bruijn(alpha, n)
        _CYCLIC_CACHE[key] = seq
    if length > len(seq):
        raise ValueError(
            f'--length={length} 超过 de Bruijn 序列上限 {len(seq)}'
            f'（字母表 {len(alpha)}，n={n}）；pwntools 默认 n=4')
    return seq[:length]


def cycfind(needle_hex, length=4096):
    pat = cyclic(length)
    nd = bytes.fromhex(needle_hex) if isinstance(needle_hex, str) else bytes(needle_hex)
    if not nd:
        raise ValueError('--data-hex 不能为空')
    i = pat.find(nd)
    print(f'[cycfind] 模式长 {len(pat)}  查找 {nd.hex()}')
    if i >= 0:
        print(f'  偏移 = {i}')
    else:
        print('  偏移 = -1  (未找到，可试 --length 更大)')
    return i


# =====================================================================
# Pwn：ROP gadget
# =====================================================================
GADGET_WINDOW = 32   # 只回溯结尾指令前 32 字节（与常见 gadget 长度匹配，控制大文件耗时）


def _terminator_len(data: bytes, pos: int) -> int:
    """pos 处开始的控制流结尾指令字节长度（ret/retf/ret imm16/syscall），否则 0。"""
    n = len(data)
    b = data[pos]
    if b == 0xC3:              # ret
        return 1
    if b == 0xCB:              # retf
        return 1
    if b == 0xC2:              # ret imm16
        return 3 if pos + 2 < n else 0
    if b == 0x0F and pos + 1 < n and data[pos + 1] == 0x05:   # syscall
        return 2
    return 0


def _is_terminator_insn(ins) -> bool:
    return ins.id in (X86_INS_RET, X86_INS_RETF, X86_INS_SYSCALL)


def find_gadgets(path, flt=None, max_insn=6, bits=64, limit=40):
    if not HAS_CS:
        print('[gadgets] 未安装 capstone')
        return []
    data = open(path, 'rb').read()
    md = Cs(CS_ARCH_X86, CS_MODE_64 if bits == 64 else CS_MODE_32)
    md.detail = False
    results = []
    seen = set()
    n = len(data)
    for tpos in range(n):
        tlen = _terminator_len(data, tpos)
        if not tlen:
            continue
        end = tpos + tlen
        lo = max(0, end - GADGET_WINDOW)
        for start in range(tpos, lo - 1, -1):
            insns = list(md.disasm(data[start:end], start))
            if not insns:
                continue
            last = insns[-1]
            if last.address + last.size != end or last.address != tpos:
                continue
            if not _is_terminator_insn(last):
                continue
            if len(insns) > max_insn:
                continue
            # 结尾指令之前不允许再出现控制流转移，否则该 gadget 无法连续执行
            if any(_is_terminator_insn(x) for x in insns[:-1]):
                continue
            text = ' ; '.join(f'{x.mnemonic} {x.op_str}'.strip() for x in insns)
            if flt and flt.lower() not in text.lower():
                continue
            entry = (start, text)
            if entry in seen:
                continue
            seen.add(entry)
            results.append(entry)
    results.sort()
    print(f'[gadgets] {path} 找到 {len(results)} 条' + (f'（过滤 "{flt}"）' if flt else ''))
    for addr, text in results[:limit]:
        print(f'  {addr:#08x}: {text}')
    return results


# =====================================================================
# Pwn：ELF 概览
# =====================================================================
_DF_BIND_NOW = 0x8
_DF_1_NOW = 0x1
_DF_1_PIE = 0x08000000
_PF_X = 0x1
_DANGEROUS = {'system', 'execve', 'gets', 'read', 'write', 'printf', 'puts',
              'scanf', 'strcpy', 'sprintf', 'mprotect'}


def _elf_dynamic_tags(elf):
    tags = {}
    try:
        for seg in elf.iter_segments():
            if seg['p_type'] == 'PT_DYNAMIC':
                for tag in seg.iter_tags():
                    tags[tag.entry.d_tag] = tag.entry.d_val
    except Exception:
        pass
    return tags


def _elf_has_segment(elf, ptype):
    try:
        for seg in elf.iter_segments():
            if seg['p_type'] == ptype:
                return seg
    except Exception:
        pass
    return None


def _elfinfo_report(path, data):
    elf = ELFFile(io.BytesIO(data))
    try:
        arch = elf.get_machine_arch()
    except Exception:
        arch = f'unknown({elf["e_machine"]})'
    bits = elf.elfclass
    etype = elf['e_type']
    entry = elf['e_entry']
    print(f'[elfinfo] {path}')
    print(f'  arch={arch}  bits={bits}  type={etype}  entry={int(entry):#x}')

    # 权限标志：直接从程序头/动态段读取（不靠文件里有没有某个字符串）
    tags = _elf_dynamic_tags(elf)
    stack_seg = _elf_has_segment(elf, 'PT_GNU_STACK')
    if stack_seg is None:
        nx = '未知'
    else:
        nx = '否' if (stack_seg['p_flags'] & _PF_X) else '是'
    has_interp = _elf_has_segment(elf, 'PT_INTERP') is not None
    is_pie = (etype == 'ET_DYN') and (
        has_interp or bool(tags.get('DT_FLAGS_1', 0) & _DF_1_PIE))
    has_relro = _elf_has_segment(elf, 'PT_GNU_RELRO') is not None
    bind_now = ('DT_BIND_NOW' in tags
                or bool(tags.get('DT_FLAGS', 0) & _DF_BIND_NOW)
                or bool(tags.get('DT_FLAGS_1', 0) & _DF_1_NOW))
    canary = b'__stack_chk_fail' in data
    print(f'  NX: {nx} | PIE: {"是" if is_pie else "否"} | '
          f'Canary: {"是" if canary else "否"} | '
          f'RELRO(Full): {"是" if (has_relro and bind_now) else "否"}')

    # 危险的动态导入（只算未定义符号 = 真正的外部导入）
    try:
        dynsym = elf.get_section_by_name('.dynsym')
    except Exception:
        dynsym = None
    if dynsym:
        imports = set()
        for s in dynsym.iter_symbols():
            if s.name in _DANGEROUS and s['st_shndx'] == 'SHN_UNDEF':
                imports.add(s.name)
        print(f'  危险导入: {sorted(imports)}')

    # PLT / GOT 地址
    for sec_name in ('.plt', '.plt.sec', '.got', '.got.plt'):
        try:
            s = elf.get_section_by_name(sec_name)
        except Exception:
            s = None
        if s:
            print(f'  {sec_name}: {s["sh_addr"]:#x} (size {s["sh_size"]})')

    # 符号地址（常用）
    try:
        symtab = elf.get_section_by_name('.symtab')
    except Exception:
        symtab = None
    if symtab:
        picks = []
        for s in symtab.iter_symbols():
            if s.name in ('main', 'win', 'vuln', 'flag', 'system', 'gets'):
                picks.append((s.name, s['st_value'], s['st_size']))
        if picks:
            print('  关键符号:')
            for nm, v, sz in picks:
                print(f'    {nm:12s} {v:#x}  size={sz}')


def elfinfo(path):
    """打印 ELF 概览；返回 0 成功 / 1 失败（非 ELF、截断、文件不存在都不抛 traceback）。"""
    if not HAS_ELF:
        print('[elfinfo] 未安装 pyelftools')
        return 1
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError as e:
        print(f'[elfinfo] 无法读取文件: {e}', file=sys.stderr)
        return 1
    try:
        _elfinfo_report(path, data)
    except ELFError as e:
        print(f'[elfinfo] 不是有效的 ELF 文件: {e}', file=sys.stderr)
        return 1
    except (struct.error, ValueError) as e:
        print(f'[elfinfo] ELF 解析失败: {e}', file=sys.stderr)
        return 1
    return 0


# =====================================================================
# selftest
# =====================================================================
def cmd_selftest(a):
    import tempfile

    fails = []

    def check(label, cond, detail=''):
        print(f'{label} {"PASS" if cond else "FAIL"}' + (f'  {detail}' if detail else ''))
        if not cond:
            fails.append(label)

    with tempfile.TemporaryDirectory(prefix='ctf_misc_selftest_') as tmp:
        # --- LSB 隐写往返（R 通道 LSB，行优先，msb 位序）---
        secret = b'flag{lsb_stego_test}'
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        bits = np.unpackbits(np.frombuffer(secret, dtype=np.uint8))
        rch = img[:, :, 0].reshape(-1).copy()                    # 只写 R 通道的 LSB
        rch[:len(bits)] = (rch[:len(bits)] & 0xFE) | bits
        img[:, :, 0] = rch.reshape(64, 64)
        p1 = os.path.join(tmp, 'stego.png')
        Image.fromarray(img).save(p1)
        res = lsb_scan(p1)
        found = any((b'flag{' in x[6]) or ('flag' in x[5]) for x in res[:5])
        print(f'[LSB 隐写往返] {"PASS" if found else "FAIL"}')
        if not found:
            fails.append('[LSB 隐写往返]')

        # --- LSB 像素交织（常见 CTF 写法：每像素依次取 R/G/B 的 LSB）---
        flat = img.reshape(-1).copy()
        bitlen = len(bits)
        idx = 0
        for pos in range(len(flat)):
            if idx >= bitlen:
                break
            flat[pos] = (flat[pos] & 0xFE) | int(bits[idx])
            idx += 1
        p_il = os.path.join(tmp, 'stego_interleaved.png')
        Image.fromarray(flat.reshape(64, 64, 3)).save(p_il)
        res_il = lsb_scan(p_il)
        il_ok = any((b'flag{' in x[6]) or ('flag' in x[5]) for x in res_il[:5])
        print(f'[LSB 像素交织] {"PASS" if il_ok else "FAIL"}')
        if not il_ok:
            fails.append('[LSB 像素交织]')

        # --- 尾附加数据扫描（PNG IEND 之后 36 字节，含 ZIP magic）---
        p2 = os.path.join(tmp, 'trail.png')
        base = open(p1, 'rb').read()
        tail = b'PK\x03\x04' + b'A' * 32
        open(p2, 'wb').write(base + tail)
        print('[尾附加扫描]')
        info = stegscan(p2)
        tail_ok = info.get('png_tail') == len(tail)
        print(f'[尾附加长度] {"PASS" if tail_ok else "FAIL"}'
              f'  (期望 {len(tail)}, 得到 {info.get("png_tail")})')
        if not tail_ok:
            fails.append('[尾附加长度]')

        # --- cyclic / cycfind：必须与 pwntools 的 de Bruijn 一致 ---
        vec_ok = (cyclic(20) == b'aaaabaaacaaadaaaeaaa'
                  and cyclic(32) == b'aaaabaaacaaadaaaeaaafaaagaaahaaa')
        print(f'[cyclic pwntools 兼容] {"PASS" if vec_ok else "FAIL"}')
        if not vec_ok:
            fails.append('[cyclic pwntools 兼容]')
        pat = cyclic(200)
        needle = pat[72:80]
        i = cycfind(needle.hex(), 4096)
        print(f'[cyclic] {"PASS" if i == 72 else "FAIL"}  (期望 72, 得到 {i})')
        if i != 72:
            fails.append('[cyclic]')
        i0 = cycfind(pat[0:4].hex(), 4096)
        print(f'[cycfind 偏移 0] {"PASS" if i0 == 0 else "FAIL"}  (期望 0, 得到 {i0})')
        if i0 != 0:
            fails.append('[cycfind 偏移 0]')

        # --- gadget 搜索：手工构造 "pop rdi ; ret" + 裸 ret/retf/syscall ---
        p3 = os.path.join(tmp, 'blob.bin')
        blob = b'\x90' * 16 + b'\x5f\xc3' + b'\x90' * 8 + b'\x58\x5a\x5b\xc3'
        open(p3, 'wb').write(blob)
        print('[gadgets]')
        gs = find_gadgets(p3, flt=None, max_insn=4)
        ok = any('pop rdi ; ret' in t for _, t in gs)
        print(f'[gadget 搜索] {"PASS" if ok else "FAIL"}')
        if not ok:
            fails.append('[gadget 搜索]')
        p4 = os.path.join(tmp, 'term.bin')
        open(p4, 'wb').write(b'\xc3\xcb\xc2\x00\x00\x0f\x05')
        gterm = {t for _, t in find_gadgets(p4, flt=None, max_insn=1)}
        term_ok = {'ret', 'retf', 'ret 0', 'syscall'} <= gterm
        print(f'[gadget 裸结尾指令] {"PASS" if term_ok else "FAIL"}  找到 {sorted(gterm)}')
        if not term_ok:
            fails.append('[gadget 裸结尾指令]')

        # --- ELF 信息：这里没有真实 ELF，跳过真实 ELF 用例，只测非 ELF 的干净失败路径 ---
        print('[elfinfo] 跳过真实 ELF（本机无 /usr/bin/ls），改测非 ELF 容错')
        p5 = os.path.join(tmp, 'not_elf.bin')
        open(p5, 'wb').write(b'not an elf file at all\n' * 4)
        with contextlib.redirect_stderr(io.StringIO()):
            rc_elf = elfinfo(p5)
        print(f'[elfinfo 非 ELF 容错] {"PASS" if rc_elf == 1 else "FAIL"}  (rc={rc_elf})')
        if rc_elf != 1:
            fails.append('[elfinfo 非 ELF 容错]')

        # --- 参数校验：bitplane 越界应给出干净错误 ---
        try:
            bitplane(p1, plane=8)
            bitplane_ok = False
        except ValueError:
            bitplane_ok = True
        print(f'[bitplane 参数校验] {"PASS" if bitplane_ok else "FAIL"}')
        if not bitplane_ok:
            fails.append('[bitplane 参数校验]')

    if fails:
        print(f'[selftest] {len(fails)} 项 FAIL: {fails}')
        return 1
    print('[selftest] 全部 PASS')
    return 0


# =====================================================================
# CLI
# =====================================================================
def _cmd_cycfind(a):
    cycfind(a.data_hex, a.length)
    # 库函数返回偏移；CLI 里不要让偏移变成进程退出码
    return None


def main():
    ap = argparse.ArgumentParser(
        description='ctf_misc — 取证与 Pwn 辅助（纯 CPU；完整说明见 CLI_REFERENCE.md）')
    sub = ap.add_subparsers(dest='cmd', required=True,
                            metavar='{selftest,lsb,stegscan,imgdiff,bitplane,cyclic,cycfind,gadgets,elfinfo}')
    sub.add_parser('selftest', help='自测（全 PASS 退出码 0）')

    p = sub.add_parser('lsb', help='图像 LSB 位平面穷举提取（按 magic/flag/可打印率打分）')
    p.add_argument('--img', required=True, help='图片路径')

    p = sub.add_parser('stegscan', help='文件级隐写扫描：尾附加/内嵌 magic/chunk 异常/zlib/字符串')
    p.add_argument('--file', required=True, help='待扫描文件路径')

    p = sub.add_parser('imgdiff', help='两图差异包围盒 + LSB 差异 + 可视化')
    p.add_argument('--a', required=True, help='图 A 路径')
    p.add_argument('--b', required=True, help='图 B 路径')
    p.add_argument('--out', default=None, help='差异可视化图输出路径（文件，父目录自动创建）')

    p = sub.add_parser('bitplane', help='单 bit 平面导出为 PNG')
    p.add_argument('--img', required=True, help='图片路径')
    p.add_argument('--plane', type=int, default=0, help='位平面 0~7，默认 0')
    p.add_argument('--channel', type=int, default=0, help='RGB 通道 0~2，默认 0')
    p.add_argument('--out', default=None, help='PNG 输出路径（文件，不是目录）')

    p = sub.add_parser('cyclic', help='生成 pwntools 兼容 de Bruijn 循环模式')
    p.add_argument('--length', type=int, default=200, help='模式长度，默认 200（上限 26^4）')

    p = sub.add_parser('cycfind', help='在数据中定位循环模式偏移')
    p.add_argument('--data-hex', required=True, help='崩溃数据，裸 hex')
    p.add_argument('--length', type=int, default=4096, help='模式长度，默认 4096')

    p = sub.add_parser('gadgets', help='ROP gadget 搜索（ret/retf/ret imm16/syscall 结尾）')
    p.add_argument('--file', required=True, help='二进制文件路径')
    p.add_argument('--filter', default=None, help='只保留含该子串的 gadget，如 "pop rdi"')
    p.add_argument('--max-insn', type=int, default=6, help='gadget 最大指令数，默认 6')
    p.add_argument('--bits', type=int, default=64, choices=[32, 64], help='架构位数 32/64，默认 64')

    p = sub.add_parser('elfinfo', help='ELF 概览：架构/保护/导入/PLT/GOT')
    p.add_argument('--file', required=True, help='ELF 文件路径')

    a = ap.parse_args()
    fns = {
        'selftest': cmd_selftest,
        'lsb': lambda x: lsb_scan(x.img),
        'stegscan': lambda x: stegscan(x.file),
        'imgdiff': lambda x: imgdiff(x.a, x.b, x.out),
        'bitplane': lambda x: bitplane(x.img, x.plane, x.channel, x.out),
        'cyclic': lambda x: print(cyclic(x.length)),
        'cycfind': _cmd_cycfind,
        'gadgets': lambda x: find_gadgets(x.file, x.filter, x.max_insn, x.bits),
        'elfinfo': lambda x: elfinfo(x.file),
    }
    try:
        rc = fns[a.cmd](a)
    except (OSError, ValueError) as e:
        print(f'[{a.cmd}] 错误: {e}', file=sys.stderr)
        return 1
    # 约定：函数返回 int 视为退出码（selftest/elfinfo），其它返回值（bytes/list/ndarray…）忽略
    return 0 if not isinstance(rc, int) else rc


if __name__ == '__main__':
    sys.exit(main())
