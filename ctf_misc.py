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
  1  参数/运行期错误（文件不存在、图像非法、非 ELF、参数越界或非法等；
     stderr 一行 "[cmd] 错误: ..."，无 traceback；参数解析错误同样是 1，不是 argparse 默认的 2）

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
# _grade 里 flag 特征的扫描窗口。原来的 256B 太短：隐写消息只要带一点二进制前缀
# （长度头/盐值/分块写入）就会把明显的 flag 判成 0 分、完全不输出。
_FLAG_SCAN = 1 << 16
_FLAG_RE = re.compile(rb'(flag|ctf|gylike|EAI|DASCTF|NSSCTF)\{', re.I)


def _valid_bmp(data: bytes, off: int) -> bool:
    """校验 BMP 头：过滤文本/随机数据里 2 字节 'BM' 的高频误报。"""
    n = len(data)
    if off + 26 > n:
        return False
    size, r1, r2, pixoff = struct.unpack_from('<IHHI', data, off + 2)
    if r1 or r2:                     # 两个 reserved 字段按规范必须为 0
        return False
    dib = struct.unpack_from('<I', data, off + 14)[0]
    if dib not in (12, 40, 52, 56, 64, 108, 124):
        return False
    remain = n - off
    if not 14 <= pixoff <= remain:   # 像素数据偏移必须落在文件内
        return False
    if size and not 26 <= size <= remain:   # 文件大小字段要么 0 要么合理
        return False
    return True


def _valid_ico(data: bytes, off: int) -> bool:
    """校验 ICO 目录项：过滤二进制里常见的 00 00 01 00 误报。"""
    n = len(data)
    if off + 6 > n:
        return False
    count = struct.unpack_from('<H', data, off + 4)[0]
    if not 1 <= count <= 64:
        return False
    ent = off + 6
    if ent + 16 * count > n:
        return False
    head_size = 6 + 16 * count
    for i in range(count):
        e = ent + 16 * i
        if data[e + 3] != 0:         # 目录项 reserved 必须为 0
            return False
        size, imgoff = struct.unpack_from('<II', data, e + 8)
        if size == 0 or not head_size <= imgoff < n - off:
            return False
    return True


def _valid_zlib(data: bytes, off: int) -> bool:
    """校验 78 xx 是真的 zlib 流（2 字节头在随机数据里约 3/65536 命中）。"""
    if len(data) - off < 2:
        return False
    # PNG 的 IDAT 被拆成多个 chunk 时 zlib 流在文件里不连续，解压器会报错；
    # 但紧跟在 IDAT 类型后的数据就是合法的 zlib 流起点（O(1) 判定，不遍历 chunk）。
    if off >= 8 and data[off - 4:off] == b'IDAT' and data.startswith(PNG_SIG):
        return True
    try:
        zlib.decompressobj().decompress(data[off:off + 4096])
    except Exception:
        return False
    return True


def _valid_gzip(data: bytes, off: int) -> bool:
    """gzip 头 3 字节在压缩数据里会误报：校验 FLG 保留位并试解压。"""
    if len(data) - off < 10:
        return False
    if data[off + 3] & 0xE0:                      # FLG 保留位必须为 0
        return False
    try:
        zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data[off:off + 4096])
    except Exception:
        return False
    return True


def _valid_bzip2(data: bytes, off: int) -> bool:
    """BZh 后必须是压缩级别 + bzip2 块/EOS 魔数（0x314159265359 / 0x177245385090）。"""
    if len(data) - off < 10:
        return False
    if data[off + 3] not in b'123456789':
        return False
    return data[off + 4:off + 10] in (b'\x31\x41\x59\x26\x53\x59',
                                      b'\x17\x72\x45\x38\x50\x90')


_MAGIC_VALIDATORS = {
    b'BM': _valid_bmp,
    b'\x00\x00\x01\x00': _valid_ico,
    b'\x78\x9c': _valid_zlib,
    b'\x78\xda': _valid_zlib,
    b'\x78\x01': _valid_zlib,
    b'\x1f\x8b\x08': _valid_gzip,
    b'BZh': _valid_bzip2,
}


def _magic_ok(m: bytes, data: bytes, off: int) -> bool:
    """短 magic 需要格式校验；其余 magic 直接算命中。"""
    fn = _MAGIC_VALIDATORS.get(m)
    return True if fn is None else fn(data, off)


def find_magics(data: bytes, min_gap=0):
    """扫描数据里的已知 magic，返回 [(offset, name)]。

    2 字节短签名（BMP/zlib）与 ICO 会做格式校验，避免把普通文本/随机字节
    误报成"内嵌文件"。min_gap > 0 时，距上一个保留命中不足 min_gap 字节的
    命中会被丢弃（用于降噪）。
    """
    hits = []
    for m, name in MAGICS.items():
        start = 0
        while True:
            i = data.find(m, start)
            if i < 0:
                break
            start = i + 1
            if _magic_ok(m, data, i):
                hits.append((i, name))
    hits.sort()
    if min_gap > 0:
        kept = []
        last = None
        for off, name in hits:
            if last is not None and off - last < min_gap:
                continue
            kept.append((off, name))
            last = off
        hits = kept
    return hits


def printable_runs(data: bytes, minlen=6):
    """提取连续可打印字符串（numpy 向量化：100MB 文件从数秒降到亚秒级）。"""
    if not data:
        return []
    arr = np.frombuffer(data, dtype=np.uint8)
    mask = ((arr >= 32) & (arr < 127)) | (arr == 9)
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    out = []
    for s, e in zip(starts, ends):
        if e - s >= minlen:
            out.append((int(s), bytes(data[int(s):int(e)])))
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
    # 整通道集直接用视图，避免 fancy-index 复制整张大图
    sub = a if order == list(range(c)) else a[:, :, order]
    if interleave:
        # 行优先：y, x, ch ；列优先：x, y, ch
        q = sub.transpose(1, 0, 2) if col_major else sub
    else:
        # 行优先：ch, y, x ；列优先：ch, x, y
        q = sub.transpose(2, 1, 0) if col_major else sub.transpose(2, 0, 1)
    return np.asarray((q >> bit) & 1, dtype=np.uint8).ravel()


def bits_to_bytes(bits, msb_first=True):
    """把 0/1 比特流按 8 位打包成 bytes（numpy 向量化，避免大图 int64 临时数组）。"""
    n = (len(bits) // 8) * 8
    if n <= 0:
        return b''
    b = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, 8)
    return np.packbits(b, axis=1, bitorder='big' if msb_first else 'little').tobytes()


def _grade(raw: bytes):
    """给一段提取数据打分：magic > flag 特征 > 内嵌 magic > 可打印率。

    只把 >=3 字节的 magic 用于"内嵌偏移"打分：BMP('BM')、zlib('78 9c') 这类
    2 字节 magic 在随机数据里命中率太高；它们即使出现在开头也先做格式校验。
    flag 特征扫描前 64 KiB（而不是 256 B），带二进制前缀的消息不再被判 0 分。
    """
    for m, name in MAGICS.items():
        if raw.startswith(m) and _magic_ok(m, raw, 0):
            return 100, f'MAGIC {name}'
    # flag 特征优先于"偏移处 magic"，否则随机噪声里的 78 9c 会压过真 flag
    fm = _FLAG_RE.search(raw, 0, _FLAG_SCAN)
    if fm:
        off = fm.start()
        if off == 0:
            return 90, f'含 flag 特征 {raw[:48]!r}'
        return 90, f'含 flag 特征 @+{off} {raw[off:off + 48]!r}'
    for m, name in MAGICS.items():
        if len(m) < 3:
            continue
        off = raw[:4096].find(m)
        if 0 <= off < 256 and _magic_ok(m, raw, off):
            return 80, f'MAGIC {name} @+{off}'
    head = raw[:64]
    printable = bytes(b for b in head if 32 <= b < 127)
    if len(printable) >= 8 and len(printable) / len(head) > 0.6:
        return 40, f'前缀可打印 {printable[:24]!r}'
    # 整体可打印率（numpy 向量化：大图一次扫描要跑上百个组合）
    if not raw:
        return 0, ''
    arr = np.frombuffer(raw, dtype=np.uint8)
    txt_len = int(((arr >= 32) & (arr < 127)).sum())
    if txt_len / len(raw) > 0.9 and txt_len > 16:
        if re.search(rb'(flag|ctf|gylike|\{|\})', raw, re.I):
            return 60, f'可打印(含 flag 特征) {raw[:32]!r}'
        return 20, f'可打印 {raw[:24]!r}'
    return 0, ''


def _png_16bit_color_hint(path):
    """PNG 16-bit 彩色/带 alpha 会被 Pillow 解码成 8-bit（低字节丢失），给出提示。"""
    try:
        with open(path, 'rb') as f:
            head = f.read(33)
    except OSError:
        return None
    if len(head) < 26 or head[:8] != PNG_SIG or head[12:16] != b'IHDR':
        return None
    bitdepth, colortype = head[24], head[25]
    if bitdepth == 16 and colortype in (2, 4, 6):
        return (f'PNG bit depth 16 + color type {colortype}：Pillow 解码会丢弃低字节，'
                f'lsb 只能看到高 8 位（隐写在 bit0~7 时提不出来）')
    return None


def lsb_scan(path, max_report=8):
    """穷举 (通道组合 × 位平面 × 行/列序 × 像素交织 × 位序)，找可识别内容。

    保留图像原生位深：16-bit 灰度不再被 convert('RGB') 压成 8-bit（那样会把
    隐写用的低位直接丢掉）；只有浮点等无法做位运算的模式才转 RGB。
    """
    img = Image.open(path)
    img.load()
    arr = np.asarray(img)
    if arr.dtype.kind not in 'uib':      # float/complex 无法做移位，转 RGB 后再扫
        img = img.convert('RGB')
        arr = np.asarray(img)
    c = arr.shape[2] if arr.ndim == 3 else 1
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
                # 单通道图 seq 与 pix 的取值顺序完全相同，跳过重复组合
                for interleave in ((False,) if c == 1 else (False, True)):
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
    warn = _png_16bit_color_hint(path)
    if warn:
        print(f'  ⚠ {warn}')
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
            # 兜底：找最后一个完整的空 IEND chunk 签名（CRC 缺失即文件被截断，不能谎报 0 字节尾数据）
            j = data.rfind(b'\x00\x00\x00\x00IEND')
            if 0 <= j and j + 12 <= len(data):
                iend_end = j + 12
        if iend_end is not None:
            tail = data[iend_end:]
            info['png_tail'] = len(tail)
            tag = '  ★ 可疑' if tail else ''
            print(f'  PNG: IEND 之后还有 {len(tail)} 字节{tag}')
            if tail:
                print(f'    尾部 head={tail[:48]!r}')
                for j, n in find_magics(tail):
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
    # 通道级 LSB 差异（隐写常见）：按"像素"计数（任一通道 LSB 不同即算一个像素）
    x = (a ^ b)
    bitdiff = (x & 1).any(axis=2)
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
    if not isinstance(n, int) or n < 1:
        raise ValueError(f'n 必须 >= 1（得到 {n!r}）')
    try:
        alpha = bytes(alphabet) if alphabet is not None else CYCLIC_ALPHABET
    except TypeError:
        raise ValueError('alphabet 必须是 bytes（元素为 0..255 的字节串）') from None
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
GADGET_WINDOW = 32   # 结尾指令之前最多回溯 32 字节（含结尾指令共 33~35 字节）


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
    if bits not in (32, 64):
        raise ValueError(f'--bits 只能是 32 或 64（得到 {bits}）')
    if max_insn < 1:
        raise ValueError(f'--max-insn 必须 >= 1（得到 {max_insn}）')
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
        # 窗口按"结尾指令之前的字节数"算：start >= tpos - 32，正好 32 字节也要包含
        lo = max(0, tpos - GADGET_WINDOW)
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


def _elf_truncation(elf, data):
    """返回程序头/节头表越界描述；None 表示结构完整。

    CLI_REFERENCE 约定"截断的 ELF 退出码 1"：pyelftools 很懒，只有真正访问到
    越界结构时才报错，而各个解析分支又都做了容错，所以这里显式校验表边界。
    """
    n = len(data)
    try:
        phnum = elf['e_phnum']
        phentsize = elf['e_phentsize']
        phoff = elf['e_phoff']
        shnum = elf['e_shnum']
        shentsize = elf['e_shentsize']
        shoff = elf['e_shoff']
    except Exception:
        return None
    if phnum and phnum != 0xffff:      # 0xffff = PN_XNUM（扩展计数，极少见，交给 pyelftools）
        if phentsize < 32 or phoff + phnum * phentsize > n:
            return f'程序头表越界（phoff={phoff:#x} phnum={phnum} phentsize={phentsize}）'
    if shnum:                          # e_shnum=0 可能是无节头，也可能是扩展计数
        if shentsize < 40 or shoff + shnum * shentsize > n:
            return f'节头表越界（shoff={shoff:#x} shnum={shnum} shentsize={shentsize}）'
    return None


def _elfinfo_report(path, data):
    elf = ELFFile(io.BytesIO(data))
    trunc = _elf_truncation(elf, data)
    if trunc:
        raise ELFError(f'ELF 结构被截断：{trunc}')
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

        # --- 16-bit 灰度 PNG：必须按原生位深提取，不能被 convert('RGB') 丢掉低位 ---
        secret16 = b'flag{16bit_native}'
        h16 = rng.integers(0, 65536, (32, 32), dtype=np.uint16)
        bits16 = np.unpackbits(np.frombuffer(secret16, dtype=np.uint8))
        flat16 = h16.reshape(-1).copy()
        mask16 = np.array(np.iinfo(np.uint16).max - 1, dtype=np.uint16)
        flat16[:len(bits16)] = (flat16[:len(bits16)] & mask16) | bits16.astype(np.uint16)
        p16 = os.path.join(tmp, 'stego16.png')
        Image.fromarray(flat16.reshape(32, 32)).save(p16)
        res16 = lsb_scan(p16)
        ok16 = any(secret16[:12] in x[6] for x in res16)
        print(f'[LSB 16-bit 灰度] {"PASS" if ok16 else "FAIL"}'
              f'  (mode={Image.open(p16).mode}, 命中 {len(res16)})')
        if not ok16:
            fails.append('[LSB 16-bit 灰度]')

        # --- flag 前有 300B 二进制前缀：_grade 的 flag 扫描窗口不能只有 256B ---
        deep = b'\x00' * 300 + b'flag{deep_offset}' + b'\x00' * 64
        rgba_d = rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)
        dbits = np.unpackbits(np.frombuffer(deep, dtype=np.uint8))
        dflat = rgba_d.reshape(-1).copy()
        dpos = np.arange(0, 64 * 64 * 4, 4)          # R 通道，行优先
        dn = min(len(dbits), len(dpos))
        dflat[dpos[:dn]] = (dflat[dpos[:dn]] & 0xFE) | dbits[:dn]
        pdeep = os.path.join(tmp, 'stego_deep.png')
        Image.fromarray(dflat.reshape(64, 64, 4)).save(pdeep)
        resdeep = lsb_scan(pdeep)
        deep_ok = any('flag{deep_offset}' in x[5] for x in resdeep)
        print(f'[LSB 深偏移 flag] {"PASS" if deep_ok else "FAIL"}'
              f'  (前缀 300B, 命中 {len(resdeep)})')
        if not deep_ok:
            fails.append('[LSB 深偏移 flag]')

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
        # --- de Bruijn 独立已知向量（FKM 二进制/三进制参考值）+ n 合法性校验 ---
        db_ok = (_de_bruijn(b'01', 3) == b'00010111'
                 and _de_bruijn(b'abc', 2) == b'aabacbbcc')
        try:
            cyclic(10, b'ab', 0)
            n_bad = False
        except ValueError:
            n_bad = True
        print(f'[de Bruijn 独立向量/n 校验] {"PASS" if (db_ok and n_bad) else "FAIL"}')
        if not (db_ok and n_bad):
            fails.append('[de Bruijn 独立向量/n 校验]')

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

        # --- gadget 回溯窗口 = 结尾指令前 32 字节（正好 32 字节必须包含，33 字节必须排除）---
        p6 = os.path.join(tmp, 'win32.bin')
        open(p6, 'wb').write(b'\x90' * 32 + b'\xc3')
        gw = find_gadgets(p6, flt=None, max_insn=40, limit=1)
        win_ok = any(a == 0 for a, _ in gw)
        open(p6, 'wb').write(b'\x90' * 33 + b'\xc3')
        gw2 = find_gadgets(p6, flt=None, max_insn=40, limit=1)
        win_ok = win_ok and (not any(a == 0 for a, _ in gw2))
        print(f'[gadget 32B 回溯窗口] {"PASS" if win_ok else "FAIL"}')
        if not win_ok:
            fails.append('[gadget 32B 回溯窗口]')
        # --- gadgets 参数校验：--bits / --max-insn 非法值干净报错 ---
        bad_param = 0
        for kw in ({'bits': 16}, {'max_insn': 0}):
            try:
                find_gadgets(p4, **kw)
            except ValueError:
                bad_param += 1
        print(f'[gadgets 参数校验] {"PASS" if bad_param == 2 else "FAIL"}')
        if bad_param != 2:
            fails.append('[gadgets 参数校验]')

        # --- ELF 信息：本机无真实 ELF，手工构造最小 ELF64（无节头）做保护解析回归 ---
        def _stub_elf64(truncate=0):
            interp = b'/lib64/ld-linux-x86-64.so.2\x00'
            phnum, phoff = 5, 64
            interp_off = 64 + phnum * 56
            dyn_off = (interp_off + len(interp) + 7) // 8 * 8
            dynstr_off = dyn_off + 48                    # 3 个 dynamic tag
            # DT_STRTAB 指向文件内 dynstr（无节头时 pyelftools 靠它建字符串表）；
            # DT_BIND_NOW 让 RELRO 判 Full；PT_GNU_STACK RW 让 NX 判"是"。
            dyn = (struct.pack('<qQ', 5, 0x400000 + dynstr_off)
                   + struct.pack('<qQ', 24, 0)
                   + struct.pack('<qQ', 0, 0))
            filesz = dynstr_off + 1

            def ph(t, flags, off, va, sz, al=0x1000):
                return struct.pack('<IIQQQQQQ', t, flags, off, va, 0x400000, sz, sz, al)
            phdrs = (ph(1, 5, 0, 0x400000, filesz)                          # PT_LOAD
                     + ph(3, 4, interp_off, 0x400000 + interp_off, len(interp), 1)
                     + ph(0x6474e551, 6, 0, 0, 0, 16)                       # PT_GNU_STACK RW
                     + ph(0x6474e552, 4, dyn_off, 0x403000, 8, 1)           # PT_GNU_RELRO
                     + ph(2, 6, dyn_off, 0x403040, len(dyn), 8))            # PT_DYNAMIC
            ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + b'\x00' * 8
            hdr = ident + struct.pack('<HHIQQQIHHHHHH', 3, 62, 1, 0x401000, phoff, 0,
                                      0, 64, 56, phnum, 64, 0, 0)
            data = bytearray(hdr + phdrs)
            for _pos, _blob in ((interp_off, interp), (dyn_off, dyn), (dynstr_off, b'\x00')):
                if len(data) < _pos:
                    data.extend(b'\x00' * (_pos - len(data)))
                data[_pos:_pos + len(_blob)] = _blob
            return bytes(data[:len(data) - truncate]) if truncate else bytes(data)

        print('[elfinfo] 本机无真实 ELF，改用手工构造的最小 ELF64 做保护/截断回归')
        p5 = os.path.join(tmp, 'stub.elf')
        open(p5, 'wb').write(_stub_elf64())
        ebuf = io.StringIO()
        with contextlib.redirect_stdout(ebuf):
            rc_stub = elfinfo(p5)
        out_stub = ebuf.getvalue()
        stub_ok = (rc_stub == 0 and 'NX: 是' in out_stub and 'PIE: 是' in out_stub
                   and 'RELRO(Full): 是' in out_stub)
        print(f'[elfinfo 最小 ELF64 保护解析] {"PASS" if stub_ok else "FAIL"}  (rc={rc_stub})')
        if not stub_ok:
            fails.append('[elfinfo 最小 ELF64 保护解析]')
        ptr = os.path.join(tmp, 'stub_trunc.elf')
        open(ptr, 'wb').write(_stub_elf64(truncate=200))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            rc_trunc = elfinfo(ptr)
        print(f'[elfinfo 截断 ELF 容错] {"PASS" if rc_trunc == 1 else "FAIL"}  (rc={rc_trunc})')
        if rc_trunc != 1:
            fails.append('[elfinfo 截断 ELF 容错]')
        p5 = os.path.join(tmp, 'not_elf.bin')
        open(p5, 'wb').write(b'not an elf file at all\n' * 4)
        with contextlib.redirect_stderr(io.StringIO()):
            rc_elf = elfinfo(p5)
        print(f'[elfinfo 非 ELF 容错] {"PASS" if rc_elf == 1 else "FAIL"}  (rc={rc_elf})')
        if rc_elf != 1:
            fails.append('[elfinfo 非 ELF 容错]')

        # --- imgdiff：LSB 层差异按"像素"计数（不是按通道值）---
        ia = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        ib = ia.copy()
        ib[1, 2] = (ib[1, 2] + 1) & 0xFF
        pia, pib = os.path.join(tmp, 'ia.png'), os.path.join(tmp, 'ib.png')
        Image.fromarray(ia).save(pia)
        Image.fromarray(ib).save(pib)
        ibuf = io.StringIO()
        with contextlib.redirect_stdout(ibuf):
            imgdiff(pia, pib)
        diff_ok = 'LSB 层差异像素 1' in ibuf.getvalue()
        print(f'[imgdiff LSB 像素计数] {"PASS" if diff_ok else "FAIL"}')
        if not diff_ok:
            fails.append('[imgdiff LSB 像素计数]')

        # --- 短 magic 校验：文本 'BM' 不再误报 BMP；真实 zlib/gzip/bzip2 流仍能识别 ---
        import bz2
        import gzip
        fp = find_magics(b'BMW cars and BM things\n' * 3)
        fake_gz = find_magics(b'\x1f\x8b\x08\xe5' + b'\x00' * 8)      # FLG 保留位非 0
        z_ok = any(o == 0 and n == 'zlib(78 9c)' for o, n in find_magics(zlib.compress(b'hello world' * 10)))
        gz_ok = find_magics(gzip.compress(b'hello')) == [(0, 'GZIP')]
        bz_ok = find_magics(bz2.compress(b'hello')) == [(0, 'BZIP2')]
        magic_ok = (not fp and not fake_gz and z_ok and gz_ok and bz_ok)
        print(f'[magic 短签名误报抑制] {"PASS" if magic_ok else "FAIL"}  (误报 {fp[:3]} {fake_gz[:2]})')
        if not magic_ok:
            fails.append('[magic 短签名误报抑制]')

        # --- printable_runs 偏移/最小长度（numpy 向量化回归）---
        pr = printable_runs(b'\x00\x01hello world\x00\x02ab\x03flag{abc}\xff\tTAB\t\x00end', 6)
        pr_ok = pr == [(2, b'hello world'), (18, b'flag{abc}')]
        print(f'[printable_runs 偏移] {"PASS" if pr_ok else "FAIL"}  {pr}')
        if not pr_ok:
            fails.append('[printable_runs 偏移]')

        # --- PNG IEND CRC 被截断：必须报"未找到 IEND"，不能谎报 0 字节尾数据 ---
        pcut = os.path.join(tmp, 'iend_nocrc.png')
        open(pcut, 'wb').write(base[:-4])
        cbuf = io.StringIO()
        with contextlib.redirect_stdout(cbuf):
            cinfo = stegscan(pcut)
        cut_ok = cinfo.get('png_tail') is None and '未找到 IEND' in cbuf.getvalue()
        print(f'[PNG IEND 截断容错] {"PASS" if cut_ok else "FAIL"}')
        if not cut_ok:
            fails.append('[PNG IEND 截断容错]')

        # --- bits_to_bytes 打包（MSB/LSB 位序）---
        pack_ok = (bits_to_bytes([1, 0, 0, 0, 0, 0, 0, 0], True) == b'\x80'
                   and bits_to_bytes([1, 0, 0, 0, 0, 0, 0, 0], False) == b'\x01')
        print(f'[bits_to_bytes 位序] {"PASS" if pack_ok else "FAIL"}')
        if not pack_ok:
            fails.append('[bits_to_bytes 位序]')

        # --- CLI 参数解析错误退出码 = 1（CLI_REFERENCE §6，不是 argparse 默认的 2）---
        argv_bak = sys.argv
        sys.argv = ['ctf_misc.py', 'gadgets', '--file', 'x', '--bits', '16']
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                main()
            cli_rc = None
        except SystemExit as e:
            cli_rc = e.code
        finally:
            sys.argv = argv_bak
        print(f'[CLI 参数错误退出码] {"PASS" if cli_rc == 1 else "FAIL"}  (rc={cli_rc})')
        if cli_rc != 1:
            fails.append('[CLI 参数错误退出码]')

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


class _MiscArgumentParser(argparse.ArgumentParser):
    """参数错误统一返回退出码 1（argparse 默认 2；CLI_REFERENCE §6 规定 ctf_misc 为 1）。"""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f'[ctf_misc] 错误: {message}', file=sys.stderr)
        raise SystemExit(1)


def main():
    ap = _MiscArgumentParser(
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
