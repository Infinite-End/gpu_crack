#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf_tools.py — CTF 常用密码学内核（与 gpu_crack.py 共用后端）

后端：torch-CUDA / torch-CPU / numpy（--backend auto|torch|numpy，默认 auto）

子命令
------
  chacha    ChaCha20 密钥流批量生成 / 候选密钥扫描（12B nonce=RFC7539；8B=原版 64bit 计数器）
  poly1305  Poly1305 / ChaCha20-Poly1305 tag 计算与校验（标量，过滤后的最终验证）
  rs63      GF(256) RS(63,31) syndrome 批量校验 + 单字节纠错定位
  crc32     CRC32 批量计算 / 尾部 1~2 字节爆破匹配
  xorbrute  1~3 字节重复 XOR 爆破，按词频 + flag 结构打分
  aesctr    AES-192 CTR 密钥流 / 加解密
  selftest  已知向量自测（9 项；全 PASS 退出码 0，失败 1）

全局参数（必须写在子命令之前）
------------------------------
  --backend {auto,torch,numpy}   默认 auto
  --chunk N                      每批候选数，默认 16384（chacha --brute 用）

输入约定
--------
  * 所有 *-hex 参数接受裸 hex 或 0x 前缀，容忍内部空白；长度错误/非法 hex 打印 [错误] 到
    stderr 并返回退出码 2（不抛 traceback）。
  * 退出码：0=成功（未命中也是 0）；1=selftest 失败 / poly1305 --expect-hex 不匹配；2=参数或输入错误。

用法
----
  python ctf_tools.py selftest
  python ctf_tools.py chacha   --target-hex 3836c93c... --keys-file keys.txt
  python ctf_tools.py chacha   --target-hex 3836c93c... --brute --key-len 32
  python ctf_tools.py poly1305 --key-hex <32B> --msg-hex <hex> --mode aead [--nonce-hex <12B>]
  python ctf_tools.py rs63     --code-hex 504d3331... [--locate] [--fcr 1 --order lsb]
  python ctf_tools.py crc32    --data-hex 89504e47 --target-hex 1a2b3c4d
  python ctf_tools.py xorbrute --data-hex 1b37373331363f78151b7f2b783431333d --max-len 3
  python ctf_tools.py aesctr   --key-hex <24B> --nonce-hex <8B> --mode ks --length 64

完整参数/输出/退出码见 CLI_REFERENCE.md。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_crack as G                      # noqa: E402


def _T():
    return G.xp()


def _dev():
    return G.DEV


def _is_torch():
    return G.BACKEND.startswith("torch")


M32 = 0xFFFFFFFF


def _parse_hex(s, name='hex'):
    """解析命令行/密钥文件里的 hex：容忍 0x 前缀与空白，出错抛 ValueError。"""
    if s is None:
        raise ValueError(f'缺少 {name}')
    t = ''.join(str(s).split())
    if t[:2].lower() == '0x':
        t = t[2:]
    if len(t) % 2:
        raise ValueError(f'{name} 的 hex 长度为奇数（{len(t)} 个字符）: {t[:64]!r}')
    try:
        return bytes.fromhex(t)
    except ValueError:
        raise ValueError(f'{name} 不是合法 hex: {t[:64]!r}')


def _load_keys_file(path, key_len):
    """读取密钥文件：每行一个 hex 密钥；支持 0x 前缀、空行、# 注释行。"""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise ValueError(f'无法读取密钥文件 {path}: {e}')
    keys = []
    for no, line in enumerate(lines, 1):
        s = ''.join(line.split())
        if not s or s.startswith('#'):
            continue
        if s[:2].lower() == '0x':
            s = s[2:]
        try:
            kb = bytes.fromhex(s)
        except ValueError:
            raise ValueError(f'{path}:{no} 不是合法 hex: {line.strip()[:64]!r}')
        if len(kb) != key_len:
            raise ValueError(f'{path}:{no} 密钥 {len(kb)} 字节，需要 {key_len} 字节')
        keys.append(list(kb))
    if not keys:
        raise ValueError(f'{path} 中没有任何密钥')
    return np.array(keys, dtype=np.uint8)


# =====================================================================
# 1. ChaCha20
# =====================================================================
def _rotl(x, n):
    # 注意：numpy 后端必须用 Python int 掩码。NumPy 2 起 int64 数组与 np.uint64
    # 标量做位运算会被提升为 float64，直接 TypeError（旧代码只测过 torch 后端）。
    return ((x << n) | (x >> (32 - n))) & M32


def chacha20_keystream(keys, nonce, counter=0, nbytes=64, rfc7539=None):
    """
    批量生成 ChaCha20 密钥流。
    keys   : [B,32] uint8 张量/ndarray（或 32 字节 bytes，单密钥）
    nonce  : 12 字节（rfc7539）/ 8 字节（原版）
    counter: 起始计数器（rfc7539 为 32 位；原版为 64 位）
    rfc7539: None = 按 nonce 长度自动判断（12B→True，8B→False）
    nbytes : 要生成的字节数（允许 0）
    返回   : [B, nbytes] uint8 (numpy)
    """
    T = _T()
    is_t = _is_torch()
    nonce = bytes(nonce)
    if rfc7539 is None:
        rfc7539 = (len(nonce) == 12)
    if rfc7539 and len(nonce) != 12:
        raise ValueError(f'rfc7539 变体需要 12 字节 nonce，收到 {len(nonce)} 字节')
    if not rfc7539 and len(nonce) != 8:
        raise ValueError(f'原版 ChaCha20 需要 8 字节 nonce，收到 {len(nonce)} 字节')
    counter, nbytes = int(counter), int(nbytes)
    if nbytes < 0:
        raise ValueError(f'nbytes 不能为负: {nbytes}')
    blocks = (nbytes + 63) // 64
    max_ctr = M32 if rfc7539 else (1 << 64) - 1
    if counter < 0 or counter > max_ctr or counter + max(0, blocks - 1) > max_ctr:
        raise ValueError(f'ChaCha20 计数器越界（counter={counter}, blocks={blocks}, '
                         f'rfc7539={rfc7539}, 上限 {max_ctr}）')
    if isinstance(keys, (bytes, bytearray)):
        kb = np.frombuffer(bytes(keys), dtype=np.uint8)
        if kb.size != 32:
            raise ValueError(f'ChaCha20 密钥必须 32 字节，收到 {kb.size} 字节')
        keys = kb.reshape(1, 32)
    k_np = (G.from_dev(keys) if (is_t and hasattr(keys, 'detach')) else np.asarray(keys))
    if k_np.ndim != 2 or k_np.shape[1] != 32:
        raise ValueError(f'keys 形状应为 [B,32]，收到 {k_np.shape}')
    B = k_np.shape[0]
    if nbytes == 0:                              # 空请求：直接给空结果，别进 block 循环
        return np.zeros((B, 0), dtype=np.uint8)

    nb = np.frombuffer(nonce, dtype=np.uint8)
    const = np.frombuffer(b'expand 32-byte k', dtype=np.uint8)

    def w(a):                                    # 4 字节小端 → uint32
        a = [int(v) for v in a]
        return (a[0] | (a[1] << 8) | (a[2] << 16) | (a[3] << 24)) & M32

    st = np.zeros((B, 16), dtype=np.int64)
    for i in range(4):
        st[:, i] = w(const[4 * i:4 * i + 4])
    for i in range(8):
        st[:, 4 + i] = (k_np[:, 4 * i].astype(np.int64)
                        | (k_np[:, 4 * i + 1].astype(np.int64) << 8)
                        | (k_np[:, 4 * i + 2].astype(np.int64) << 16)
                        | (k_np[:, 4 * i + 3].astype(np.int64) << 24))
    if rfc7539:
        st[:, 12] = counter & M32
        for i in range(3):
            st[:, 13 + i] = w(nb[4 * i:4 * i + 4])
    else:
        # 原版 ChaCha20：counter 占 words 12-13（64 位），nonce 占 words 14-15
        st[:, 12] = counter & M32
        st[:, 13] = (counter >> 32) & M32
        st[:, 14] = w(nb[0:4]); st[:, 15] = w(nb[4:8])
    ctr_lo = int(counter) & M32
    ctr_hi = (int(counter) >> 32) & M32

    st = (T.tensor(st, dtype=T.int64, device=_dev()) if is_t else st)
    outs = []
    for bi in range(blocks):
        if bi > 0:
            # 每个 block 计数器 +1；原版变体是 64 位计数器（words 12-13），需要处理进位
            ctr_lo = (ctr_lo + 1) & M32
            if ctr_lo == 0:
                ctr_hi = (ctr_hi + 1) & M32
            if is_t:
                st = st.clone()
                st[:, 12] = ctr_lo
                if not rfc7539:
                    st[:, 13] = ctr_hi
            else:
                st = st.copy()
                st[:, 12] = ctr_lo
                if not rfc7539:
                    st[:, 13] = ctr_hi
        x = [st[:, i].clone() if is_t else st[:, i].copy() for i in range(16)]

        def qr(a, b, c, d):
            a = (a + b) & M32; d ^= a; d = _rotl(d, 16)
            c = (c + d) & M32; b ^= c; b = _rotl(b, 12)
            a = (a + b) & M32; d ^= a; d = _rotl(d, 8)
            c = (c + d) & M32; b ^= c; b = _rotl(b, 7)
            return a, b, c, d

        for _ in range(10):
            x[0], x[4], x[8], x[12] = qr(x[0], x[4], x[8], x[12])
            x[1], x[5], x[9], x[13] = qr(x[1], x[5], x[9], x[13])
            x[2], x[6], x[10], x[14] = qr(x[2], x[6], x[10], x[14])
            x[3], x[7], x[11], x[15] = qr(x[3], x[7], x[11], x[15])
            x[0], x[5], x[10], x[15] = qr(x[0], x[5], x[10], x[15])
            x[1], x[6], x[11], x[12] = qr(x[1], x[6], x[11], x[12])
            x[2], x[7], x[8], x[13] = qr(x[2], x[7], x[8], x[13])
            x[3], x[4], x[9], x[14] = qr(x[3], x[4], x[9], x[14])
        parts = [((x[i] + st[:, i]) & M32) for i in range(16)]
        word = T.stack(parts, dim=1) if is_t else np.stack(parts, axis=1)   # [B,16] uint32
        # words → 小端字节（统一用 Python int 位移，避免 int64×uint64 提升问题）
        if is_t:
            bs = T.stack([(word >> (8 * j)) & 0xFF for j in range(4)], dim=2).reshape(B, 64)
        else:
            bs = np.stack([(word >> (8 * j)) & 0xFF for j in range(4)], axis=2).reshape(B, 64)
        outs.append(bs)
    out = T.cat(outs, dim=1) if is_t else np.concatenate(outs, axis=1)
    return G.from_dev(out)[:, :nbytes].astype(np.uint8)   # 统一返回 numpy


# =====================================================================
# 2. Poly1305（标量，最终验证用）
# =====================================================================
def poly1305(key32, msg, aad=b''):
    """标准 Poly1305（RFC 8439 单块式），key32 = otk + r 的子密钥。"""
    key32 = bytes(key32)
    if len(key32) != 32:
        raise ValueError(f'Poly1305 子密钥必须 32 字节，收到 {len(key32)} 字节')
    r = int.from_bytes(key32[:16], 'little')
    r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
    s = int.from_bytes(key32[16:32], 'little')
    p = (1 << 130) - 5
    a = 0
    data = bytes(aad) + bytes(msg)
    for i in range(0, len(data), 16):
        blk = data[i:i + 16]
        n = int.from_bytes(blk + b'\x01', 'little')
        a = ((a + n) * r) % p
    a = (a + s) & ((1 << 128) - 1)
    return a.to_bytes(16, 'little')


def poly1305_tag(key32, aad, ct):
    """ChaCha20-Poly1305 的 tag：Poly1305(pad(aad)||pad(ct)||len)"""
    def pad(x):
        return x + b'\x00' * ((-len(x)) % 16)
    mac_data = pad(bytes(aad)) + pad(bytes(ct)) + \
        len(bytes(aad)).to_bytes(8, 'little') + len(bytes(ct)).to_bytes(8, 'little')
    return poly1305(key32, mac_data)


# =====================================================================
# 3. GF(256) / RS(63,31)
# =====================================================================
GF_POLY = 0x11D
_EXP = np.zeros(512, dtype=np.uint8)
_LOG = np.zeros(256, dtype=np.int16)
_v, _x = 1, 1
for _i in range(255):
    _EXP[_i] = _v
    _LOG[_v] = _i
    _v <<= 1
    if _v & 0x100:
        _v ^= GF_POLY
_EXP[255:510] = _EXP[0:255]


def gf_mul_tab(a, b):
    """GF(256) 乘法（numpy 广播，查表）。"""
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    mask = (a == 0) | (b == 0)
    idx = (_LOG[a & 0xFF].astype(np.int64) + _LOG[b & 0xFF].astype(np.int64)) % 255
    out = _EXP[idx]
    return np.where(mask, 0, out).astype(np.uint8)


def rs63_syndromes(codewords, nsym=32, fcr=0, order='msb'):
    """
    批量计算 RS syndrome（GF(256), 本原多项式 0x11d）。
    codewords : [B, n] uint8（n=63 时为 RS(63,31)，nsym=32）；也接受单条 1-D
    order     : 'msb' = c[0] 是最高次项（码字左端为 X^(n-1) 系数）
                'lsb' = c[0] 是常数项
    返回      : [B, nsym] uint8；全 0 表示合法码字
    """
    cw = (np.frombuffer(bytes(codewords), dtype=np.uint8)
          if isinstance(codewords, (bytes, bytearray))
          else np.asarray(codewords, dtype=np.uint8))
    if cw.ndim == 1:
        cw = cw.reshape(1, -1)
    nsym, fcr = int(nsym), int(fcr)
    if not 1 <= nsym <= 255:
        raise ValueError(f'nsym 必须在 1~255，收到 {nsym}')
    if not 0 <= fcr < 255:
        raise ValueError(f'fcr 必须在 0~254，收到 {fcr}')
    if order not in ('msb', 'lsb'):
        raise ValueError(f"order 只能是 'msb'/'lsb'，收到 {order!r}")
    B, n = cw.shape
    if not 1 <= n <= 255:
        raise ValueError(f'码字长度必须在 1~255（GF(256) 上限），收到 {n}')
    j = np.arange(nsym)[None, :]
    i = np.arange(n)[:, None]
    if order == 'msb':
        i = (n - 1 - i)
    pw = _EXP[((j + fcr) * i) % 255]                     # [n, nsym]
    res = np.zeros((B, nsym), dtype=np.uint8)
    for k in range(n):
        col = cw[:, k][:, None]
        res = res ^ gf_mul_tab(col, pw[k][None, :])
    return res


def rs63_check(codeword, nsym=32, fcr=0, order='msb', auto=True):
    """
    单个码字校验，返回 (ok, info)。
    auto=True 时自动尝试 (order, fcr) 的 4 种组合，命中即返回命中的约定；
    否则按给定 (order, fcr) 校验，info 为 syndrome 十六进制。
    """
    if isinstance(codeword, (bytes, bytearray)):
        cw = np.frombuffer(bytes(codeword), dtype=np.uint8)
    else:
        cw = np.asarray(codeword, dtype=np.uint8)
    cw = cw.reshape(1, -1)
    if auto:
        for o in ('msb', 'lsb'):
            for f in (0, 1):
                s = rs63_syndromes(cw, nsym, f, o)[0]
                if (s == 0).all():
                    return True, 'order=%s fcr=%d' % (o, f)
        s = rs63_syndromes(cw, nsym, 0, 'msb')[0]
        return False, s.tobytes().hex()
    s = rs63_syndromes(cw, nsym, fcr, order)[0]
    return bool((s == 0).all()), s.tobytes().hex()


def rs63_locate(codeword, nsym=32, fcr=0, order='msb'):
    """
    单字节错误定位（利用 syndrome 对错误位置的线性性，O(n·nsym)，不逐字节试值）。
    返回 (pos, delta)：把 cw[pos] ^= delta 即得到合法码字；无法单字节修正返回 None。
    """
    cw = (np.frombuffer(bytes(codeword), dtype=np.uint8)
          if isinstance(codeword, (bytes, bytearray))
          else np.asarray(codeword, dtype=np.uint8)).reshape(-1)
    n = cw.size
    s = rs63_syndromes(cw.reshape(1, -1), nsym, fcr, order)[0]
    if (s == 0).all():
        return None                                    # 本来就是合法码字
    if not (s != 0).all():
        return None          # 单字节错误下所有 syndrome 分量必非零，出现 0 说明错≥2 字节
    j = np.arange(int(nsym))
    log_s = _LOG[s].astype(np.int64)
    for k in range(n):
        deg = (n - 1 - k) if order == 'msb' else k
        pw = _EXP[((j + int(fcr)) * deg) % 255]
        # 误差位置 k 时 s_j = delta·α^((j+fcr)·deg)，故 s_j / pw_j 必须是同一个非零 delta
        ratio = _EXP[(log_s - _LOG[pw].astype(np.int64)) % 255]
        if not (ratio == ratio[0]).all():
            continue
        delta = int(ratio[0])
        if delta == 0:
            continue
        fixed = cw.copy(); fixed[k] ^= delta
        if (rs63_syndromes(fixed.reshape(1, -1), nsym, fcr, order)[0] == 0).all():
            return int(k), delta
    return None


# =====================================================================
# 4. CRC32
# =====================================================================
_CRC_TAB = None


def _crc32_tab():
    global _CRC_TAB
    if _CRC_TAB is None:
        t = np.zeros(256, dtype=np.uint32)
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ (0xEDB88320 if c & 1 else 0)
            t[i] = c
        _CRC_TAB = t
    return _CRC_TAB


def crc32_batch(data):
    """
    批量 CRC32（标准多项式 0xEDB88320，初值/异或 0xFFFFFFFF）。
    data : [B, L] uint8（不足的尾部填 0 时结果不同，调用方保证等长）
    返回 : [B] uint32
    """
    tab = _crc32_tab()
    d = np.asarray(data, dtype=np.uint8)
    B, L = d.shape
    crc = np.full(B, 0xFFFFFFFF, dtype=np.uint32)
    for k in range(L):
        crc = tab[(crc ^ d[:, k]) & 0xFF] ^ (crc >> np.uint32(8))
    return crc ^ np.uint32(0xFFFFFFFF)


def crc32_state(data):
    """CRC32 中间状态（未做最终异或），供「尾部追加字节」增量爆破用。"""
    tab = _crc32_tab()
    c = 0xFFFFFFFF
    for b in bytes(data):
        c = int(tab[(c ^ b) & 0xFF]) ^ (c >> 8)
    return c & 0xFFFFFFFF


# =====================================================================
# 5. 重复 XOR 爆破
# =====================================================================
# 字节权重表：英文频率 + flag 特征。
# 旧版按「花括号/下划线」计权，会被 XOR 乱码里的 `{`/`}`/`_` 堆砌骗过
# （cryptopals 单字节题里 5 个假花括号的候选得分反而高于真英文），现改为
# 频率加权 + 严格的结构加分：结构分只给「全可打印且带右花括号」的候选。
_W256 = np.full(256, -20, dtype=np.int64)            # 非可打印：重罚
for _c in range(0x20, 0x7F):
    _W256[_c] = 0                                    # 可打印：默认 0
_W256[0x20] = 12                                     # 空格最像英文
for _c in b'etaoinshr':
    _W256[_c] = 9                                    # 高频字母
for _c in b'dlucmfwgypb':
    _W256[_c] = 6                                    # 中频字母
for _c in b'vkjxqz':
    _W256[_c] = 3                                    # 低频字母
for _c in range(0x41, 0x5B):
    _W256[_c] = 2                                    # 大写字母（英文/flag 里少见）
for _c in range(0x30, 0x3A):
    _W256[_c] = 3                                    # 数字
for _c in b'{}_-':
    _W256[_c] = 2                                    # flag 标点：低权重，防刷分

_FLAG_WORDS = (b'flag', b'ctf', b'gylike', b'eai', b'nssctf', b'dasctf')


def _has_seq_ci(a, seq, lowered=False):
    """
    [B,L] 字节矩阵里是否含 seq（ASCII 大小写不敏感），返回 [B] bool。
    lowered=True 表示 a 已转小写（_score_plaintext 批量调用时复用，省大数组拷贝）。
    """
    B, L = a.shape
    k = len(seq)
    if k > L:
        return np.zeros(B, dtype=bool)
    low = a if lowered else np.where((a >= 0x41) & (a <= 0x5A), a + 32, a)  # 大写 → 小写
    first = (low[:, :L - k + 1] == seq[0])                      # 候选起点必须匹配首字节
    rows = np.nonzero(first.any(1))[0]
    out = np.zeros(B, dtype=bool)
    if rows.size == 0:
        return out
    sub = low[rows]
    hit = first[rows]
    for j in range(1, k):
        hit &= (sub[:, j:L - k + 1 + j] == seq[j])
    out[rows] = hit.any(1)
    return out


def _score_plaintext(arr):
    """
    批量打分：[B,L] uint8 → [B] 分数（numpy 全向量化，无 Python 逐行循环）。
    逐字节权重：空格 12 / 高频字母 9、中频 6、低频 3 / 数字 3 / 大写 2 /
                flag 标点 { } _ - 2 / 其它可打印 0 / 非可打印 −20；
    结构加分：已知前缀 flag{ / ctf{ … +80（大小写不敏感）；
              「单词+{」且整行可打印、含右花括号 +20；「右花括号」+5。
    """
    a = np.asarray(arr, dtype=np.uint8)
    B, L = a.shape
    score = _W256[a].sum(1)
    low = np.where((a >= 0x41) & (a <= 0x5A), a + 32, a)       # 小写副本（uint8）
    printable = ((a >= 0x20) & (a <= 0x7E)).all(1)
    flag = np.zeros(B, dtype=bool)
    for word in _FLAG_WORDS:
        flag |= _has_seq_ci(low, word + b'{', lowered=True)
    if L >= 4:
        wordc = ((low >= 0x61) & (low <= 0x7A)) | (low == 0x5F)
        generic = (wordc[:, :-3] & wordc[:, 1:-2] & wordc[:, 2:-1]
                   & (low[:, 3:] == 0x7B)).any(1)
    else:
        generic = np.zeros(B, dtype=bool)
    has_close = (a == 0x7D).any(1)
    score = score + flag.astype(np.int64) * 80
    score = score + np.where(printable & generic & has_close, 20, 0)
    score = score + np.where(printable & has_close, 5, 0)
    return score


def xorbrute(data, max_len=3, top=10):
    """
    重复 XOR 爆破。data: bytes。返回 [(score, key_hex, plaintext), ...]
    max_len 只支持 1~3：3 字节 = 1677 万候选（几秒），4 字节 = 43 亿，不可行。
    """
    d = np.frombuffer(bytes(data), dtype=np.uint8)
    n = len(d)
    if n == 0:
        raise ValueError('xorbrute 的 data 不能为空')
    max_len, top = int(max_len), int(top)
    if not 1 <= max_len <= 3:
        raise ValueError(f'max_len 只支持 1~3，收到 {max_len}')
    if top < 1:
        raise ValueError(f'top 必须 >= 1，收到 {top}')
    results = []
    for klen in range(1, max_len + 1):
        total = 256 ** klen
        keys = np.arange(total, dtype=np.uint64)
        keybytes = np.stack([(keys >> np.uint64(8 * (klen - 1 - i))) & np.uint64(0xFF)
                             for i in range(klen)], axis=1).astype(np.uint8)
        step = max(1, 4_000_000 // max(1, n))
        for s in range(0, total, step):
            kb = keybytes[s:s + step]
            ks = np.tile(kb, (1, (n + klen - 1) // klen))[:, :n]
            pt = d[None, :] ^ ks
            sc = _score_plaintext(pt)
            idx = np.argsort(sc)[-top:][::-1]
            for i in idx:
                results.append((int(sc[i]), kb[i].tobytes().hex(), pt[i].tobytes()))
    results.sort(key=lambda r: -r[0])
    # 去重
    seen, out = set(), []
    for r in results:
        if r[2] in seen:
            continue
        seen.add(r[2]); out.append(r)
    return out[:top]


# =====================================================================
# 6. AES-192 CTR 密钥流
# =====================================================================
def aesctr_keystream(key, nonce8, nbytes=64, counter=0):
    """AES-192-CTR 密钥流（16 字节块，计数器块 = nonce8 || counter64 小端）。"""
    key, nonce8 = bytes(key), bytes(nonce8)
    if len(key) != 24:
        raise ValueError(f'AES-192 密钥必须 24 字节，收到 {len(key)} 字节')
    if len(nonce8) != 8:
        raise ValueError(f'CTR 计数器块前缀（nonce）必须 8 字节，收到 {len(nonce8)} 字节')
    nbytes, counter = int(nbytes), int(counter)
    if nbytes < 0:
        raise ValueError(f'nbytes 不能为负: {nbytes}')
    blocks = (nbytes + 15) // 16
    if counter < 0 or counter + max(0, blocks - 1) > (1 << 64) - 1:
        raise ValueError(f'AES-CTR 计数器越界（counter={counter}, blocks={blocks}）')
    if blocks == 0:
        return b''
    T = _T(); is_t = _is_torch()
    ctrs = [nonce8 + (counter + i).to_bytes(8, 'little') for i in range(blocks)]
    blk = np.frombuffer(b''.join(ctrs), dtype=np.uint8).reshape(blocks, 16)
    blk_t = T.tensor(blk, dtype=T.uint8, device=_dev()) if is_t else blk
    key_t = T.tensor(list(key), dtype=T.uint8, device=_dev()) if is_t else np.frombuffer(key, dtype=np.uint8)
    out = G.aes192_ecb(blk_t, key_t)
    out = G.from_dev(out).tobytes()
    return out[:nbytes]


# =====================================================================
# 7. 批量模幂（小指数筛选）
# =====================================================================
def modexp_batch(bases, exp, mod):
    """批量 m^e mod N（int64 范围内，Python 循环 + numpy 加速）。"""
    b = np.asarray(bases, dtype=object)
    return np.array([pow(int(x), exp, mod) for x in b], dtype=object)


# =====================================================================
# 子命令
# =====================================================================
def cmd_selftest(a):
    print(f'[backend] {G.BACKEND} {G.DEV}')
    results = []

    def check(name, ok, detail=''):
        """记录一项 PASS/FAIL（输出文本保持与原版一致），最后据结果决定返回码。"""
        results.append((name, bool(ok)))
        print(f'{name} {"PASS" if ok else "FAIL"}{detail}')

    # --- ChaCha20 RFC 8439 §2.4.2 测试向量 ---
    key = bytes(range(32))
    nonce = bytes.fromhex('000000090000004a00000000')      # 注意：RFC 向量是 00 00 00 09 00 00 00 4a ...
    ks = bytes(chacha20_keystream(key, nonce, counter=1, nbytes=64)[0])
    exp = ('10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e'
           'd2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e')
    check('[ChaCha20 RFC8439]', ks.hex() == exp)

    # --- ChaCha20 原版变体（8B nonce + 64bit counter；对照值由 pycryptodome 独立生成）---
    ks0 = bytes(chacha20_keystream(key, bytes(8), counter=0, nbytes=64, rfc7539=False)[0])
    exp0 = ('39fd2b7dd9c5196a8dbd0377b8dc4a498a35d86fbcde6accb2cc7d4cd8ea2492'
            '2b23cce7a26023ab3f0eef693ac87f64258235eab1f7a32dc22762a0485b410c')
    check('[ChaCha20 原版变体]', ks0.hex() == exp0)

    # --- Poly1305 RFC 8439 §2.5.2 ---
    pk = bytes.fromhex('85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b')
    tag = poly1305(pk, b'Cryptographic Forum Research Group')
    check('[Poly1305 RFC8439]', tag.hex() == 'a8061dc1305136c6c22b8baf0c0127a9')

    # --- ChaCha20-Poly1305 AEAD RFC 8439 §2.8.2（CT 60B + tag16B；校验多块密钥流布局）---
    aead_key = bytes.fromhex('808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f')
    aead_nonce = bytes.fromhex('070000004041424344454647')
    aead_aad = bytes.fromhex('50515253c0c1c2c3c4c5c6c7')
    aead_pt = bytes.fromhex(
        '4c616469657320616e642047656e746c656d656e206f662074686520636c617373206f66'
        '202739393a204966204920636f756c64206f6666657220796f75206f6e6c79206f6e6520'
        '74697020666f7220746865206675747572652c2073756e73637265656e20776f756c6420'
        '62652069742e')
    aead_ct = ('d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6'
               '3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36'
               '92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc'
               '3ff4def08e4b7a9de576d26586cec64b6116')
    aead_tag = '1ae10b594f09e26a7e902ecbd0600691'
    otk = bytes(chacha20_keystream(aead_key, aead_nonce, counter=0, nbytes=32)[0])
    body = bytes(chacha20_keystream(aead_key, aead_nonce, counter=1, nbytes=len(aead_pt))[0])
    our_ct = bytes(x ^ y for x, y in zip(aead_pt, body))
    our_tag = poly1305_tag(otk, aead_aad, our_ct)
    check('[ChaCha20-Poly1305 AEAD]', our_ct.hex() == aead_ct and our_tag.hex() == aead_tag)

    # --- RS(63,31)：合法码字 syndrome 全零 + 破坏 1 字节后非零 ---
    good = bytes.fromhex('504d333145c89c44bdc8b495f33d47423739807b62151310db48dc95c37f79'
                         'a8ea2397662150254460bc05d349ac9f7d2a600cbca3b4e9d75f923bf07b2941')
    if len(good) == 63:
        ok3, sh = rs63_check(good)
        bad = bytearray(good); bad[5] ^= 0x01
        ok4, _ = rs63_check(bytes(bad))
        check('[RS(63,31) 合法码字]', ok3, f' (syndrome={sh[:16]}…)')
        check('[RS(63,31) 错误码字]', not ok4, ' (应为非零)')
    else:
        print('[RS(63,31)] 跳过（题包已清理）')

    # --- CRC32 已知值 ---
    c = int(crc32_batch(np.frombuffer(b'123456789', dtype=np.uint8).reshape(1, -1))[0])
    check('[CRC32 "123456789"]', c == 0xCBF43926, f' (got {c:08x})')

    # --- AES-192 CTR：NIST SP800-38A F.5.3 首块 + 多块计数器布局 ---
    nist_key = bytes.fromhex('8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b')
    nist_nonce = bytes.fromhex('f0f1f2f3f4f5f6f7')
    nist_ctr = int.from_bytes(bytes.fromhex('f8f9fafbfcfdfeff'), 'little')
    ks_nist = aesctr_keystream(nist_key, nist_nonce, 16, nist_ctr)
    ks_multi = aesctr_keystream(bytes(range(24)), bytes(8), 48, 0)
    ok_aes = (ks_nist.hex() == '717d2dc639128334a6167a488ded7921' and
              ks_multi.hex() == '916251821c73a522c396d627380196072ff04883eaa73662'
                                'e711ccd6ee7544beecb2a87e320d4718bd748842c6dccac2')
    check('[AES-192 CTR]', ok_aes)

    # --- xorbrute 经典题 ---
    ct = bytes.fromhex('1b37373331363f78151b7f2b783431333d78397828372d363c78373e783a393b3736')
    r = xorbrute(ct, max_len=1, top=1)
    okx = bool(r) and b"Cooking MC's like a pound of bacon" in r[0][2]
    check('[xorbrute]', okx, f'  key={r[0][1]} → {r[0][2][:28]!r}' if r else '')

    # --- 性能 ---
    print('\n[性能]')
    keys = np.random.randint(0, 256, (1 << 14, 32), dtype=np.uint8)
    t0 = time.time(); chacha20_keystream(keys, nonce, 0, 64); dt = time.time() - t0
    print(f'  ChaCha20 : {keys.shape[0]/dt/1e3:8.1f} K keys/s (×64B)')
    cws = np.random.randint(0, 256, (1 << 13, 63), dtype=np.uint8)
    t0 = time.time(); rs63_syndromes(cws); dt = time.time() - t0
    print(f'  RS63     : {cws.shape[0]/dt/1e3:8.1f} K codewords/s')
    d = np.random.randint(0, 256, (1 << 15, 64), dtype=np.uint8)
    t0 = time.time(); crc32_batch(d); dt = time.time() - t0
    print(f'  CRC32    : {d.shape[0]/dt/1e3:8.1f} K /s (×64B)')

    nfail = sum(1 for _, ok in results if not ok)
    return 1 if nfail else 0


def cmd_chacha(a):
    target = _parse_hex(a.target_hex, '--target-hex')
    if not target:
        raise ValueError('--target-hex 不能为空（至少 1 字节）')
    nonce = _parse_hex(a.nonce_hex, '--nonce-hex')
    if len(nonce) not in (8, 12):
        raise ValueError(f'--nonce-hex 需 8 字节（原版）或 12 字节（rfc7539），收到 {len(nonce)} 字节')
    if a.key_len != 32:
        raise ValueError(f'ChaCha20 密钥固定 32 字节，--key-len 只支持 32，收到 {a.key_len}')
    rfc7539 = (len(nonce) == 12)
    if a.keys_file:
        keys = _load_keys_file(a.keys_file, 32)
        print(f'[scan] {len(keys)} 个候选密钥 × {len(target)}B 目标 '
              f'（{"rfc7539 12B nonce" if rfc7539 else "原版 8B nonce + 64bit 计数器"}）')
        t0 = time.time()
        out = chacha20_keystream(keys, nonce, a.counter, len(target), rfc7539)
        eq = (out == np.frombuffer(target, dtype=np.uint8)).all(1)
        dt = time.time() - t0
        print(f'[done] {len(keys)/dt/1e3:.1f} K/s')
        for i in np.nonzero(eq)[0]:
            print(f'  ★ 命中 #{i} key={keys[i].tobytes().hex()}')
        if not eq.any():
            best = None
            target_t = np.frombuffer(target, dtype=np.uint8)
            for i in range(0, len(keys), 4096):
                cand = (G.xp().tensor(out[i:i+4096], device=_dev()) if _is_torch()
                        else out[i:i+4096])
                sc = G.from_dev(G.bit_agreement(
                    cand, (G.xp().tensor(list(target), dtype=G.xp().uint8, device=_dev())
                           if _is_torch() else target_t)))
                sc = np.atleast_1d(sc)
                j = int(np.argmax(sc))
                if best is None or sc[j] > best[0]:
                    best = (int(sc[j]), i + j)
            print(f'  未命中；最佳 {best[0]}/{len(target)*8} bit (候选 #{best[1]})')
    elif a.brute:
        print('[brute] 随机密钥搜索（仅为演示吞吐；真实场景用 --keys-file 或直接生成候选）')
        rng = np.random.default_rng(1)
        batch = int(a.chunk)
        tries = int(a.tries)
        if batch < 1 or tries < 1:
            raise ValueError(f'--chunk/--tries 必须 >= 1（chunk={batch}, tries={tries}）')
        total, found = 0, False
        t0 = time.time()
        target_t = np.frombuffer(target, dtype=np.uint8)
        while total < tries:                       # 注意：这里永不全量扫 2^256，仅演示吞吐
            n = min(batch, tries - total)
            keys = rng.integers(0, 256, (n, 32), dtype=np.uint8)
            out = chacha20_keystream(keys, nonce, a.counter, len(target), rfc7539)
            if (out == target_t).all(1).any():
                found = True
                total += n
                break
            total += n
            if total % (batch * 10) == 0:
                print(f'  {total:,}  键/{(time.time()-t0):.1f}s', flush=True)
        dt = time.time() - t0
        print(f'[done] {total:,} 键 / {dt:.2f}s = {total/dt/1e3:.1f} K/s  {"命中" if found else "未命中"}')
    else:
        raise ValueError('chacha 需要 --keys-file（候选密钥扫描）或 --brute（随机搜索演示）')


def cmd_poly1305(a):
    """Poly1305 / ChaCha20-Poly1305 tag 计算与校验。"""
    key = _parse_hex(a.key_hex, '--key-hex')
    if len(key) != 32:
        raise ValueError(f'Poly1305 子密钥必须 32 字节，收到 {len(key)} 字节')
    msg = _parse_hex(a.msg_hex, '--msg-hex')
    aad = _parse_hex(a.aad_hex, '--aad-hex') if a.aad_hex else b''
    otk = key
    if a.nonce_hex is not None:
        if a.mode != 'aead':
            raise ValueError('--nonce-hex 只在 --mode aead 下使用')
        nonce = _parse_hex(a.nonce_hex, '--nonce-hex')
        if len(nonce) != 12:
            raise ValueError(f'AEAD nonce 需要 12 字节，收到 {len(nonce)} 字节')
        # 方便用法：给 ChaCha20 主密钥 + nonce，自动派生 Poly1305 子密钥（计数器 0 的前 32B）
        otk = bytes(chacha20_keystream(key, nonce, counter=0, nbytes=32)[0])
        print(f'[poly1305] 派生 otk = {otk.hex()}')
    if a.mode == 'aead':
        # aead 模式下 msg 按「密文」处理：tag 覆盖 pad(aad)||pad(ct)||len(aad)||len(ct)
        tag = poly1305_tag(otk, aad, msg)
    else:
        tag = poly1305(otk, msg)
    print(f'[poly1305/{a.mode}] tag = {tag.hex()}')
    if a.expect_hex:
        want = _parse_hex(a.expect_hex, '--expect-hex')
        if len(want) != 16:
            raise ValueError(f'--expect-hex 应为 16 字节 tag，收到 {len(want)} 字节')
        ok = tag == want
        print(f'  期望 {want.hex()}  →  {"匹配 ✅" if ok else "不匹配 ❌"}')
        return 0 if ok else 1          # 校验命令：不匹配时退出码 1，脚本可感知
    return tag


def cmd_aesctr(a):
    """AES-192-CTR 密钥流 / 加解密。"""
    key = _parse_hex(a.key_hex, '--key-hex')
    nonce = _parse_hex(a.nonce_hex, '--nonce-hex')
    if len(key) != 24:
        raise ValueError(f'AES-192 密钥必须 24 字节，收到 {len(key)} 字节')
    if len(nonce) != 8:
        raise ValueError(f'CTR nonce 必须 8 字节，收到 {len(nonce)} 字节')
    if a.mode == 'ks':
        if a.length < 0:
            raise ValueError(f'--length 不能为负: {a.length}')
        ks = aesctr_keystream(key, nonce, a.length, a.counter)
        print(f'[aesctr] keystream({a.length}B) = {ks.hex()}')
        return ks
    data = _parse_hex(a.data_hex, '--data-hex') if a.data_hex else b''
    ks = aesctr_keystream(key, nonce, len(data), a.counter)
    out = bytes(x ^ y for x, y in zip(data, ks))
    print(f'[aesctr/{a.mode}] {out.hex()}')
    if a.out_text:
        print(f'  text = {out!r}')
    return out


def cmd_rs63(a):
    code = _parse_hex(a.code_hex, '--code-hex')
    if not 1 <= len(code) <= 255:
        raise ValueError(f'--code-hex 码字长度必须在 1~255（GF(256) 上限），收到 {len(code)}')
    # --fcr/--order 都可单独指定：未指定的维度仍自动尝试（都不填 = 原版 4 种约定全试）
    orders = ('msb', 'lsb') if a.order == 'auto' else (a.order,)
    fcrs = (0, 1) if a.fcr is None else (a.fcr,)
    hit = None
    for o in orders:
        for f in fcrs:
            if rs63_check(code, nsym=a.nsym, fcr=f, order=o, auto=False)[0]:
                hit = (o, f)
                break
        if hit:
            break
    if hit:
        print(f'[RS(63,31)] 长度 {len(code)}  合法=True  '
              f'(order={hit[0]} fcr={hit[1]} nsym={a.nsym})')
        return
    o0, f0 = orders[0], fcrs[0]
    _, sh = rs63_check(code, nsym=a.nsym, fcr=f0, order=o0, auto=False)
    print(f'[RS(63,31)] 长度 {len(code)}  合法=False')
    print(f'  syndrome = {sh}  (order={o0} fcr={f0} nsym={a.nsym})')
    if not a.locate:
        return
    print('  尝试单字节定位…')
    for o in orders:
        for f in fcrs:
            r = rs63_locate(code, nsym=a.nsym, fcr=f, order=o)
            if r:
                pos, delta = r
                print(f'  ★ 修正（order={o} fcr={f}）：pos={pos} '
                      f'{code[pos]:#04x} → {code[pos] ^ delta:#04x}')
                return
    print('  单字节无法修正（可能是合法码字或错更多字节）')


def cmd_crc32(a):
    data = _parse_hex(a.data_hex, '--data-hex')
    if a.target_hex is None:
        c = int(crc32_batch(np.frombuffer(data, dtype=np.uint8).reshape(1, -1))[0])
        print(f'[crc32] {c:08x}')
        return
    target = _parse_hex(a.target_hex, '--target-hex')
    if not 1 <= len(target) <= 4:
        raise ValueError(f'--target-hex 需 1~4 字节的 CRC 值，收到 {len(target)} 字节')
    want = int.from_bytes(target, 'big')
    print(f'[crc32] 目标 {want:08x}，在原数据尾部爆破 1~2 字节')
    # 增量算 CRC：先算 data 的原始状态，再对「追加 1/2 字节」做向量化扫描。
    # （旧实现把 1 字节变体塞进 len+2 的数组里，末尾多补一个 0x00，导致单字节搜索永远错。）
    tab = _crc32_tab().astype(np.int64)
    st = np.int64(crc32_state(data))
    b = np.arange(256, dtype=np.int64)
    st1 = tab[(st ^ b) & 0xFF] ^ (st >> 8)                   # 追加 1 字节后的状态
    crc1 = (st1 ^ 0xFFFFFFFF) & 0xFFFFFFFF
    nhit = 0
    for i in np.nonzero(crc1 == want)[0]:
        print(f'  ★ 命中：{data.hex()}{int(i):02x}')
        nhit += 1
    st2 = tab[(st1[:, None] ^ b[None, :]) & 0xFF] ^ (st1[:, None] >> 8)
    crc2 = (st2 ^ 0xFFFFFFFF) & 0xFFFFFFFF
    for i, j in np.argwhere(crc2 == want):
        print(f'  ★ 命中：{data.hex()}{int(i):02x}{int(j):02x}')
        nhit += 1
    if not nhit:
        print('  未命中')
    print(f'  扫描 {256 + 65536:,} 变体完成')


def cmd_xorbrute(a):
    data = _parse_hex(a.data_hex, '--data-hex')
    if not data:
        raise ValueError('--data-hex 不能为空')
    t0 = time.time()
    res = xorbrute(data, max_len=a.max_len, top=a.top)
    print(f'[xorbrute] {len(data)}B, key 长度 ≤{a.max_len}, 用时 {time.time()-t0:.2f}s')
    for sc, kh, pt in res:
        print(f'  score={sc:5d} key={kh:<12} → {pt[:60]!r}')


def main():
    ap = argparse.ArgumentParser(
        description='ctf_tools — CTF 密码学内核（完整说明见 CLI_REFERENCE.md）')
    ap.add_argument('--backend', default='auto', choices=['auto', 'torch', 'numpy'],
                    help='计算后端：auto=torch-cuda→torch-cpu→numpy（默认 auto）')
    ap.add_argument('--chunk', type=int, default=1 << 14,
                    help='每批候选数（默认 16384，chacha --brute 用）')
    sub = ap.add_subparsers(dest='cmd', required=True,
                            metavar='{selftest,chacha,poly1305,rs63,crc32,xorbrute,aesctr}')
    sub.add_parser('selftest', help='已知向量自测（9 项；全 PASS 退出码 0）')

    p = sub.add_parser('chacha', help='ChaCha20 密钥流生成 / 候选密钥扫描')
    p.add_argument('--target-hex', required=True, help='目标密钥流，hex（>=1 字节）')
    p.add_argument('--nonce-hex', default='000000000000004a00000000',
                   help='12 字节 = RFC7539 变体；8 字节 = 原版（64bit 计数器）；默认 12B')
    p.add_argument('--counter', type=int, default=0, help='起始计数器，默认 0')
    p.add_argument('--keys-file', default=None,
                   help='候选密钥文件：每行一个 32B hex（容忍 0x/#注释/空行）')
    p.add_argument('--brute', action='store_true', help='随机密钥搜索演示（配合 --tries/--chunk）')
    p.add_argument('--key-len', type=int, default=32, help='ChaCha20 密钥固定 32 字节（保留参数）')
    p.add_argument('--tries', type=int, default=1 << 20, help='--brute 最多尝试密钥数，默认 1048576')

    p = sub.add_parser('rs63', help='GF(256) RS(63,31) syndrome 校验 / 单字节纠错')
    p.add_argument('--code-hex', required=True, help='码字 hex，长度 1~255')
    p.add_argument('--nsym', type=int, default=32, help='校验符号数，默认 32')
    p.add_argument('--fcr', type=int, default=None,
                   help='固定 first consecutive root；不填 = 自动试 {0,1}')
    p.add_argument('--order', default='auto', choices=['auto', 'msb', 'lsb'],
                   help='系数顺序；auto = 自动试两种（默认）')
    p.add_argument('--locate', action='store_true', help='非法码字时尝试单字节纠错定位')

    p = sub.add_parser('crc32', help='CRC32 计算 / 尾部 1~2 字节爆破匹配')
    p.add_argument('--data-hex', required=True, help='原始数据 hex（任意长度）')
    p.add_argument('--target-hex', default=None, help='目标 CRC 值（1~4 字节，按大端解释）')

    p = sub.add_parser('xorbrute', help='重复 XOR 爆破（1~3 字节密钥）')
    p.add_argument('--data-hex', required=True, help='密文 hex，>=1 字节')
    p.add_argument('--max-len', type=int, default=3, help='密钥长度上限 1~3（默认 3；4 字节拒绝）')
    p.add_argument('--top', type=int, default=10, help='输出候选数，默认 10')

    p = sub.add_parser('poly1305', help='Poly1305 / ChaCha20-Poly1305 tag 计算与校验')
    p.add_argument('--key-hex', required=True,
                   help='32 字节 Poly1305 子密钥（otk||r）；给 --nonce-hex 时改为 ChaCha20 主密钥')
    p.add_argument('--msg-hex', required=True, help='消息 hex；aead 模式下按密文处理')
    p.add_argument('--aad-hex', default='', help='AEAD 附加数据 hex（默认空）')
    p.add_argument('--nonce-hex', default=None,
                   help='可选：12 字节 ChaCha20 nonce，自动由主密钥派生 otk（仅 aead 模式）')
    p.add_argument('--mode', default='plain', choices=['plain', 'aead'],
                   help='plain=原始 Poly1305（默认）；aead=ChaCha20-Poly1305 tag')
    p.add_argument('--expect-hex', default=None, help='期望 tag（16B hex）；不匹配退出码 1')

    p = sub.add_parser('aesctr', help='AES-192-CTR 密钥流 / 加解密')
    p.add_argument('--key-hex', required=True, help='24 字节 AES-192 密钥')
    p.add_argument('--nonce-hex', required=True, help='8 字节 nonce（计数器块 = nonce||counter64 小端）')
    p.add_argument('--mode', default='dec', choices=['ks', 'enc', 'dec'],
                   help='ks=输出密钥流；enc/dec=对 --data-hex 异或（默认 dec）')
    p.add_argument('--data-hex', default='', help='enc/dec 模式的输入数据 hex')
    p.add_argument('--length', type=int, default=64, help='ks 模式密钥流字节数，默认 64')
    p.add_argument('--counter', type=int, default=0, help='起始 64bit 计数器，默认 0')
    p.add_argument('--out-text', action='store_true', help='额外打印 text = repr(out)')

    a = ap.parse_args()
    name, dev = G.init_backend(a.backend)
    print(f'[backend] {name} device={dev}\n')
    handler = {'selftest': cmd_selftest, 'chacha': cmd_chacha, 'rs63': cmd_rs63,
               'crc32': cmd_crc32, 'xorbrute': cmd_xorbrute,
               'poly1305': cmd_poly1305, 'aesctr': cmd_aesctr}[a.cmd]
    try:
        rc = handler(a)
    except ValueError as e:                     # 参数/输入错误：干净报错，不再抛 traceback
        print(f'[错误] {e}', file=sys.stderr)
        return 2
    return 0 if not isinstance(rc, int) else rc


if __name__ == '__main__':
    sys.exit(main())
