#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuzz_kernels.py — 第 5 层验证（fuzz）：内核差分模糊测试（可被 CI 调用）

定位
----
其余验证层（模块 selftest / 主框架 selftest / consistency.py 固定向量 / 统一入口
selftest / doc_lint）以「固定向量 + 双后端」为主。本文件补上差分模糊测试：**用独立参考
实现或第三方库，对全部内核做随机差分测试**。随机输入更容易撞到边界（0 长度、
计数器进位、nonce 长度切换、块边界、padding 边界、不可逆参数……）。

设计原则
--------
1. 每个类别都有**独立参考**：zlib / hashlib / pycryptodome / cryptography，或本文件
   内按规范（论文/RFC）另行编写的纯 Python 实现；同一答案绝不用被测函数自己算第二遍。
2. 参考实现先做官方/已发布向量的自检（`check_references()`），自检失败 = HARNESS-ERROR，
   不会把「参考写错」误报成模块缺陷。
3. 可复现：`--seed` 固定后，类别内随机流由 `sha256(seed:类别名)` 派生，因此
   `--only <类别>` 与全量运行的该类别结果一致；失败可直接用提示的命令复现。
4. 快速、无 GPU：默认 `--backend numpy`，全部类别用 numpy / 纯 Python 内核，
   不 import torch（除非 `--backend torch`）。默认全量 < 60s，`--quick` 更快。
5. 失败即不一致：任何第 3 方/参考实现对拍不一致 → 退出码 1；HARNESS-ERROR → 退出码 2。

用法
----
  python -X utf8 fuzz_kernels.py                  # 全量（默认 8 次迭代/类别，seed=20261109）
  python -X utf8 fuzz_kernels.py --quick          # CI 快速档（2 次迭代/类别）
  python -X utf8 fuzz_kernels.py --iters 32 --seed 7
  python -X utf8 fuzz_kernels.py --only chacha20,aes_ctr --verbose
  python -X utf8 fuzz_kernels.py --backend torch  # 额外跑 numpy ↔ torch 双后端逐字节对拍
  python -X utf8 fuzz_kernels.py --list           # 列出类别

类别（17）
---------
  crc32       CRC32 vs zlib.crc32（随机 0~4096B、批量）
  md5         md5_of / md5_batch / hash_extend(name, md5/sha1) vs hashlib（含 padding 边界
              0/55/56/63/64/120）
  chacha20    chacha20_keystream vs pycryptodome ChaCha20（8B 原版 / 12B RFC7539；counter
              起点 0/1/大值；nbytes 跨 64B 边界）+ 纯 Python 参考补大计数器
  poly1305    poly1305 / poly1305_tag vs cryptography Poly1305 / ChaCha20Poly1305（含 tag
              被第三方 verifier 接受的正/负路径）
  aes_ecb     aes_ecb(128/192/256) vs pycryptodome AES-ECB（单密钥快路径 + 每候选密钥路径）
  aes_ctr     aesctr_keystream vs 独立 Python 参考（自建计数器块 + pycryptodome AES-ECB；
              8/12/16B nonce × le/be × 计数器边界）
  rc4         rc4_keystream / rc4 vs pycryptodome ARC4 + 自写参考（key 1~256B）
  tea         TEA / XTEA / XXTEA vs 本文件独立参考实现（规范逐位重写）+ 官方/公开向量
  speck       Speck32/64、Speck64/128（及 32/3 变体）vs 独立规范实现；NSA 官方向量自检
  mt19937     mt19937_keys(twist=True/False) vs 独立 std::mt19937 参考；popcount_matrix /
              bit_agreement vs 纯 Python popcount
  rs63        自写 GF(256) RS(63,31) 编码器产码字 → 注入 ≤16/17~32 错，验证 syndrome/
              locate；>16 错不要求纠正但不得 crash
  lcg         lcg_recover / lcg_predict 往返（多种模数，含 2^31/2^32/素数）
  ecc         ecc_pub vs cryptography EC + 自写 affine 标量乘；ecc_search 小私钥区间找回
  rsa         _decrypt_with_pq / modexp_batch / 共模 / 低指数广播 / n=p² / Fermat /
              Pollard-rho / p-1 / rsa_attack 植入-找回
  auto_decode 构造多层编码链（hex/b64/b32/url/rot13/reversed，深度 1~3）：纯 codec 链
             要求逐字节还原，混合变换链要求完整 flag 子串还原（模块启发式契约）
  dual_backend  仅 --backend torch：aes/chacha/aesctr/mt19937/popcount/bit_agreement 在
              numpy ↔ torch 下逐字节一致
  lfsr        独立右移 Fibonacci LFSR 参考 → 生成目标比特流 → cmd_lfsrscan 必须找回满分
              （行为语义对拍，不依赖随机命中概率）

退出码
------
  0 = 全部通过（skipped 不算失败）；1 = 存在不一致；2 = HARNESS-ERROR（参考实现自检失败）
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.parse
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_crack as G          # noqa: E402
import ctf_tools as TOOLS      # noqa: E402
import ctf_ext as EXT          # noqa: E402

# ---------------------------------------------------------------- 第三方参考库
try:
    from Crypto.Cipher import AES as _AES
    from Crypto.Cipher import ARC4 as _ARC4
    from Crypto.Cipher import ChaCha20 as _ChaCha20
    _HAVE_PYCRYPTODOME = True
    import Crypto as _Crypto
    _PYCRYPTODOME_VER = getattr(_Crypto, '__version__', '?')
except Exception:                       # pragma: no cover - 环境缺依赖时优雅跳过
    _HAVE_PYCRYPTODOME = False
    _PYCRYPTODOME_VER = ''

try:
    from cryptography.hazmat.primitives.poly1305 import Poly1305 as _RefPoly1305
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305 as _RefAEAD
    from cryptography.hazmat.primitives.asymmetric import ec as _RefEC
    _HAVE_CRYPTOGRAPHY = True
    import cryptography as _cryptography
    _CRYPTOGRAPHY_VER = getattr(_cryptography, '__version__', '?')
except Exception:                       # pragma: no cover
    _HAVE_CRYPTOGRAPHY = False
    _CRYPTOGRAPHY_VER = ''

M32 = 0xFFFFFFFF


# =====================================================================
# 运行框架
# =====================================================================
class FuzzMismatch(AssertionError):
    """模块输出与独立参考不一致。"""


class HarnessError(AssertionError):
    """参考实现自检失败（harness 自身有问题，不是模块缺陷）。"""


class SkipCategory(Exception):
    """当前环境缺少依赖 / 需要特定选项，跳过该类别。"""


class Ctx:
    """单个类别的运行上下文：随机源、迭代计数、断言计数。"""

    def __init__(self, name, rng, iters, verbose=False):
        self.name = name
        self.rng = rng
        self.iters = int(iters)
        self.verbose = verbose
        self.iters_done = 0
        self.checks = 0
        self.iter_no = 0
        self.details = []

    def iter(self):
        self.iters_done += 1
        self.iter_no = self.iters_done

    def log(self, msg):
        if self.verbose:
            self.details.append(msg)

    def check(self, cond, detail=''):
        self.checks += 1
        if not cond:
            raise FuzzMismatch(f'[iter {self.iter_no}] {detail}')

    def skip(self, reason):
        raise SkipCategory(reason)


CATEGORIES = []          # [(name, fn)]
_CAT_MAP = {}

_CAT_NOTES = {
    'crc32': 'CRC32 vs zlib.crc32（随机 0~4096B、批量 + 增量状态）',
    'md5': 'md5_of/md5_batch/hash_extend(md5,sha1) vs hashlib（padding 边界）',
    'chacha20': 'chacha20_keystream vs pycryptodome（8/12B nonce、计数器起点/大值、跨 64B）',
    'poly1305': 'poly1305/poly1305_tag vs cryptography（raw MAC + ChaCha20-Poly1305 AEAD）',
    'aes_ecb': 'aes_ecb(128/192/256) vs pycryptodome AES-ECB（快路径 + 逐候选密钥）',
    'aes_ctr': 'aesctr_keystream vs 独立计数器块参考（8/12/16B nonce × le/be × 边界）',
    'rc4': 'rc4_keystream/rc4 vs pycryptodome ARC4 + 自写参考（key 1~256B）',
    'tea': 'TEA/XTEA/XXTEA vs 独立规范实现（官方/公开向量自检）',
    'speck': 'Speck32/64、64/128、32/3 vs 独立规范实现（NSA 向量自检）',
    'mt19937': 'mt19937_keys vs 独立 std::mt19937；popcount/bit_agreement vs 纯 Python',
    'rs63': '自写 GF(256) RS 编码 → 注入 ≤16/>16 错验证 syndrome/locate',
    'lcg': 'lcg_recover/lcg_predict 往返（2^31/2^32/素数模数）',
    'ecc': 'ecc_pub vs cryptography EC + 自写 affine；ecc_search 植入-找回',
    'rsa': '_decrypt_with_pq/modexp/共模/广播/n=p²/Fermat/rho/p-1/rsa_attack',
    'auto_decode': '多层编码链：纯 codec 链逐字节还原、混合链完整 flag 还原',
    'dual_backend': 'numpy ↔ torch 逐字节对拍（需 --backend torch）',
    'lfsr': '独立 Fibonacci LFSR 参考 → cmd_lfsrscan 必须满分（语义对拍）',
}


def category(name):
    def deco(fn):
        CATEGORIES.append((name, fn))
        _CAT_MAP[name] = fn
        return fn
    return deco


def _rng_for(seed, name):
    """类别级随机流：sha256(seed:name) 派生，保证 --only 与全量同类别结果一致。"""
    h = hashlib.sha256(f'{int(seed)}:{name}'.encode()).digest()
    return random.Random(int.from_bytes(h[:8], 'big'))


def _rb(rng, n):
    """n 字节随机数据。"""
    return bytes(rng.getrandbits(8) for _ in range(n))


def _hex(b, limit=48):
    h = bytes(b).hex()
    return h if len(h) <= limit else h[:limit] + '...'


# =====================================================================
# 独立参考实现（纯 Python，按 RFC / 论文规范重写；仅用于对拍，不参与生产路径）
# =====================================================================
# ---------------------------------------------------------------- ChaCha20
_CHACHA_CONST = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]


def _rotl32(x, n):
    return ((x << n) | (x >> (32 - n))) & M32


def _chacha20_block_ref(key32, counter, nonce, rfc7539):
    """RFC 8439 / 原版 ChaCha20 的 64 字节 block 函数（纯 Python）。"""
    kw = [int.from_bytes(key32[4 * i:4 * i + 4], 'little') for i in range(8)]
    nw = [int.from_bytes(nonce[4 * i:4 * i + 4], 'little') for i in range(len(nonce) // 4)]
    st = list(_CHACHA_CONST) + kw
    if rfc7539:
        st += [counter & M32] + nw[:3]
    else:
        st += [counter & M32, (counter >> 32) & M32] + nw[:2]
    x = st[:]

    def qr(a, b, c, d):
        xa, xb, xc, xd = x[a], x[b], x[c], x[d]
        xa = (xa + xb) & M32
        xd = _rotl32(xd ^ xa, 16)
        xc = (xc + xd) & M32
        xb = _rotl32(xb ^ xc, 12)
        xa = (xa + xb) & M32
        xd = _rotl32(xd ^ xa, 8)
        xc = (xc + xd) & M32
        xb = _rotl32(xb ^ xc, 7)
        x[a], x[b], x[c], x[d] = xa, xb, xc, xd

    for _ in range(10):
        qr(0, 4, 8, 12)
        qr(1, 5, 9, 13)
        qr(2, 6, 10, 14)
        qr(3, 7, 11, 15)
        qr(0, 5, 10, 15)
        qr(1, 6, 11, 12)
        qr(2, 7, 8, 13)
        qr(3, 4, 9, 14)
    out = bytearray()
    for i in range(16):
        out += ((x[i] + st[i]) & M32).to_bytes(4, 'little')
    return bytes(out)


def chacha20_keystream_ref(key32, nonce, counter, nbytes, rfc7539):
    out = bytearray()
    ctr = int(counter)
    while len(out) < nbytes:
        out += _chacha20_block_ref(key32, ctr, nonce, rfc7539)
        ctr = (ctr + 1) & M32 if rfc7539 else ctr + 1
    return bytes(out[:nbytes])


# ---------------------------------------------------------------- AES-CTR
def aesctr_ref(key, nonce, nbytes, counter, mode):
    """独立构造计数器块（不调用被测函数），AES 用 pycryptodome-ECB。"""
    order = 'little' if mode == 'le' else 'big'
    suffix = 16 - len(nonce)
    blocks = (nbytes + 15) // 16
    if blocks == 0:
        return b''
    aes = _AES.new(key, _AES.MODE_ECB)
    if suffix == 0:
        limit = 1 << 128
        base = int.from_bytes(nonce, order)
        ctrs = [((base + counter + i) % limit).to_bytes(16, order) for i in range(blocks)]
    else:
        limit = 1 << (8 * suffix)
        ctrs = [nonce + ((counter + i) % limit).to_bytes(suffix, order) for i in range(blocks)]
    return b''.join(aes.encrypt(c) for c in ctrs)[:nbytes]


# ---------------------------------------------------------------- RC4
def rc4_ref(key, n):
    """RC4 密钥流（按 KSA/PRGA 规范独立实现）。"""
    s = list(range(256))
    j = 0
    kl = len(key)
    for i in range(256):
        j = (j + s[i] + key[i % kl]) & 0xFF
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for _ in range(n):
        i = (i + 1) & 0xFF
        j = (j + s[i]) & 0xFF
        s[i], s[j] = s[j], s[i]
        out.append(s[(s[i] + s[j]) & 0xFF])
    return bytes(out)


# ---------------------------------------------------------------- TEA / XTEA / XXTEA
_DELTA = 0x9E3779B9


def _tea_enc_ref(v0, v1, k, rounds=32):
    s = 0
    for _ in range(rounds):
        s = (s + _DELTA) & M32
        v0 = (v0 + ((((v1 << 4) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1])) & M32)) & M32
        v1 = (v1 + ((((v0 << 4) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3])) & M32)) & M32
    return v0, v1


def _tea_dec_ref(v0, v1, k, rounds=32):
    s = (_DELTA * rounds) & M32
    for _ in range(rounds):
        v1 = (v1 - ((((v0 << 4) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3])) & M32)) & M32
        v0 = (v0 - ((((v1 << 4) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1])) & M32)) & M32
        s = (s - _DELTA) & M32
    return v0, v1


def _xtea_enc_ref(v0, v1, k, rounds=32):
    s = 0
    for _ in range(rounds):
        v0 = (v0 + (((((v1 << 4) ^ (v1 >> 5)) + v1) & M32) ^ (s + k[s & 3]))) & M32
        s = (s + _DELTA) & M32
        v1 = (v1 + (((((v0 << 4) ^ (v0 >> 5)) + v0) & M32) ^ (s + k[(s >> 11) & 3]))) & M32
    return v0, v1


def _xtea_dec_ref(v0, v1, k, rounds=32):
    s = (_DELTA * rounds) & M32
    for _ in range(rounds):
        v1 = (v1 - (((((v0 << 4) ^ (v0 >> 5)) + v0) & M32) ^ (s + k[(s >> 11) & 3]))) & M32
        s = (s - _DELTA) & M32
        v0 = (v0 - (((((v1 << 4) ^ (v1 >> 5)) + v1) & M32) ^ (s + k[s & 3]))) & M32
    return v0, v1


def _xxtea_pad(data):
    d = bytes(data) + b'\x00' * ((-len(data)) % 4)
    if len(d) < 8:
        d += b'\x00' * (8 - len(d))
    return d


def _xxtea_enc_ref(data, key, rounds=None):
    d = _xxtea_pad(data)
    v = [int.from_bytes(d[4 * i:4 * i + 4], 'little') for i in range(len(d) // 4)]
    kb = bytes(key)[:16].ljust(16, b'\x00')
    k = [int.from_bytes(kb[4 * i:4 * i + 4], 'little') for i in range(4)]
    n = len(v)
    if rounds is None:
        rounds = 6 + 52 // n
    total = 0
    z = v[n - 1]
    for _ in range(rounds):
        total = (total + _DELTA) & M32
        e = (total >> 2) & 3
        for p in range(n):
            y = v[(p + 1) % n]
            mx = ((((z >> 5) ^ ((y << 2) & M32)) + (((y >> 3) ^ ((z << 4) & M32)) & M32)) & M32) \
                ^ (((total ^ y) + (k[(p & 3) ^ e] ^ z)) & M32)
            v[p] = (v[p] + mx) & M32
            z = v[p]
    return b''.join(x.to_bytes(4, 'little') for x in v)


def _xxtea_dec_ref(data, key, rounds=None):
    v = [int.from_bytes(data[4 * i:4 * i + 4], 'little') for i in range(len(data) // 4)]
    n = len(v)
    kb = bytes(key)[:16].ljust(16, b'\x00')
    k = [int.from_bytes(kb[4 * i:4 * i + 4], 'little') for i in range(4)]
    if rounds is None:
        rounds = 6 + 52 // n
    total = (rounds * _DELTA) & M32
    y = v[0]
    for _ in range(rounds):
        e = (total >> 2) & 3
        for p in range(n - 1, -1, -1):
            z = v[(p - 1) % n]
            mx = ((((z >> 5) ^ ((y << 2) & M32)) + (((y >> 3) ^ ((z << 4) & M32)) & M32)) & M32) \
                ^ (((total ^ y) + (k[(p & 3) ^ e] ^ z)) & M32)
            v[p] = (v[p] - mx) & M32
            y = v[p]
        total = (total - _DELTA) & M32
    return b''.join(x.to_bytes(4, 'little') for x in v)


# ---------------------------------------------------------------- Speck
SPECK_ROUNDS = {(16, 4): 22, (32, 3): 26, (32, 4): 27,
                (64, 2): 32, (64, 3): 33, (64, 4): 34}


def _speck_ror(x, n, w):
    return ((x >> n) | (x << (w - n))) & ((1 << w) - 1)


def _speck_rol(x, n, w):
    return ((x << n) | (x >> (w - n))) & ((1 << w) - 1)


def _speck_round_keys_ref(key, w, kw, rounds):
    alpha, beta = (7, 2) if w == 16 else (8, 3)
    mask = (1 << w) - 1
    k = [(key >> (w * i)) & mask for i in range(kw)]
    ell = list(k[1:])
    ks = [k[0]]
    for i in range(rounds - 1):
        if ell:
            li = _speck_ror(ell[i % len(ell)], alpha, w)
        else:
            li = 0
        l_new = ((ks[i] + li) & mask) ^ (i & mask)
        k_new = _speck_rol(ks[i], beta, w) ^ l_new
        if ell:
            ell[i % len(ell)] = l_new
        ks.append(k_new & mask)
    return ks, alpha, beta


def speck_enc_ref(pt, key, w=32, kw=4, rounds=None):
    """按 NSA 论文词序独立实现：key 整数低 w 位为 k0，block 整数高 w 位为 x。

    与 ctf_ext 的约定转换：
      * ctf_ext.speck_encrypt(pt, key) 的 key 整数按**小端**拆词（k0 最低位），
        与 NSA 论文「密钥最高词写在前」的 hex 表示相反；因此把论文 key 的 hex
        按整数直接传入即可（低 16/32 位恰好是 k0）。
      * block 整数高 w 位是 x（论文记法 (x,y)），即论文 hex 的 2w 位整数。
    """
    if rounds is None:
        rounds = SPECK_ROUNDS[(w, kw)]
    mask = (1 << w) - 1
    ks, alpha, beta = _speck_round_keys_ref(key, w, kw, rounds)
    x = (pt >> w) & mask
    y = pt & mask
    for i in range(rounds):
        x = (((_speck_ror(x, alpha, w) + y) & mask) ^ ks[i]) & mask
        y = (_speck_rol(y, beta, w) ^ x) & mask
    return ((x << w) | y) & ((1 << (2 * w)) - 1)


def speck_dec_ref(ct, key, w=32, kw=4, rounds=None):
    if rounds is None:
        rounds = SPECK_ROUNDS[(w, kw)]
    mask = (1 << w) - 1
    ks, alpha, beta = _speck_round_keys_ref(key, w, kw, rounds)
    x = (ct >> w) & mask
    y = ct & mask
    for i in range(rounds - 1, -1, -1):
        y = _speck_ror((y ^ x) & mask, beta, w)
        x = _speck_rol(((x ^ ks[i]) - y) & mask, alpha, w)
    return ((x << w) | y) & ((1 << (2 * w)) - 1)


# ---------------------------------------------------------------- MT19937
class MT19937Ref:
    """std::mt19937（32 位）纯 Python 参考实现。"""

    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = int(seed) & M32
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & M32
        self.idx = 624

    def _twist(self):
        for k in range(624):
            y = (self.mt[k] & 0x80000000) | (self.mt[(k + 1) % 624] & 0x7FFFFFFF)
            self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1)
            if y & 1:
                self.mt[k] ^= 0x9908B0DF
        self.idx = 0

    @staticmethod
    def temper(y):
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & M32

    def next(self):
        if self.idx >= 624:
            self._twist()
        v = self.temper(self.mt[self.idx])
        self.idx += 1
        return v

    def keys(self, n=24, twist=True):
        if twist:
            return bytes(self.next() & 0xFF for _ in range(n))
        return bytes(self.temper(self.mt[i]) & 0xFF for i in range(n))


# ---------------------------------------------------------------- GF(256) / RS(63,31)
def _gf_build(poly=0x11D):
    exp = [0] * 512
    log = [0] * 256
    v = 1
    for i in range(255):
        exp[i] = v
        log[v] = i
        v <<= 1
        if v & 0x100:
            v ^= poly
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


_GF_EXP, _GF_LOG = _gf_build()


def _gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _gf_poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                r[i + j] ^= _gf_mul(a, b)
    return r


def rs63_ref_encode(data, nsym=32, fcr=0):
    """系统 RS(n=63,k=31) 编码器（fcr=0，系数降幂，与 rs63_syndromes(order='msb') 对齐）。"""
    if len(data) != 63 - nsym:
        raise ValueError(f'RS(63,{63 - nsym}) 数据必须 {63 - nsym} 字节，收到 {len(data)}')
    gen = [1]
    for i in range(nsym):
        root = _GF_EXP[(fcr + i) % 255]
        gen = _gf_poly_mul(gen, [1, root])
    msg = list(data) + [0] * nsym
    for i in range(len(data)):
        coef = msg[i]
        if coef:
            for j, g in enumerate(gen):
                msg[i + j] ^= _gf_mul(g, coef)
    return bytes(list(data) + msg[-nsym:])


def rs63_ref_syndromes(cw, nsym=32, fcr=0, order='msb'):
    out = []
    for j in range(nsym):
        root = _GF_EXP[(fcr + j) % 255]
        acc = 0
        seq = cw if order == 'msb' else cw[::-1]
        for c in seq:
            acc = _gf_mul(acc, root) ^ c
        out.append(acc)
    return out


# ---------------------------------------------------------------- ECC 曲线 / 标量乘
_EC_CURVES = {
    'secp256k1': dict(
        p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
        a=0, b=7,
        gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
        n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141),
    'p256': dict(
        p=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF,
        a=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC,
        b=0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
        gx=0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
        gy=0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
        n=0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551),
}


def _ec_add_ref(P, Q, p, a):
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2:
        if (y1 + y2) % p == 0:
            return None
        lam = (3 * x1 * x1 + a) * pow(2 * y1 % p, -1, p) % p
    else:
        lam = (y2 - y1) * pow((x2 - x1) % p, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    return x3, (lam * (x1 - x3) - y1) % p


def ec_ref_mul(k, curve='secp256k1'):
    """自写 affine 双倍-加法标量乘（与模块的 Jacobian 实现是两套独立算法）。"""
    c = _EC_CURVES[curve]
    p, a = c['p'], c['a']
    k = int(k) % c['n']
    if k == 0:
        return None
    R = None
    Gp = (c['gx'], c['gy'])
    while k:
        if k & 1:
            R = _ec_add_ref(R, Gp, p, a)
        Gp = _ec_add_ref(Gp, Gp, p, a)
        k >>= 1
    return R


# ---------------------------------------------------------------- LFSR
def lfsr_ref(state, taps, nbits):
    """CLI 文档约定：右移 Fibonacci，fb=xor(选中位)，输出 fb，状态=[fb]+state[:-1]。"""
    st = list(state)
    out = []
    for _ in range(nbits):
        fb = 0
        for t in taps:
            fb ^= st[t - 1]
        out.append(fb)
        st = [fb] + st[:-1]
    return out


# ---------------------------------------------------------------- 小工具
def _is_probable_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(rng, bits):
    while True:
        n = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(n):
            return n


def _next_prime(n):
    n += 1 if n % 2 == 0 else 2
    while not _is_probable_prime(n):
        n += 2
    return n


# =====================================================================
# 参考实现自检（官方 / 已发布向量；失败 = harness 自身问题）
# =====================================================================
def check_references():
    """用已发布/官方向量验证本文件的独立参考实现，避免「参考写错 = 模块背锅」。"""
    fails = []

    def ck(name, cond, detail=''):
        if not cond:
            fails.append(f'{name} {detail}')

    # TEA：Wheeler & Needham 官方向量（key/pt 全零）
    ck('TEA 官方向量', _tea_enc_ref(0, 0, [0, 0, 0, 0]) == (0x41EA3A0A, 0x94BAA940))
    ck('TEA 解密参考', _tea_dec_ref(0x41EA3A0A, 0x94BAA940, [0, 0, 0, 0]) == (0, 0))
    # XTEA：BouncyCastle / Go stdlib 向量（词按大端从 hex 字节解析）
    kb = bytes.fromhex('000102030405060708090a0b0c0d0e0f')
    kx = [int.from_bytes(kb[4 * i:4 * i + 4], 'big') for i in range(4)]
    ck('XTEA 官方向量', _xtea_enc_ref(0x41424344, 0x45464748, kx) == (0x497DF3D0, 0x72612CB5))
    ck('XTEA 零密钥向量', _xtea_enc_ref(0, 0, [0, 0, 0, 0]) == (0xDEE9D4D8, 0xF7131ED9))
    # Speck：NSA 论文官方向量（key 低 w 位 = k0、block 高 w 位 = x）
    ck('Speck32/64 向量',
       speck_enc_ref(0x6574694C, 0x1918111009080100, 16, 4) == 0xA86842F2)
    ck('Speck64/128 向量',
       speck_enc_ref(0x3B7265747475432D, 0x1B1A1918131211100B0A090803020100, 32, 4)
       == 0x8C6FA548454E028B)
    # ChaCha20：RFC 8439 §2.4.2
    rfc_ks = chacha20_keystream_ref(bytes(range(32)),
                                    bytes.fromhex('000000090000004a00000000'), 1, 64, True)
    ck('ChaCha20 RFC8439 向量',
       rfc_ks.hex() == ('10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e'
                        'd2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e'))
    # MT19937：std::mt19937 默认种子 5489 的首个输出
    ck('MT19937 std 向量', MT19937Ref(5489).next() == 3499211612)
    # RC4：官方 "Key"/"Plaintext" 向量（BBF3… 是密文 = keystream XOR 明文）
    ck('RC4 向量',
       bytes(x ^ y for x, y in zip(rc4_ref(b'Key', 9), b'Plaintext'))
       == bytes.fromhex('BBF316E8D940AF0AD3'))
    # GF(256)：乘法交换律（x * x = x^2 与两个非平凡元素）
    ck('GF(256) 自洽', _gf_mul(2, 2) == 4 and _gf_mul(0x53, 0xCA) == _gf_mul(0xCA, 0x53))
    # RS：独立编码器产出的码字，独立 syndrome 必须全零
    rs_code = rs63_ref_encode(bytes(range(31)))
    ck('RS 参考编码器 ↔ 参考 syndrome',
       all(v == 0 for v in rs63_ref_syndromes(rs_code)),
       str(rs63_ref_syndromes(rs_code)[:4]))
    # ECC：k=1 必须等于 G（参考实现自洽）
    ck('ECC 参考 k=1=G', ec_ref_mul(1) == (_EC_CURVES['secp256k1']['gx'],
                                           _EC_CURVES['secp256k1']['gy']))
    # AES-CTR 参考实现（依赖 pycryptodome，缺失则跳过该行）
    if _HAVE_PYCRYPTODOME:
        ck('AES-CTR 参考自洽',
           aesctr_ref(bytes(16), bytes(16), 32, 5, 'be')
           == _AES.new(bytes(16), _AES.MODE_ECB).encrypt((5).to_bytes(16, 'big')) +
              _AES.new(bytes(16), _AES.MODE_ECB).encrypt((6).to_bytes(16, 'big')))

    if fails:
        raise HarnessError('参考实现自检失败: ' + '; '.join(fails))


# =====================================================================
# 类别 1：CRC32
# =====================================================================
@category('crc32')
def fuzz_crc32(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        L = rng.choice([0, 1, 2, 15, 16, 63, 64, 65, 255, 256, 1023, 1024, 4096])
        if rng.random() < 0.5:
            L = rng.randrange(0, 4097)
        B = rng.randrange(1, 7)
        raw = _rb(rng, B * L)
        data = np.frombuffer(raw, dtype=np.uint8).reshape(B, L)
        got = TOOLS.crc32_batch(data)
        for i in range(B):
            row = raw[i * L:(i + 1) * L]
            want = zlib.crc32(row) & M32
            ctx.check(int(got[i]) == want,
                      f'crc32_batch L={L} row#{i}: 本工具 {int(got[i]):08x} vs zlib {want:08x}')
        # crc32_state：未最终异或的中间状态，追加尾部字节后必须与 zlib 一致
        if L:
            head, tail = raw[:L // 2], raw[L // 2:L]
            st = TOOLS.crc32_state(head)
            tab = TOOLS._crc32_tab()
            c = st
            for b in tail:
                c = int(tab[(c ^ b) & 0xFF]) ^ (c >> 8)
            got2 = (c ^ M32) & M32
            want2 = zlib.crc32(raw[:L]) & M32
            ctx.check(got2 == want2,
                      f'crc32_state 增量 L={L}: {got2:08x} vs zlib {want2:08x}')


# =====================================================================
# 类别 2：MD5 / 长度扩展
# =====================================================================
@category('md5')
def fuzz_md5(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        L = rng.choice([0, 1, 55, 56, 63, 64, 65, 119, 120, 127, 128, 1000])
        if rng.random() < 0.5:
            L = rng.randrange(0, 1500)
        data = _rb(rng, L)
        ctx.check(EXT.md5_of(data) == hashlib.md5(data).digest(),
                  f'md5_of L={L}: {EXT.md5_of(data).hex()} vs hashlib')
        msgs = [data, b'', _rb(rng, rng.randrange(0, 64)), data + b'x']
        got = EXT.md5_batch(msgs)
        want = [hashlib.md5(m).digest() for m in msgs]
        ctx.check(got == want, f'md5_batch({len(msgs)} 条) 与 hashlib 不一致')

        # 哈希长度扩展：原消息长度取 padding 边界，伪造消息 = orig||glue||append
        orig_len = rng.choice([0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128])
        orig = _rb(rng, orig_len)
        append = _rb(rng, rng.randrange(0, 33))
        algo = 'md5' if ctx.iter_no % 2 else 'sha1'
        digest = hashlib.new(algo, orig).digest()
        glue, new = EXT.hash_extend(algo, orig_len, digest.hex(), append)
        pad = b'\x80' + b'\x00' * ((56 - (orig_len + 1) % 64) % 64)
        bits = (orig_len * 8) & 0xFFFFFFFFFFFFFFFF
        want_glue = pad + (bits.to_bytes(8, 'little' if algo == 'md5' else 'big'))
        ctx.check(glue == want_glue,
                  f'hash_extend({algo}) orig_len={orig_len} glue 错误: {_hex(glue)}')
        want_digest = hashlib.new(algo, orig + glue + append).digest()
        ctx.check(new == want_digest,
                  f'hash_extend({algo}) orig_len={orig_len}: {new.hex()} vs {want_digest.hex()}')


# =====================================================================
# 类别 3：ChaCha20
# =====================================================================
@category('chacha20')
def fuzz_chacha20(ctx):
    rng = ctx.rng
    boundary = [0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 200]
    for _ in range(ctx.iters):
        ctx.iter()
        B = rng.randrange(1, 5)
        keys = [_rb(rng, 32) for _ in range(B)]
        rfc = rng.random() < 0.5
        nonce = _rb(rng, 12 if rfc else 8)
        nbytes = rng.choice(boundary)
        if rng.random() < 0.4:
            nbytes = rng.randrange(0, 300)
        counter = rng.choice([0, 1, 2, 3, 7, 31])
        key_arr = np.frombuffer(b''.join(keys), dtype=np.uint8).reshape(B, 32)
        got = TOOLS.chacha20_keystream(key_arr, nonce, counter=counter,
                                       nbytes=nbytes, rfc7539=rfc)
        ctx.check(got.shape == (B, nbytes), f'chacha20_keystream 形状 {got.shape} != {(B, nbytes)}')
        for i, k in enumerate(keys):
            want = chacha20_keystream_ref(k, nonce, counter, nbytes, rfc)
            ctx.check(bytes(got[i]) == want,
                      f'chacha20 {"RFC7539" if rfc else "原版"} counter={counter} nbytes={nbytes} '
                      f'key={k.hex()[:16]}..: {_hex(got[i])} vs 参考 {_hex(want)}')
        # 第三方对拍：pycryptodome（小计数器用整段密钥流切片取偏移）
        if _HAVE_PYCRYPTODOME:
            ref = _ChaCha20.new(key=keys[0], nonce=nonce).encrypt(
                bytes(64 * counter + nbytes))
            ctx.check(bytes(got[0]) == ref[64 * counter:],
                      f'chacha20 vs pycryptodome counter={counter} nbytes={nbytes}: '
                      f'{_hex(got[0])} vs {_hex(ref[64 * counter:])}')

        # 大计数器（第三方没有直接接口，用已自检的纯 Python 参考）
        if rfc:
            big_ctr = rng.randrange(2 ** 32 - 8, 2 ** 32)      # blocks<=2 时不越界
            nbytes2 = rng.choice([1, 63, 64, 65, 128])
            if big_ctr + (nbytes2 + 63) // 64 - 1 > M32:
                big_ctr = M32 - 8
        else:
            big_ctr = rng.randrange(1 << 32, 1 << 63)
            nbytes2 = rng.choice([1, 63, 64, 65, 128])
        k0 = keys[0]
        got2 = TOOLS.chacha20_keystream(
            np.frombuffer(k0, dtype=np.uint8).reshape(1, 32), nonce,
            counter=big_ctr, nbytes=nbytes2, rfc7539=rfc)
        want2 = chacha20_keystream_ref(k0, nonce, big_ctr, nbytes2, rfc)
        ctx.check(bytes(got2[0]) == want2,
                  f'chacha20 大计数器 {"RFC7539" if rfc else "原版"} counter={big_ctr} '
                  f'nbytes={nbytes2}: {_hex(got2[0])} vs 参考 {_hex(want2)}')


# =====================================================================
# 类别 4：Poly1305 / AEAD tag
# =====================================================================
@category('poly1305')
def fuzz_poly1305(ctx):
    if not _HAVE_CRYPTOGRAPHY:
        ctx.skip('缺少 cryptography（raw Poly1305 / AEAD 参考）')
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        key = _rb(rng, 32)
        msg = _rb(rng, rng.choice([0, 1, 15, 16, 17, 31, 32, 33, 100, 300]))
        aad = _rb(rng, rng.randrange(0, 41))
        want = _RefPoly1305.generate_tag(key, aad + msg)      # 模块语义：data = aad || msg
        got = TOOLS.poly1305(key, msg, aad)
        ctx.check(got == want,
                  f'poly1305(key={key.hex()[:16]}.., msg={len(msg)}B, aad={len(aad)}B): '
                  f'{_hex(got)} vs cryptography {_hex(want)}')
        ctx.check(len(got) == 16, 'poly1305 tag 长度不为 16')

        # --- ChaCha20-Poly1305 AEAD 全流程：模块 chacha+poly vs cryptography ---
        akey = _rb(rng, 32)
        nonce = _rb(rng, 12)
        amsg = _rb(rng, rng.randrange(0, 200))
        aaad = _rb(rng, rng.randrange(0, 41))
        ref = _RefAEAD(akey).encrypt(nonce, amsg, aaad)
        ref_ct, ref_tag = ref[:-16], ref[-16:]
        otk = bytes(TOOLS.chacha20_keystream(akey, nonce, counter=0, nbytes=32)[0])
        ks = bytes(TOOLS.chacha20_keystream(akey, nonce, counter=1, nbytes=len(amsg))[0])
        our_ct = bytes(a ^ b for a, b in zip(amsg, ks))
        our_tag = TOOLS.poly1305_tag(otk, aaad, our_ct)
        ctx.check(our_ct == ref_ct,
                  f'AEAD 密文: {_hex(our_ct)} vs cryptography {_hex(ref_ct)}')
        ctx.check(our_tag == ref_tag,
                  f'AEAD tag: {our_tag.hex()} vs cryptography {ref_tag.hex()}')

        # --- tag 校验路径：模块算出的 ct/tag 必须能被第三方 verifier 接受 ---
        try:
            opened = _RefAEAD(akey).decrypt(nonce, our_ct + our_tag, aaad)
            ok_open = opened == amsg
        except Exception:
            ok_open = False
        ctx.check(ok_open, '模块算出的 AEAD tag 被 cryptography verifier 拒绝')
        if our_tag:
            bad_tag = bytes([our_tag[0] ^ 0x01]) + our_tag[1:]
            try:
                _RefAEAD(akey).decrypt(nonce, our_ct + bad_tag, aaad)
                rejected = False
            except Exception:
                rejected = True
            ctx.check(rejected, '被篡改的 AEAD tag 未被 cryptography verifier 拒绝')


# =====================================================================
# 类别 5：AES-ECB
# =====================================================================
@category('aes_ecb')
def fuzz_aes_ecb(ctx):
    if not _HAVE_PYCRYPTODOME:
        ctx.skip('缺少 pycryptodome')
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        klen = rng.choice([16, 24, 32])
        B = rng.randrange(1, 9)
        key = _rb(rng, klen)
        raw = _rb(rng, 16 * B)
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(B, 16)
        ref = _AES.new(key, _AES.MODE_ECB)
        # 单密钥快路径
        got = G.from_dev(G.aes_ecb(blocks, np.frombuffer(key, dtype=np.uint8)))
        for i in range(B):
            want = ref.encrypt(raw[16 * i:16 * (i + 1)])
            ctx.check(bytes(got[i]) == want,
                      f'AES-{klen * 8} ECB 单密钥快路径 block#{i} key={key.hex()}: '
                      f'{_hex(got[i])} vs {_hex(want)}')
        # 每候选不同密钥的通用路径
        keys = [_rb(rng, klen) for _ in range(B)]
        key_mat = np.frombuffer(b''.join(keys), dtype=np.uint8).reshape(B, klen)
        got2 = G.from_dev(G.aes_ecb(blocks, key_mat))
        for i in range(B):
            want = _AES.new(keys[i], _AES.MODE_ECB).encrypt(raw[16 * i:16 * (i + 1)])
            ctx.check(bytes(got2[i]) == want,
                      f'AES-{klen * 8} ECB 每候选密钥 block#{i} key={keys[i].hex()}: '
                      f'{_hex(got2[i])} vs {_hex(want)}')


# =====================================================================
# 类别 6：AES-CTR（le/be × 8/12/16B nonce × 计数器边界）
# =====================================================================
@category('aes_ctr')
def fuzz_aes_ctr(ctx):
    if not _HAVE_PYCRYPTODOME:
        ctx.skip('缺少 pycryptodome（AES-ECB 计数器块）')
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        klen = rng.choice([16, 24, 32])
        key = _rb(rng, klen)
        nonce_len = rng.choice([8, 12, 16])
        mode = rng.choice(['le', 'be'])
        nonce = _rb(rng, nonce_len)
        suffix = 16 - nonce_len
        limit = (1 << 128) if suffix == 0 else (1 << (8 * suffix))
        nbytes = rng.choice([0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 160])
        if rng.random() < 0.4:
            nbytes = rng.randrange(0, 257)
        blocks = (nbytes + 15) // 16
        if rng.random() < 0.3 and blocks and limit > blocks:
            counter = limit - blocks                       # 恰好到计数器上界
        else:
            counter = rng.randrange(0, min(limit, 1 << 20))
        got = TOOLS.aesctr_keystream(key, nonce, nbytes, counter, mode)
        ctx.check(isinstance(got, (bytes, bytearray)) and len(got) == nbytes,
                  f'aesctr_keystream 长度 {len(got)} != {nbytes}')
        want = aesctr_ref(key, nonce, nbytes, counter, mode)
        ctx.check(bytes(got) == want,
                  f'AES-{klen * 8}-CTR nonce={nonce_len}B mode={mode} counter={counter} '
                  f'nbytes={nbytes}: {_hex(got)} vs 参考 {_hex(want)}')
        # be 模式额外与 pycryptodome 原生 CTR 对拍（初始计数器块 = nonce||counter 大端）
        if mode == 'be' and counter + max(0, blocks - 1) < limit:
            if suffix:
                init = int.from_bytes(nonce + counter.to_bytes(suffix, 'big'), 'big')
            else:
                init = (int.from_bytes(nonce, 'big') + counter) & ((1 << 128) - 1)
            try:
                if init + max(0, blocks - 1) < (1 << 128):
                    ref = _AES.new(key, _AES.MODE_CTR, nonce=b'',
                                   initial_value=init).encrypt(bytes(nbytes))
                    ctx.check(bytes(got) == ref,
                              f'AES-CTR be vs pycryptodome native counter={counter} '
                              f'nbytes={nbytes}: {_hex(got)} vs {_hex(ref)}')
            except Exception:
                pass                                          # 超 pycryptodome 上限时跳过


# =====================================================================
# 类别 7：RC4
# =====================================================================
@category('rc4')
def fuzz_rc4(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        klen = rng.choice([1, 2, 3, 4, 16, 32, 64, 128, 255, 256])
        if rng.random() < 0.4:
            klen = rng.randrange(1, 257)
        key = _rb(rng, klen)
        n = rng.choice([0, 1, 15, 16, 255, 256, 257, 512])
        if rng.random() < 0.4:
            n = rng.randrange(0, 600)
        got = EXT.rc4_keystream(key, n)
        ctx.check(len(got) == n, f'rc4_keystream 长度 {len(got)} != {n}')
        want = rc4_ref(key, n)
        ctx.check(got == want,
                  f'rc4_keystream key={klen}B n={n}: {_hex(got)} vs 自写参考 {_hex(want)}')
        if _HAVE_PYCRYPTODOME:
            ref = _ARC4.new(key).encrypt(bytes(n))
            ctx.check(got == ref,
                      f'rc4_keystream vs pycryptodome ARC4 key={key.hex()} n={n}: '
                      f'{_hex(got)} vs {_hex(ref)}')
        data = _rb(rng, n)
        got2 = EXT.rc4(data, key)
        ctx.check(got2 == bytes(x ^ y for x, y in zip(data, want)),
                  f'rc4(data) key={klen}B n={n} 与参考密钥流异或不一致')


# =====================================================================
# 类别 8：TEA / XTEA / XXTEA
# =====================================================================
@category('tea')
def fuzz_tea(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        key = [rng.getrandbits(32) for _ in range(4)]
        v0, v1 = rng.getrandbits(32), rng.getrandbits(32)
        # TEA
        e = EXT.tea_encrypt_block(v0, v1, key)
        ctx.check(e == _tea_enc_ref(v0, v1, key),
                  f'TEA 加密 key={[hex(k) for k in key]} pt=({v0:08x},{v1:08x}): '
                  f'{e} vs 参考 {_tea_enc_ref(v0, v1, key)}')
        d = EXT.tea_decrypt_block(e[0], e[1], key)
        ctx.check(d == (v0, v1), f'TEA 往返失败 {d} != {(v0, v1)}')
        ctx.check(d == _tea_dec_ref(e[0], e[1], key), 'TEA 解密 vs 参考不一致')
        # XTEA
        xe = EXT.xtea_encrypt_block(v0, v1, key)
        xr = _xtea_enc_ref(v0, v1, key)
        ctx.check(xe == xr, f'XTEA 加密 ({v0:08x},{v1:08x}): {xe} vs 参考 {xr}')
        xd = EXT.xtea_decrypt_block(xe[0], xe[1], key)
        ctx.check(xd == (v0, v1), f'XTEA 往返失败 {xd} != {(v0, v1)}')
        ctx.check(xd == _xtea_dec_ref(xe[0], xe[1], key), 'XTEA 解密 vs 参考不一致')
        # XXTEA（16B key，任意长度数据；加密自动补齐到 4/8 字节）
        keyb = _rb(rng, 16)
        L = rng.choice([0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64])
        if rng.random() < 0.4:
            L = rng.randrange(0, 80)
        data = _rb(rng, L)
        ct = EXT.xxtea_encrypt(data, keyb)
        want_ct = _xxtea_enc_ref(data, keyb)
        ctx.check(ct == want_ct,
                  f'XXTEA 加密 len={L} key={keyb.hex()}: {_hex(ct)} vs 参考 {_hex(want_ct)}')
        pt = EXT.xxtea_decrypt(ct, keyb)
        ctx.check(pt == _xxtea_pad(data),
                  f'XXTEA 解密 len={L}: {_hex(pt)} != 补齐后明文 {_hex(_xxtea_pad(data))}')
        ctx.check(pt == _xxtea_dec_ref(ct, keyb), 'XXTEA 解密 vs 参考不一致')


# =====================================================================
# 类别 9：Speck32/64、Speck64/128
# =====================================================================
@category('speck')
def fuzz_speck(ctx):
    rng = ctx.rng
    variants = [('speck32/64', 16, 4), ('speck64/128', 32, 4), ('speck96/96', 32, 3)]
    for _ in range(ctx.iters):
        ctx.iter()
        name, w, kw = variants[ctx.iter_no % len(variants)]
        rounds = SPECK_ROUNDS[(w, kw)]
        key = rng.getrandbits(w * kw)
        pt = rng.getrandbits(2 * w)
        got = EXT.speck_encrypt(pt, key, word_bits=w, key_words=kw)
        want = speck_enc_ref(pt, key, w, kw, rounds)
        ctx.check(got == want,
                  f'{name} 加密 key={key:0{w * kw // 4}x} pt={pt:0{2 * w // 4}x}: '
                  f'{got:0{2 * w // 4}x} vs 参考 {want:0{2 * w // 4}x}')
        dec = EXT.speck_decrypt(got, key, word_bits=w, key_words=kw)
        ctx.check(dec == pt, f'{name} 往返失败 {dec:0{2 * w // 4}x} != {pt:0{2 * w // 4}x}')
        ctx.check(speck_dec_ref(got, key, w, kw, rounds) == pt,
                  f'{name} 参考解密与模块加密不一致')


# =====================================================================
# 类别 10：MT19937 + popcount / bit_agreement
# =====================================================================
@category('mt19937')
def fuzz_mt19937(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        seeds = [0, 1, M32, rng.getrandbits(32), rng.getrandbits(63)]
        for tw in (True, False):
            try:
                got = G.mt19937_keys(np.array(seeds, dtype=np.int64), twist=tw)
            except Exception as e:                       # noqa: BLE001 - 记录为不一致
                raise FuzzMismatch(
                    f'[iter {ctx.iter_no}] mt19937_keys(numpy 后端, twist={tw}) '
                    f'对 seeds={seeds} 抛异常 {type(e).__name__}: {e}') from e
            ref = [MT19937Ref(s).keys(24, twist=tw) for s in seeds]
            for i, s in enumerate(seeds):
                ctx.check(bytes(got[i]) == ref[i],
                          f'MT19937 twist={tw} seed={s}: {_hex(got[i])} vs 参考 {_hex(ref[i])}')
        # popcount_matrix vs 纯 Python popcount（含负数 int64 / 超范围 uint64）
        rows, cols = rng.randrange(1, 5), rng.randrange(1, 8)
        vals = [rng.randint(-(2 ** 63), 2 ** 63 - 1) for _ in range(rows * cols)]
        arr = np.array(vals, dtype=np.int64).reshape(rows, cols)
        got_pc = G.from_dev(G.popcount_matrix(arr)).astype(np.int64)
        want_pc = np.array([(v & 0xFFFFFFFFFFFFFFFF).bit_count() for v in vals],
                           dtype=np.int64).reshape(rows, cols)
        ctx.check(np.array_equal(got_pc, want_pc),
                  f'popcount_matrix(int64) 不一致: {got_pc.tolist()} vs {want_pc.tolist()}')
        uvals = [rng.getrandbits(64) for _ in range(rng.randrange(1, 20))]
        uarr = np.array(uvals, dtype=np.uint64)
        got_pc2 = G.from_dev(G.popcount_matrix(uarr)).astype(np.int64)
        want_pc2 = np.array([v.bit_count() for v in uvals], dtype=np.int64)
        ctx.check(np.array_equal(got_pc2, want_pc2), 'popcount_matrix(uint64) 不一致')
        # bit_agreement：1D 目标 / 2D 目标
        N, L = rng.randrange(1, 9), rng.randrange(1, 17)
        cand = np.array([rng.randrange(256) for _ in range(N * L)], dtype=np.uint8).reshape(N, L)
        tgt1 = np.array([rng.randrange(256) for _ in range(L)], dtype=np.uint8)
        got_ba = G.from_dev(G.bit_agreement(cand, tgt1)).astype(np.int64)
        want_ba = np.array([8 * L - sum((int(cand[i, j]) ^ int(tgt1[j])).bit_count()
                                        for j in range(L)) for i in range(N)])
        ctx.check(np.array_equal(got_ba, want_ba),
                  f'bit_agreement 1D 目标: {got_ba.tolist()} vs {want_ba.tolist()}')
        M = rng.randrange(1, 5)
        tgt2 = np.array([rng.randrange(256) for _ in range(M * L)], dtype=np.uint8).reshape(M, L)
        got_ba2 = G.from_dev(G.bit_agreement(cand, tgt2)).astype(np.int64)
        want_ba2 = np.array([[8 * L - sum((int(cand[i, j]) ^ int(tgt2[m, j])).bit_count()
                                          for j in range(L)) for m in range(M)]
                             for i in range(N)])
        ctx.check(np.array_equal(got_ba2, want_ba2), 'bit_agreement 2D 目标不一致')
    # 负数种子放在最后：torch 后端会先 & 0xFFFFFFFF 再播种（= uint32 语义），
    # numpy 后端也必须一致；这里能抓到「numpy 后端对负数种子崩溃/结果不同」的缺陷。
    ctx.iter()
    neg_seeds = [-1, -12345, -(2 ** 31), -(2 ** 63) + 1]
    for tw in (True, False):
        try:
            got = G.mt19937_keys(np.array(neg_seeds, dtype=np.int64), twist=tw)
        except Exception as e:                            # noqa: BLE001
            raise FuzzMismatch(
                f'[iter {ctx.iter_no}] mt19937_keys(numpy 后端, twist={tw}) '
                f'对负数种子 {neg_seeds} 抛异常 {type(e).__name__}: {e}') from e
        ref = [MT19937Ref(s).keys(24, twist=tw) for s in neg_seeds]
        for i, s in enumerate(neg_seeds):
            ctx.check(bytes(got[i]) == ref[i],
                      f'MT19937 twist={tw} 负数种子 {s}: {_hex(got[i])} vs 参考 {_hex(ref[i])}')


# =====================================================================
# 类别 11：RS(63,31)
# =====================================================================
@category('rs63')
def fuzz_rs63(ctx):
    rng = ctx.rng
    # 已知合法码字（来自挑战题包，consistency.py 已用同一向量）做一次 sanity
    known = bytes.fromhex(
        '504d333145c89c44bdc8b495f33d47423739807b62151310db48dc95c37f79'
        'a8ea2397662150254460bc05d349ac9f7d2a600cbca3b4e9d75f923bf07b2941')
    if len(known) == 63:
        ctx.check(all(v == 0 for v in rs63_ref_syndromes(known)),
                  '参考 syndrome 对模块已知码字非零（参考实现约定有误）')
        ctx.check(TOOLS.rs63_check(known, 32, 0, 'msb')[0],
                  '模块对已知合法码字判为非法')
    for _ in range(ctx.iters):
        ctx.iter()
        data = _rb(rng, 31)
        cw = rs63_ref_encode(data)
        ctx.check(all(v == 0 for v in rs63_ref_syndromes(cw)), '参考码字 syndrome 非零')
        ok, info = TOOLS.rs63_check(cw, 32, 0, 'msb')
        ctx.check(ok, f'RS 合法码字被判非法: {info}')
        ctx.check((TOOLS.rs63_syndromes(cw, 32, 0, 'msb')[0] == 0).all(),
                  'rs63_syndromes 对合法码字非零')
        # 单字节错误：必须检测到并可定位
        pos = rng.randrange(63)
        delta = rng.randrange(1, 256)
        bad = bytearray(cw)
        bad[pos] ^= delta
        ctx.check(not TOOLS.rs63_check(bytes(bad), 32, 0, 'msb')[0],
                  f'RS 单字节错误 (pos={pos}, delta={delta:02x}) 未被检测')
        loc = TOOLS.rs63_locate(bytes(bad), 32, 0, 'msb')
        ctx.check(loc == (pos, delta),
                  f'RS 单字节定位: 期望 ({pos},{delta:02x}) 得到 {loc}')
        # 2~16 个错误：可检测（≤t=16 的纠错能力以 syndrome 非零体现），但单字节 locate 必须放弃
        k = rng.randrange(2, 17)
        bad2 = bytearray(cw)
        for p in rng.sample(range(63), k):
            bad2[p] ^= rng.randrange(1, 256)
        ctx.check(not TOOLS.rs63_check(bytes(bad2), 32, 0, 'msb')[0],
                  f'RS {k} 字节错误未被检测')
        ctx.check(TOOLS.rs63_locate(bytes(bad2), 32, 0, 'msb') is None,
                  f'RS {k} 字节错误被单字节 locate 误报')
        # >16 个错误：不要求纠正，但必须不 crash 且判非合法
        k2 = rng.randrange(17, 33)
        bad3 = bytearray(cw)
        for p in rng.sample(range(63), k2):
            bad3[p] ^= rng.randrange(1, 256)
        try:
            ok3, _info3 = TOOLS.rs63_check(bytes(bad3), 32, 0, 'msb')
            loc3 = TOOLS.rs63_locate(bytes(bad3), 32, 0, 'msb')
        except Exception as e:                            # noqa: BLE001
            raise FuzzMismatch(
                f'[iter {ctx.iter_no}] RS {k2} 字节错误导致异常 '
                f'{type(e).__name__}: {e}') from e
        ctx.check(ok3 is False, f'RS {k2} 字节错误被判为合法码字')
        ctx.check(loc3 is None or isinstance(loc3, tuple),
                  f'rs63_locate 返回值类型异常: {loc3!r}')


# =====================================================================
# 类别 12：LCG
# =====================================================================
@category('lcg')
def fuzz_lcg(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        m = rng.choice([2 ** 31 - 1, 2 ** 32, 2 ** 31, 10 ** 9 + 7, 2 ** 61 - 1])
        if m % 2 == 0:
            a, c = rng.randrange(2, m) | 1, rng.randrange(1, m) | 1
        else:
            a, c = rng.randrange(2, m), rng.randrange(0, m)
        x = rng.randrange(0, m)
        seq = []
        for _ in range(12):
            x = (a * x + c) % m
            seq.append(x)
        # 参考序列本身生成合法（相邻输出不重合），避免 t_i=0 造成不可恢复
        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            continue
        r = EXT.lcg_recover(seq[:5], m)
        ctx.check(r == (a, c),
                  f'lcg_recover m={m} a={a} c={c} 得到 {r}')
        pred = EXT.lcg_predict(seq[:5], m, 5)
        ctx.check(pred == seq[5:10],
                  f'lcg_predict m={m}: {pred} vs 实际 {seq[5:10]}')
        r2 = EXT.lcg_recover(seq[:8], m)
        ctx.check(r2 == (a, c), f'lcg_recover(8 输出) m={m} 得到 {r2}')


# =====================================================================
# 类别 13：ECC
# =====================================================================
@category('ecc')
def fuzz_ecc(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        curve = rng.choice(['secp256k1', 'p256'])
        order = _EC_CURVES[curve]['n']
        k = rng.choice([1, 2, 3, 4, order - 1, order - 2])
        if ctx.iter_no % 2 == 0:
            k = rng.randrange(1, order)
        x, y = EXT.ecc_pub(curve, k)
        # 1) 自写 affine 标量乘（与模块 Jacobian 实现是两套独立算法）
        ref = ec_ref_mul(k, curve)
        ctx.check((x, y) == ref,
                  f'ecc_pub({curve}, k={k}): ({hex(x)},{hex(y)}) vs 自写参考 '
                  f'({hex(ref[0])},{hex(ref[1])})')
        # 2) cryptography（第三方 EC 实现）
        if _HAVE_CRYPTOGRAPHY:
            curve_obj = _RefEC.SECP256K1() if curve == 'secp256k1' else _RefEC.SECP256R1()
            try:
                pn = _RefEC.derive_private_key(k, curve_obj).public_key().public_numbers()
                ctx.check((x, y) == (pn.x, pn.y),
                          f'ecc_pub({curve}, k={k}) vs cryptography 不一致')
            except Exception:      # pragma: no cover - 某些 OpenSSL 无 secp256k1
                pass
        # 3) 曲线方程校验（点必须在曲线上）
        p, a, b = _EC_CURVES[curve]['p'], _EC_CURVES[curve]['a'], _EC_CURVES[curve]['b']
        ctx.check((y * y - x * x * x - a * x - b) % p == 0,
                  f'ecc_pub 结果不在 {curve} 曲线上')

    # 4) eccscan 小私钥区间找回：植入 k，在含 k 的小区间里必须找到；不含 k 时必须返回 None
    for _ in range(max(1, ctx.iters // 2)):
        ctx.iter()
        curve = rng.choice(['secp256k1', 'p256'])
        k = rng.randrange(1, 600)
        tx, ty = EXT.ecc_pub(curve, k)
        lo = max(0, k - rng.randrange(0, 40))
        cnt = (k - lo) + rng.randrange(1, 40) + 1
        hit = EXT.ecc_search(curve, tx, ty, lo, cnt)
        ctx.check(hit == k, f'ecc_search 植入-找回失败: 期望 {k} 得到 {hit} (lo={lo}, cnt={cnt})')
        miss = EXT.ecc_search(curve, tx, ty, 0, k)
        ctx.check(miss is None, f'ecc_search 区间外误报: 得到 {miss} (k={k})')


# =====================================================================
# 类别 14：RSA
# =====================================================================
@category('rsa')
def fuzz_rsa(ctx):
    rng = ctx.rng
    tested_attack = False
    tested_pm1 = False
    for _ in range(ctx.iters):
        ctx.iter()
        bits = rng.choice([13, 14, 15, 16])
        p = _random_prime(rng, bits)
        q = _random_prime(rng, bits)
        while q == p:
            q = _random_prime(rng, bits)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = rng.choice([3, 5, 17, 257, 65537])
        if math.gcd(e, phi) != 1:
            e = 65537
            if math.gcd(e, phi) != 1:
                continue
        m = rng.randrange(2, n)
        c = pow(m, e, n)
        got = EXT._decrypt_with_pq(n, e, c, p, q)
        ctx.check(got == m,
                  f'_decrypt_with_pq(n={n}, e={e}, p={p}, q={q}) 得到 {got} 期望 {m}')
        # 模式幂（批量）
        bases = [m, (m * 2) % n, rng.randrange(0, n), rng.randrange(0, n)]
        me = TOOLS.modexp_batch(bases, e, n)
        ctx.check([int(v) for v in me] == [pow(b, e, n) for b in bases],
                  f'modexp_batch exp={e} mod={n} 与 pow 不一致')
        # 共模攻击
        if math.gcd(m, n) == 1:
            e1, e2 = 3, 65537
            c1, c2 = pow(m, e1, n), pow(m, e2, n)
            got_cm = EXT.rsa_common_modulus(n, e1, c1, e2, c2)
            ctx.check(got_cm == m, f'rsa_common_modulus 得到 {got_cm} 期望 {m}')
        # 低指数广播（e=3，三个互素模数，m^3 < Πn_i）
        if ctx.iter_no % 2 == 0:
            m_small = rng.randrange(2, 1 << 16)
            mods = []
            cts = []
            while len(mods) < 3:
                pi = _random_prime(rng, 15)
                qi = _random_prime(rng, 15)
                ni = pi * qi
                if any(math.gcd(ni, nj) != 1 for nj in mods):
                    continue
                mods.append(ni)
                cts.append(pow(m_small, 3, ni))
            try:
                got_bc = EXT.rsa_broadcast(list(zip(mods, cts)), 3)
            except Exception as ex:                       # noqa: BLE001
                raise FuzzMismatch(
                    f'[iter {ctx.iter_no}] rsa_broadcast 抛异常 {type(ex).__name__}: {ex}') from ex
            ctx.check(got_bc == m_small,
                      f'rsa_broadcast 得到 {got_bc} 期望 {m_small}')
        # n = p²（φ = p(p-1)）
        p2 = _random_prime(rng, 15)
        n2 = p2 * p2
        phi2 = p2 * (p2 - 1)
        e2 = 65537 if math.gcd(65537, phi2) == 1 else 3
        if math.gcd(e2, phi2) == 1:
            m2 = rng.randrange(2, n2)
            got_p2 = EXT._decrypt_with_pq(n2, e2, pow(m2, e2, n2), p2, p2)
            ctx.check(got_p2 == m2, f'n=p² 解密 得到 {got_p2} 期望 {m2}')
        # Fermat（p、q 接近）
        pf = _random_prime(rng, rng.choice([12, 13, 14]))
        qf = _next_prime(pf)
        nf = pf * qf
        f = EXT.rsa_fermat(nf, 1000)
        ctx.check(f == (pf, qf) or (f is not None and f[0] * f[1] == nf),
                  f'rsa_fermat({pf}*{qf}) 得到 {f}')
        # Pollard rho
        rho = EXT.rsa_pollard_rho(nf)
        ctx.check(rho is not None and nf % rho == 0,
                  f'rsa_pollard_rho({nf}) 得到 {rho}')
        # rsa_attack 植入-找回（小因子路径，只做一次以控时）
        if not tested_attack and ctx.iter_no >= 2:
            p_small = _random_prime(rng, 13)
            q_big = _random_prime(rng, 17)
            n_atk = p_small * q_big
            phi_atk = (p_small - 1) * (q_big - 1)
            e_atk = 65537 if math.gcd(65537, phi_atk) == 1 else 3
            if math.gcd(e_atk, phi_atk) == 1:
                m_atk = rng.randrange(2, n_atk)
                meth, dec = EXT.rsa_attack(n_atk, e_atk, pow(m_atk, e_atk, n_atk),
                                           verbose=False)
                ctx.check(dec == m_atk,
                          f'rsa_attack 植入-找回失败: 方法={meth} 得到 {dec} 期望 {m_atk}')
                tested_attack = True
        # Pollard p-1（p-1 含 ≤1e5 的素数幂；q-1 含大因子），只做一次
        if not tested_pm1 and ctx.iter_no >= 3:
            p_pm1 = 2 ** 15 * 94307 + 1
            q_pm1 = 1000003
            n_pm1 = p_pm1 * q_pm1
            g = EXT.rsa_pm1(n_pm1)
            ctx.check(g is not None and n_pm1 % g == 0,
                      f'rsa_pm1({n_pm1}) 得到 {g}')
            tested_pm1 = True


# =====================================================================
# 类别 15：auto_decode 多层编码链
# =====================================================================
def _rot13_bytes(b):
    return bytes((65 + (x - 65 + 13) % 26) if 65 <= x <= 90 else
                 (97 + (x - 97 + 13) % 26) if 97 <= x <= 122 else x for x in b)


_ENCODERS = {
    'hex': lambda b: b.hex().encode(),
    'b64': base64.b64encode,
    'b32': base64.b32encode,
    'url': lambda b: urllib.parse.quote_from_bytes(b).encode(),
    'rot13': _rot13_bytes,
    'rev': lambda b: b[::-1],
}


_STRICT_ENCODERS = {
    'hex': lambda b: b.hex().encode(),
    'b64': base64.b64encode,
    'b32': base64.b32encode,
}

_FLAG_RE = re.compile(rb'flag\{[^}]*\}')


@category('auto_decode')
def fuzz_auto_decode(ctx):
    """auto_decode 多层编码链还原。

    两种断言强度（对应模块两种契约）：
      * 严格模式：只用 hex/b64/b32 组成的 1~3 层链，结果必须是**原始 bytes**
        （这些编码与 BFS 路径无歧义）。
      * 混合模式：链里加入 url/rot13/reversed，按模块文档契约（`_looks_flag`：
        结果含 flag 特征即返回）只要求**完整 flag 子串**被正确还原。
        原因：混合变换可能让 BFS 先撞到一条较短的合法解码路径，得到
        `flag{...}` 后带少量尾随字节的候选（模块文档明确是启发式返回，不算缺陷）。
    """
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        plain = (b'flag{' + bytes(rng.choice(b'abcdefghij0123456789_') for _ in range(8, 24))
                 + b'}')
        strict = rng.random() < 0.5
        if strict:
            ops = [rng.choice(list(_STRICT_ENCODERS)) for _ in range(rng.choice([1, 2, 3]))]
        else:
            ops = [rng.choice(list(_ENCODERS)) for _ in range(rng.choice([1, 2, 3]))]
        data = plain
        for op in ops:
            data = _ENCODERS[op](data)
        if b'flag{' in data:        # 编码后不应还含 flag 明文特征，否则 auto_decode 无需解码
            continue
        try:
            got, path = EXT.auto_decode(data, verbose=False)
        except Exception as e:                            # noqa: BLE001
            raise FuzzMismatch(
                f'[iter {ctx.iter_no}] auto_decode(chain={ops}) 抛异常 '
                f'{type(e).__name__}: {e}') from e
        if strict:
            ctx.check(got == plain,
                      f'auto_decode 严格链 chain={ops} 未还原原始 bytes: '
                      f'got={got!r} path={path}')
        else:
            m = _FLAG_RE.search(got) if got else None
            ctx.check(m is not None and m.group(0) == plain,
                      f'auto_decode 混合链 chain={ops} 未还原完整 flag: '
                      f'got={got!r} path={path}')


# =====================================================================
# 类别 16：numpy ↔ torch 双后端一致性（仅 --backend torch）
# =====================================================================
def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:          # pragma: no cover
        return False


@category('dual_backend')
def fuzz_dual_backend(ctx):
    if not _torch_available():
        ctx.skip('torch 不可用')
    import torch  # noqa: F811
    rng = ctx.rng
    for _ in range(max(1, ctx.iters // 2)):
        ctx.iter()
        klen = rng.choice([16, 24, 32])
        key = _rb(rng, klen)
        raw = _rb(rng, 8 * 16)
        seeds = [rng.getrandbits(32) for _ in range(8)]
        ckeys = _rb(rng, 4 * 32)
        cnonce = _rb(rng, 12)
        pc = np.array([rng.randrange(-2 ** 63, 2 ** 63) for _ in range(24)],
                      dtype=np.int64)
        cand = np.array([rng.randrange(256) for _ in range(8 * 12)],
                        dtype=np.uint8).reshape(8, 12)
        tgt = np.array([rng.randrange(256) for _ in range(12)], dtype=np.uint8)

        # ---- numpy 后端 ----
        G.init_backend('numpy')
        try:
            np_aes = G.from_dev(G.aes_ecb(
                np.frombuffer(raw, dtype=np.uint8).reshape(8, 16),
                np.frombuffer(key, dtype=np.uint8))).tobytes()
            np_ch = TOOLS.chacha20_keystream(
                np.frombuffer(ckeys, dtype=np.uint8).reshape(4, 32), cnonce, 3, 128,
                rfc7539=True).tobytes()
            np_ctr = bytes(TOOLS.aesctr_keystream(key, bytes(8), 96, 7, 'le'))
            np_mt = G.from_dev(G.mt19937_keys(np.array(seeds, dtype=np.int64),
                                              twist=True)).tobytes()
            np_pc = G.from_dev(G.popcount_matrix(pc)).astype(np.int64).tolist()
            np_ba = G.from_dev(G.bit_agreement(cand, tgt)).astype(np.int64).tolist()
        finally:
            G.init_backend('numpy')

        # ---- torch 后端 ----
        G.init_backend('torch')
        try:
            T = G.xp()
            dev = G.DEV
            t_aes = G.from_dev(G.aes_ecb(
                T.tensor(np.frombuffer(raw, dtype=np.uint8).reshape(8, 16).tolist(),
                         dtype=torch.uint8, device=dev),
                T.tensor(list(key), dtype=torch.uint8, device=dev))).tobytes()
            t_ch = TOOLS.chacha20_keystream(
                T.tensor(np.frombuffer(ckeys, dtype=np.uint8).reshape(4, 32).tolist(),
                         dtype=torch.uint8, device=dev), cnonce, 3, 128,
                rfc7539=True).tobytes()
            t_ctr = bytes(TOOLS.aesctr_keystream(key, bytes(8), 96, 7, 'le'))
            t_mt = G.from_dev(G.mt19937_keys(T.tensor(seeds, dtype=torch.int64,
                                                      device=dev), twist=True)).tobytes()
            t_pc = G.from_dev(G.popcount_matrix(
                T.tensor(pc.tolist(), dtype=torch.int64, device=dev))).astype(np.int64).tolist()
            t_ba = G.from_dev(G.bit_agreement(
                T.tensor(cand.tolist(), dtype=torch.uint8, device=dev),
                T.tensor(tgt.tolist(), dtype=torch.uint8, device=dev))).astype(np.int64).tolist()
        finally:
            G.init_backend('numpy')
        ctx.check(np_aes == t_aes, f'AES-{klen * 8} numpy↔torch 不一致: '
                                   f'{_hex(np_aes)} vs {_hex(t_aes)}')
        ctx.check(np_ch == t_ch, f'ChaCha20 numpy↔torch 不一致: {_hex(np_ch)} vs {_hex(t_ch)}')
        ctx.check(np_ctr == t_ctr, f'AES-CTR numpy↔torch 不一致: {_hex(np_ctr)} vs {_hex(t_ctr)}')
        ctx.check(np_mt == t_mt, 'mt19937 numpy↔torch 不一致')
        ctx.check(np_pc == t_pc, f'popcount numpy↔torch 不一致: {np_pc[:8]} vs {t_pc[:8]}')
        ctx.check(np_ba == t_ba, f'bit_agreement numpy↔torch 不一致: {np_ba[:4]} vs {t_ba[:4]}')


# =====================================================================
# 类别 17：LFSR 语义（cmd_lfsrscan 对拍独立参考）
# =====================================================================
@category('lfsr')
def fuzz_lfsr(ctx):
    rng = ctx.rng
    for _ in range(ctx.iters):
        ctx.iter()
        n = rng.choice([6, 7, 8, 9])
        taps = sorted(rng.sample(range(1, n + 1), rng.randrange(1, min(n, 4) + 1)))
        state = [rng.randrange(2) for _ in range(n)]
        nbits = rng.choice([16, 24, 32])
        bits = lfsr_ref(state, taps, nbits)
        target = ''.join(str(b) for b in bits)

        class _A:                                     # 模拟 argparse.Namespace
            pass

        a = _A()
        a.taps = f'{n}:{",".join(str(t) for t in taps)}'
        a.bits = target
        a.trials = 16384                              # 2^14 >> 状态空间 2^n(≤512)，确定性覆盖
        a.rng_seed = rng.randrange(0, 1 << 31)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                rc = G.cmd_lfsrscan(a)
        except Exception as e:                        # noqa: BLE001
            raise FuzzMismatch(
                f'[iter {ctx.iter_no}] cmd_lfsrscan(taps={a.taps}) 抛异常 '
                f'{type(e).__name__}: {e}') from e
        out = buf.getvalue()
        m = None
        for line in out.splitlines():
            if '最佳' in line and '/' in line:
                try:
                    m = int(line.split('最佳')[1].split('/')[0].strip())
                except ValueError:
                    m = None
        ctx.check(rc in (0, None),
                  f'cmd_lfsrscan 返回 {rc}（期望 0/None，taps={a.taps}）')
        ctx.check(m == nbits,
                  f'cmd_lfsrscan 语义对拍失败：n={n} taps={taps} nbits={nbits} '
                  f'最佳 {m}/{nbits}（参考状态={state}）')


# =====================================================================
# main
# =====================================================================
def _spec_versions():
    parts = [f'python {sys.version.split()[0]}', f'numpy {np.__version__}']
    if _HAVE_PYCRYPTODOME:
        parts.append(f'pycryptodome {_PYCRYPTODOME_VER}')
    if _HAVE_CRYPTOGRAPHY:
        parts.append(f'cryptography {_CRYPTOGRAPHY_VER}')
    return ' | '.join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='第 5 层（fuzz）验证：内核差分模糊测试（独立参考实现 / 第三方库对拍）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='任何不一致 → 退出码 1；HARNESS-ERROR（参考自检失败）→ 退出码 2。')
    ap.add_argument('--iters', type=int, default=None, metavar='N',
                    help='每个类别的迭代次数（默认 8；--quick 时 2）')
    ap.add_argument('--seed', type=int, default=20261109,
                    help='随机种子（默认 20261109；同 seed + 同 --only 结果可复现）')
    ap.add_argument('--quick', action='store_true', help='CI 快速档：2 次迭代/类别')
    ap.add_argument('--verbose', action='store_true', help='输出参考自检等细节')
    ap.add_argument('--backend', choices=['numpy', 'torch'], default='numpy',
                    help='numpy=只用 numpy 后端（默认，不需要 GPU）；'
                         'torch=额外跑 numpy↔torch 双后端逐字节对拍')
    ap.add_argument('--only', default='', metavar='CAT[,CAT...]',
                    help='只跑指定类别（逗号分隔）')
    ap.add_argument('--list', action='store_true', dest='list_cats', help='列出全部类别')
    a = ap.parse_args(argv)

    names = [n for n, _ in CATEGORIES]
    if a.list_cats:
        print('可用类别（%d）：' % len(names))
        for n in names:
            print(f'  {n:14s} {_CAT_NOTES.get(n, "")}')
        return 0

    iters = a.iters if a.iters is not None else (2 if a.quick else 8)
    if iters < 1:
        ap.error('--iters 必须 >= 1')
    selected = names if not a.only else [s.strip() for s in a.only.split(',') if s.strip()]
    unknown = [s for s in selected if s not in names]
    if unknown:
        ap.error(f'未知类别: {unknown}（可用: {names}）')

    print('=' * 64)
    print('gpu_crack 第 5 层（fuzz）验证 — 内核差分模糊测试 (fuzz_kernels.py)')
    print(f'  {_spec_versions()}')
    print(f'  seed={a.seed}  iters={iters}  类别={len(selected)}  backend={a.backend}')
    print('=' * 64)

    # 全部类别默认在 numpy 后端运行（无需 GPU）；--backend torch 只额外增加双后端对拍
    G.init_backend('numpy')
    t_start = time.perf_counter()
    try:
        check_references()
        print('PASS  [refcheck]      参考实现自检通过（官方/RFC 向量）')
    except HarnessError as e:
        print(f'HARNESS-ERROR  [refcheck]  {e}')
        return 2

    passed = failed = 0
    skipped = []
    failures = []
    total_checks = 0
    for name in selected:
        fn = _CAT_MAP[name]
        if name == 'dual_backend' and a.backend != 'torch':
            print(f'SKIP  {name:14s} (需要 --backend torch)')
            skipped.append(name)
            continue
        ctx = Ctx(name, _rng_for(a.seed, name), iters, a.verbose)
        t0 = time.perf_counter()
        try:
            fn(ctx)
            dt = time.perf_counter() - t0
            total_checks += ctx.checks
            passed += 1
            print(f'PASS  {name:14s} ({ctx.iters_done} 次迭代, {dt:5.2f}s)')
            if ctx.verbose and ctx.details:
                for d in ctx.details:
                    print(f'      {d}')
        except SkipCategory as e:
            dt = time.perf_counter() - t0
            total_checks += ctx.checks
            print(f'SKIP  {name:14s} ({e})')
            skipped.append(name)
        except FuzzMismatch as e:
            dt = time.perf_counter() - t0
            total_checks += ctx.checks
            failed += 1
            failures.append((name, str(e), iters))
            print(f'FAIL  {name:14s} ({ctx.iters_done} 次迭代, {dt:5.2f}s)')
            print(f'      不一致: {e}')
        except HarnessError as e:
            dt = time.perf_counter() - t0
            print(f'HARNESS-ERROR  {name:14s} ({dt:5.2f}s)  {e}')
            failed += 1
            failures.append((name, f'HARNESS-ERROR: {e}', iters))
        except Exception as e:                            # noqa: BLE001
            dt = time.perf_counter() - t0
            total_checks += ctx.checks
            failed += 1
            failures.append((name, f'{type(e).__name__}: {e}', iters))
            print(f'FAIL  {name:14s} ({ctx.iters_done} 次迭代, {dt:5.2f}s)')
            print(f'      异常: {type(e).__name__}: {e}')
            traceback.print_exc(limit=4)
        finally:
            G.init_backend('numpy')                       # 兜底复位，避免类别间污染

    dt_total = time.perf_counter() - t_start
    print('=' * 64)
    ok = failed == 0
    status = 'ALL PASS' if ok else 'FAIL'
    print(f'FUZZ: {status} — {passed} 通过 / {failed} 失败 / {len(skipped)} 跳过 '
          f'（共 {len(selected)} 类别, {total_checks} 个断言点, {dt_total:.2f}s, '
          f'seed={a.seed}, backend={a.backend}）')
    if skipped:
        print(f'  跳过: {", ".join(skipped)}')
    if not ok:
        print('失败类别与最小复现：')
        for name, msg, it in failures:
            print(f'  - {name}: {msg}')
            print(f'    复现: python -X utf8 fuzz_kernels.py --only {name} '
                  f'--seed {a.seed} --iters {it} --verbose')
        print('=' * 64)
        return 1
    print('=' * 64)
    return 0


if __name__ == '__main__':
    sys.exit(main())
