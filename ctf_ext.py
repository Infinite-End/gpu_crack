#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf_ext.py — CTF 算法族内核（与 gpu_crack.py 共用后端）

子命令
------
  tea        TEA / XTEA / XXTEA 加解密（key 固定 16B；TEA/XTEA 数据须 8B 倍数）
  speck      Speck32/64、Speck64/128 加解密（NSA 组合 16/4、32/3、32/4）
  rc4        RC4 加解密（key 1~256B）
  lcg        LCG 参数恢复 / 状态预测（--mod 支持 2**32 等表达式）
  md5        MD5 批量哈希 / 口令爆破（含空口令）
  rsa        RSA 多攻击自动尝试：小因子 → Wiener → Fermat → rho → p-1（结果经 m^e≡c 校验）
  encode     多层编码 BFS 自动解码（hex/b64/b32/url/rot13/rot47/反转/单字节 XOR）
  selftest   已知向量自测（失败时退出码 = 失败项数）

全局参数（必须写在子命令之前）
------------------------------
  --backend {auto,torch,numpy}   默认 auto（本模块无 --chunk）

输入约定
--------
  * 整数参数（--n/--c/--e/--outs/--mod、Speck 的 key/data）接受十进制、0x 前缀或裸 hex；
    字节串参数（tea/rc4/encode 的 --data-hex）必须是裸 hex。
  * 退出码：0=成功（未攻破/无结果也是 0）；1..N=selftest 失败项数；2=参数或输入错误
    （stderr 一行 "ctf_ext.py: error: ..."，不抛 traceback）。

用法
----
  python ctf_ext.py selftest
  python ctf_ext.py tea    --mode xtea --key-hex 000102... --data-hex aabb... --op decrypt
  python ctf_ext.py speck  --key-hex 1918111009080100 --data-hex 6574694c --word-bits 16 --key-words 4
  python ctf_ext.py rc4    --key-hex 4b6579 --data-hex 506c61696e74657874
  python ctf_ext.py lcg    --mod 2**32 --outs 1,2,3,4 --predict 3
  python ctf_ext.py md5    --target 900150983cd24fb0d6963f7d28e17f72 --charset abc --max-len 4
  python ctf_ext.py rsa    --n <hex> --e 65537 --c <hex>
  python ctf_ext.py encode --data-hex 666c61677b746573747d   (或 --data "flag{test}")

完整参数/输出/退出码见 E:\\gpu_crack\\CLI_REFERENCE.md。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import string
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_crack as G    # noqa: E402

M32 = 0xFFFFFFFF


# =====================================================================
# CLI 公共辅助：干净报错（不抛 traceback）
# =====================================================================
def _err(msg):
    """打印一行干净的错误信息并以退出码 2 结束（等价于 argparse 报错）。"""
    print(f'ctf_ext.py: error: {msg}', file=sys.stderr)
    raise SystemExit(2)


def _hex_arg(value, name):
    """解析命令行十六进制参数；非法输入 -> 干净报错而不是 traceback。"""
    try:
        return bytes.fromhex(value)
    except ValueError:
        _err(f'{name} 必须是十六进制字符串（偶数个十六进制字符），收到 {value!r}')


def _iroot(x, k):
    """整数 k 次方根 floor(x**(1/k))，纯整数二进制搜索（大整数上比 float 精确）。"""
    if x < 2:
        return max(x, 0)
    lo, hi = 0, 1 << ((x.bit_length() + k - 1) // k)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _parse_mod(s):
    """解析 LCG 模数：支持十进制 / 0x 前缀 / 纯 hex / `2**32` 这类幂表达式。"""
    s = str(s).strip().replace('_', '').replace(' ', '')
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1].strip()
    if '**' in s:
        base_s, _, exp_s = s.partition('**')
        base = _parse_int(base_s)
        try:
            exp = int(exp_s, 10)
        except ValueError:
            _err(f'--mod 指数无法解析: {exp_s!r}')
        if not 1 <= exp <= 4096:
            _err(f'--mod 指数必须在 1..4096 之间，收到 {exp}')
        return base ** exp
    return _parse_int(s)


# =====================================================================
# TEA / XTEA / XXTEA
# =====================================================================
DELTA = 0x9E3779B9


def tea_encrypt_block(v0, v1, k, rounds=32):
    """TEA 单块加密（标量），返回 (v0, v1)。"""
    s = 0
    for _ in range(rounds):
        s = (s + DELTA) & M32
        v0 = (v0 + ((((v1 << 4) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1])) & M32)) & M32
        v1 = (v1 + ((((v0 << 4) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3])) & M32)) & M32
    return v0, v1


def tea_decrypt_block(v0, v1, k, rounds=32):
    s = (DELTA * rounds) & M32
    for _ in range(rounds):
        v1 = (v1 - ((((v0 << 4) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3])) & M32)) & M32
        v0 = (v0 - ((((v1 << 4) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1])) & M32)) & M32
        s = (s - DELTA) & M32
    return v0, v1


def xtea_encrypt_block(v0, v1, k, rounds=32):
    s = 0
    for _ in range(rounds):
        v0 = (v0 + ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + k[s & 3]))) & M32
        s = (s + DELTA) & M32
        v1 = (v1 + ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + k[(s >> 11) & 3]))) & M32
    return v0, v1


def xtea_decrypt_block(v0, v1, k, rounds=32):
    s = (DELTA * rounds) & M32
    for _ in range(rounds):
        v1 = (v1 - ((((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + k[(s >> 11) & 3]))) & M32
        s = (s - DELTA) & M32
        v0 = (v0 - ((((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + k[s & 3]))) & M32
    return v0, v1


def _xxtea_mx(z, y, s, k, p, e):
    return ((((z >> 5) ^ (y << 2)) + ((y >> 3) ^ (z << 4))) ^ ((s ^ y) + (k[(p & 3) ^ e] ^ z))) & M32


def xxtea_encrypt(data, key, rounds=None):
    """XXTEA 加密（标准 MX 结构；data 长度自动补齐到 4 的倍数）。"""
    v = [int(x) for x in np.frombuffer(data + b'\x00' * ((-len(data)) % 4), dtype='<u4')]
    if len(v) < 2:                          # XXTEA 需要至少 2 个 32 位字
        v += [0] * (2 - len(v))
    n = len(v)
    k = [int(x) for x in np.frombuffer(key[:16].ljust(16, b'\x00'), dtype='<u4')]
    if rounds is None:
        rounds = 6 + 52 // n
    total = 0
    z = v[n - 1]
    for _ in range(rounds):
        total = (total + DELTA) & M32
        e = (total >> 2) & 3
        for p in range(n):
            y = v[(p + 1) % n]
            mx = ((((z >> 5) ^ ((y << 2) & M32)) + (((y >> 3) ^ ((z << 4) & M32)) & M32)) & M32) ^ \
                 (((total ^ y) + (k[(p & 3) ^ e] ^ z)) & M32)
            v[p] = (v[p] + (mx & M32)) & M32
            z = v[p]
    return np.array(v, dtype='<u4').tobytes()


def xxtea_decrypt(data, key, rounds=None):
    """XXTEA 解密（与 encrypt 严格互逆）。"""
    v = [int(x) for x in np.frombuffer(data, dtype='<u4')]
    n = len(v)
    if n < 2:
        return data
    k = [int(x) for x in np.frombuffer(key[:16].ljust(16, b'\x00'), dtype='<u4')]
    if rounds is None:
        rounds = 6 + 52 // n
    total = (rounds * DELTA) & M32
    y = v[0]
    for _ in range(rounds):
        e = (total >> 2) & 3
        for p in range(n - 1, -1, -1):
            z = v[(p - 1) % n]
            mx = ((((z >> 5) ^ ((y << 2) & M32)) + (((y >> 3) ^ ((z << 4) & M32)) & M32)) & M32) ^ \
                 (((total ^ y) + (k[(p & 3) ^ e] ^ z)) & M32)
            v[p] = (v[p] - (mx & M32)) & M32
            y = v[p]
        total = (total - DELTA) & M32
    return np.array(v, dtype='<u4').tobytes()


# =====================================================================
# Speck（NSA 轻量级分组密码）
# =====================================================================
def _ror(x, n, w):
    return ((x >> n) | (x << (w - n))) & ((1 << w) - 1)


def _rol(x, n, w):
    return ((x << n) | (x >> (w - n))) & ((1 << w) - 1)


# NSA 规格 (word_bits, key_words) -> 轮数
SPECK_ROUNDS = {(16, 4): 22, (32, 3): 26, (32, 4): 27,
                (64, 2): 32, (64, 3): 33, (64, 4): 34}


def speck_round_keys(key, word_bits=32, key_words=4, rounds=None):
    """
    Speck 密钥编排（NSA 规格：S^-alpha 为循环右移 alpha，S^beta 为循环左移 beta）。
    key 按小端拆词：k[0] 最低位。
    """
    w = word_bits
    alpha, beta = (7, 2) if w == 16 else (8, 3)
    if rounds is None:
        rounds = SPECK_ROUNDS.get((w, key_words), 27)
    mask = (1 << w) - 1
    k = [(key >> (w * i)) & mask for i in range(key_words)]
    l = k[1:]
    ks = [k[0]]
    for i in range(rounds - 1):
        # l_s = (k[i] + S^-alpha(l[i])) ^ i
        l_new = ((ks[i] + _ror(l[i % len(l)] if l else 0, alpha, w)) & mask) ^ (i & mask)
        k_new = _rol(ks[i], beta, w) ^ l_new
        if l:
            l[i % len(l)] = l_new
        ks.append(k_new & mask)
    return ks, alpha, beta, rounds


def speck_encrypt(pt, key, word_bits=32, key_words=4, rounds=None):
    """
    Speck 加密单块（pt 为 2w 位整数，高 w 位是 x，低 w 位是 y）。
    Speck32/64: word=16, 4 词, 22 轮  |  Speck64/128: word=32, 4 词, 27 轮
    """
    w = word_bits
    mask = (1 << w) - 1
    ks, alpha, beta, T = speck_round_keys(key, w, key_words, rounds)
    x = (pt >> w) & mask
    y = pt & mask
    for i in range(T):
        x = (((_ror(x, alpha, w) + y) & mask) ^ ks[i]) & mask
        y = (_rol(y, beta, w) ^ x) & mask
    return ((x << w) | y) & ((1 << (2 * w)) - 1)


def speck_decrypt(ct, key, word_bits=32, key_words=4, rounds=None):
    """Speck 解密单块（encrypt 的严格逆运算）。"""
    w = word_bits
    mask = (1 << w) - 1
    ks, alpha, beta, T = speck_round_keys(key, w, key_words, rounds)
    x = (ct >> w) & mask
    y = ct & mask
    for i in range(T - 1, -1, -1):
        y = _ror((y ^ x) & mask, beta, w)
        x = _rol(((x ^ ks[i]) - y) & mask, alpha, w)
    return ((x << w) | y) & ((1 << (2 * w)) - 1)


# =====================================================================
# RC4
# =====================================================================
def rc4_keystream(key, n):
    S = list(range(256))
    j = 0
    kl = len(key)
    for i in range(256):
        j = (j + S[i] + key[i % kl]) & 0xFF
        S[i], S[j] = S[j], S[i]
    out = bytearray()
    i = j = 0
    for _ in range(n):
        i = (i + 1) & 0xFF
        j = (j + S[i]) & 0xFF
        S[i], S[j] = S[j], S[i]
        out.append(S[(S[i] + S[j]) & 0xFF])
    return bytes(out)


def rc4(data, key):
    ks = rc4_keystream(key, len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


# =====================================================================
# LCG
# =====================================================================
def lcg_recover(outputs, modulus):
    """
    已知连续输出与模数，恢复 a, c（a*x+c mod m）。
    需要至少 3 个输出。
    方法：t_i = x_{i+1} - x_i 满足 t_{i+1} ≡ a*t_i (mod m)
    """
    xs = list(outputs)
    if len(xs) < 3:
        return None
    t = [(xs[i + 1] - xs[i]) % modulus for i in range(len(xs) - 1)]
    # a ≡ t_{i+1} * t_i^{-1} (mod m)，用 gcd 处理不可逆情形
    a = None
    for i in range(len(t) - 1):
        g = math.gcd(t[i], modulus)
        if g == 1:
            a = (t[i + 1] * pow(t[i], -1, modulus)) % modulus
            break
        # 部分可逆：解 t_{i+1} ≡ a t_i (mod m)
        if t[i + 1] % g == 0:
            m2, t2, r2 = modulus // g, t[i] // g, t[i + 1] // g
            if math.gcd(t2, m2) == 1:
                a0 = (r2 * pow(t2, -1, m2)) % m2
                a = a0
                break
    if a is None:
        return None
    c = (xs[1] - a * xs[0]) % modulus
    # 验证
    for i in range(len(xs) - 1):
        if (a * xs[i] + c) % modulus != xs[i + 1]:
            return None
    return a, c


def lcg_predict(outputs, modulus, n=1):
    r = lcg_recover(outputs, modulus)
    if not r:
        return None
    a, c = r
    x = list(outputs)
    out = []
    for _ in range(n):
        nx = (a * x[-1] + c) % modulus
        x.append(nx)
        out.append(nx)
    return out


# =====================================================================
# MD5（批量）
# =====================================================================
_MD5_S = [7,12,17,22]*4 + [5,9,14,20]*4 + [4,11,16,23]*4 + [6,10,15,21]*4
_MD5_K = [int(abs(math.sin(i + 1)) * 2**32) & M32 for i in range(64)]


def md5_batch(msgs):
    """批量 MD5：msgs 为 list[bytes]，返回 list[bytes]（16 字节）。用 hashlib（C 实现已足够快）。"""
    return [hashlib.md5(m).digest() for m in msgs]


def md5_crack(target_hex, charset, max_len, limit=None):
    """暴力 MD5（长度 0 ≤ L ≤ max_len）。返回命中的明文。"""
    tgt = bytes.fromhex(target_hex)
    tried = 0
    t0 = time.time()
    for L in range(0, max_len + 1):        # L=0：空口令也是合法候选
        for combo in itertools.product(charset, repeat=L):
            s = ''.join(combo).encode()
            tried += 1
            if hashlib.md5(s).digest() == tgt:
                return s, tried, time.time() - t0
            if limit and tried >= limit:
                return None, tried, time.time() - t0
    return None, tried, time.time() - t0


# =====================================================================
# RSA 工具箱
# =====================================================================
def rsa_fermat(n, iters=1_000_000):
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    for _ in range(iters):
        b2 = a * a - n
        b = math.isqrt(b2)
        if b * b == b2:
            return a - b, a + b
        a += 1
    return None


def rsa_wiener(n, e):
    def cf(a, b):
        while b:
            q = a // b
            yield q
            a, b = b, a - q * b

    p0, p1, q0, q1 = 0, 1, 1, 0
    for k in cf(e, n):
        p0, p1 = p1, k * p1 + p0
        q0, q1 = q1, k * q1 + q0
        if p1 == 0 or q1 == 0:
            continue
        if (e * q1 - 1) % p1:
            continue
        phi = (e * q1 - 1) // p1
        s = n - phi + 1
        d_ = s * s - 4 * n
        if d_ < 0:
            continue
        r = math.isqrt(d_)
        if r * r == d_:
            return q1, (s + r) // 2
    return None


def rsa_common_modulus(n, e1, c1, e2, c2):
    """共模攻击：同一 m，两个 (e,c)。m = c1^a * c2^b (mod n)，a*e1+b*e2=1。"""
    g, a, b = _egcd(e1, e2)
    if g != 1:
        return None
    try:
        # Bézout 系数为负时，先把对应密文取模逆，指数转为正数（不能把指数置 0！）
        if a < 0:
            c1 = pow(c1, -1, n)
            a = -a
        if b < 0:
            c2 = pow(c2, -1, n)
            b = -b
        return (pow(c1, a, n) * pow(c2, b, n)) % n
    except ValueError:          # c 与 n 不互素，无法求逆
        return None


def _egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def rsa_broadcast(pairs, e):
    """低指数广播攻击：pairs = [(n_i, c_i)]，用 CRT 合并后开 e 次方。"""
    N = 1
    for n, _ in pairs:
        N *= n
    x = 0
    try:
        for n, c in pairs:
            Ni = N // n
            x = (x + c * Ni * pow(Ni, -1, n)) % N
    except ValueError:          # 模数之间不互素，CRT 不成立
        return None
    r = _iroot(x, e)            # 精确整数开方：float 版本在 m 超过 ~2^53 后误差巨大
    for cand in range(max(0, r - 2), r + 3):
        if cand ** e == x:
            return cand
    return None


def rsa_pollard_rho(n, max_iter=1_000_000):
    if n % 2 == 0:
        return 2
    for c in (1, 3, 5, 7):
        x = y = 2
        d = 1
        f = lambda v: (v * v + c) % n
        it = 0
        while d == 1 and it < max_iter:
            x = f(x); y = f(f(y))
            d = math.gcd(abs(x - y), n)
            it += 1
        if 1 < d < n:
            return d
    return None


def rsa_pm1(n, B=100000):
    """Pollard p-1：按素数幂 p^k (p^k ≤ B) 累乘指数，p-1 含小素数平方因子时也能成功。"""
    a = 2
    for p in _primes_upto(B):
        pk = p
        while pk * p <= B:
            pk *= p
        a = pow(a, pk, n)
        g = math.gcd(a - 1, n)
        if 1 < g < n:
            return g
    return None


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _rsa_verify(n, e, c, m):
    """校验 m 确实是密文 c 对应的明文（防止攻击误报把错误结果当成答案）。"""
    return m is not None and pow(m, e, n) == c % n


def rsa_attack(n, e, c, verbose=True):
    """按性价比顺序尝试多种攻击。返回 (方法, 明文或 None)。"""
    if n < 4:
        return None, None
    if verbose:
        print(f'[rsa] n={n.bit_length()}bit  e={e}')
    # 1) n 有平方因子 / 小因子
    for p in _primes_upto(100000):
        if n % p == 0 and 1 < p < n:
            m = _decrypt_with_pq(n, e, c, p, n // p)
            if _rsa_verify(n, e, c, m):
                return 'small-factor', m
    if verbose:
        print('  small-factor: 无')
    # 2) Wiener
    w = rsa_wiener(n, e)
    if w:
        d = w[0]
        m = pow(c, d, n)
        if _rsa_verify(n, e, c, m):
            return 'wiener', m
    if verbose:
        print('  wiener: 无')
    # 3) Fermat（p,q 接近）
    f = rsa_fermat(n, 200000)
    if f and 1 < f[0] < n and 1 < f[1] < n:
        m = _decrypt_with_pq(n, e, c, f[0], f[1])
        if _rsa_verify(n, e, c, m):
            return 'fermat', m
    if verbose:
        print('  fermat(2e5): 无')
    # 4) Pollard rho
    r = rsa_pollard_rho(n)
    if r:
        m = _decrypt_with_pq(n, e, c, r, n // r)
        if _rsa_verify(n, e, c, m):
            return 'pollard-rho', m
    if verbose:
        print('  pollard-rho: 无')
    # 5) Pollard p-1
    r = rsa_pm1(n)
    if r:
        m = _decrypt_with_pq(n, e, c, r, n // r)
        if _rsa_verify(n, e, c, m):
            return 'pollard-p1', m
    if verbose:
        print('  pollard-p1: 无')
    return None, None


def _decrypt_with_pq(n, e, c, p, q):
    """已知因子 p,q 解出明文；非法 RSA 参数（p≤1 / e 与 φ 不互素）返回 None。"""
    if p <= 1 or q <= 1 or p * q != n:
        return None
    phi = p * (q - 1) if p == q else (p - 1) * (q - 1)   # n=p^2 时 φ=p(p-1)
    try:
        d = pow(e, -1, phi)
    except ValueError:
        return None
    return pow(c, d, n)


# =====================================================================
# 多层编码自动解码
# =====================================================================
def _try_decoders(data: bytes):
    """返回 [(名字, 结果bytes)]"""
    out = []
    s = data.strip()
    # hex
    if len(s) % 2 == 0 and all(chr(b) in '0123456789abcdefABCDEF' for b in s):
        try:
            out.append(('hex', bytes.fromhex(s.decode())))
        except Exception:
            pass
    # base64 / base32（b32 允许小写字母）
    import base64
    for name in ('b64', 'b32'):
        try:
            if name == 'b64':
                r = base64.b64decode(s + b'=' * (-len(s) % 4))
            else:
                r = base64.b32decode(s + b'=' * (-len(s) % 8), casefold=True)
            if r:
                out.append((name, r))
        except Exception:
            pass
    # rot13 / rot47
    out.append(('rot13', bytes([_rot13(b) for b in s])))
    out.append(('rot47', bytes([(33 + (b - 33 + 47) % 94) if 33 <= b <= 126 else b for b in s])))
    # URL（unquote_to_bytes 可直接吃 bytes，非 ASCII 字节不会被二次 UTF-8 编码破坏）
    try:
        import urllib.parse
        out.append(('url', urllib.parse.unquote_to_bytes(s)))
    except Exception:
        pass
    # 反转
    out.append(('reversed', s[::-1]))
    # 单字节异或：只保留可打印率高且带 flag 特征的，按得分取前 8（避免淹没 BFS 前沿）
    xor_cands = []
    for k in range(1, 256):
        r = bytes(b ^ k for b in s)
        ratio = sum(32 <= c < 127 for c in r) / max(1, len(r))
        if ratio > 0.95:
            score = ratio * 10 + (8 if any(t in r for t in (b'{', b'}', b'_', b'flag')) else 0)
            xor_cands.append((score, k, r))
    xor_cands.sort(reverse=True)
    for _, k, r in xor_cands[:8]:
        out.append((f'xor{k:02x}', r))
    return out


def _rot13(b):
    if 65 <= b <= 90:
        return 65 + (b - 65 + 13) % 26
    if 97 <= b <= 122:
        return 97 + (b - 97 + 13) % 26
    return b


def _looks_flag(b):
    return any(t in b for t in (b'flag{', b'CTF{', b'ctf{', b'gylike{', b'EAI{'))


def auto_decode(data, max_depth=4, verbose=True):
    """广度优先多层解码，优先返回像 flag 的结果。"""
    frontier = [(data, [])]
    seen = {data}
    for depth in range(max_depth):
        nxt = []
        for cur, path in frontier:
            if _looks_flag(cur):
                return cur, path
            for name, res in _try_decoders(cur):
                if res in seen or len(res) == 0:
                    continue
                seen.add(res)
                nxt.append((res, path + [name]))
                if _looks_flag(res):
                    return res, path + [name]
        if verbose:
            print(f'  depth {depth+1}: {len(nxt)} 个候选, 示例 {nxt[:3]}')
        frontier = nxt[:200]
    return None, None


# =====================================================================
# 子命令
# =====================================================================
def cmd_speck(a):
    """Speck 单块加解密（--data-hex 为 2*word_bits/8 字节，大端整数解释）。"""
    w = a.word_bits
    if a.key_words not in (2, 3, 4):
        _err(f'--key-words 只支持 2/3/4，收到 {a.key_words}')
    if a.rounds is None and (w, a.key_words) not in SPECK_ROUNDS:
        _err(f'不支持的参数组合 --word-bits {w} --key-words {a.key_words}'
             f'（NSA 规格为 16/4、32/3、32/4；自定义轮数请显式给 --rounds）')
    if a.rounds is not None and a.rounds < 1:
        _err(f'--rounds 必须 >= 1，收到 {a.rounds}')
    try:
        key = int(a.key_hex.replace(' ', ''), 16)
    except ValueError:
        _err(f'--key-hex 必须是十六进制字符串，收到 {a.key_hex!r}')
    try:
        data = int(a.data_hex.replace(' ', ''), 16)
    except ValueError:
        _err(f'--data-hex 必须是十六进制字符串，收到 {a.data_hex!r}')
    if key.bit_length() > w * a.key_words:
        _err(f'--key-hex 超过 {w}bit x {a.key_words} 词的密钥宽度')
    if data.bit_length() > 2 * w:
        _err(f'--data-hex 超过单块 {2 * w}bit 宽度')
    if a.op == 'encrypt':
        out = speck_encrypt(data, key, word_bits=w, key_words=a.key_words, rounds=a.rounds)
    else:
        out = speck_decrypt(data, key, word_bits=w, key_words=a.key_words, rounds=a.rounds)
    nbytes = (2 * w) // 8
    print(f'[speck/{a.op}] w={w} keys={a.key_words}  {out:0{nbytes*2}x}  ({out.to_bytes(nbytes, "big")!r})')


def cmd_selftest(a):
    print(f'[backend] {G.BACKEND} {G.DEV}\n')
    fails = 0            # 任一 FAIL → 返回非零，供上层集成脚本判断

    # TEA 已知向量：pt(0,0), key(0,0,0,0) → 41ea3a0a 94baa940
    v0, v1 = tea_encrypt_block(0, 0, [0, 0, 0, 0])
    ok = (v0, v1) == (0x41EA3A0A, 0x94BAA940)
    print(f'[TEA 向量] {"PASS" if ok else "FAIL"}  {v0:08x} {v1:08x}')
    fails += not ok
    d0, d1 = tea_decrypt_block(v0, v1, [0, 0, 0, 0])
    ok = (d0, d1) == (0, 0)
    print(f'[TEA 往返] {"PASS" if ok else "FAIL"}')
    fails += not ok

    # XTEA 往返
    k = [0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210]
    e0, e1 = xtea_encrypt_block(0x12345678, 0x9ABCDEF0, k)
    b0, b1 = xtea_decrypt_block(e0, e1, k)
    ok = (b0, b1) == (0x12345678, 0x9ABCDEF0)
    print(f'[XTEA 往返] {"PASS" if ok else "FAIL"}  ct={e0:08x}{e1:08x}')
    fails += not ok

    # XXTEA 往返
    key = b'0123456789abcdef'
    msg = b'flag{xxtea_test}'
    ct = xxtea_encrypt(msg, key)
    pt = xxtea_decrypt(ct, key)[:len(msg)]
    ok = pt == msg
    print(f'[XXTEA 往返] {"PASS" if ok else "FAIL"}')
    fails += not ok

    # Speck32/64 已知向量：key=1918 1110 0908 0100, pt=6574 694c → ct=a868 42f2
    ct = speck_encrypt(0x6574694C, 0x1918111009080100, word_bits=16, key_words=4)
    ok = ct == 0xA86842F2
    print(f'[Speck32/64 向量] {"PASS" if ok else "FAIL"}  {ct:08x}')
    fails += not ok

    # Speck64/128 已知向量：key=1b1a1918 13121110 0b0a0908 03020100,
    # pt=3b726574 7475432d → ct=8c6fa548 454e028b
    ct64 = speck_encrypt(0x3B7265747475432D, 0x1B1A1918131211100B0A090803020100,
                         word_bits=32, key_words=4)
    ok = ct64 == 0x8C6FA548454E028B
    print(f'[Speck64/128 向量] {"PASS" if ok else "FAIL"}  {ct64:016x}')
    fails += not ok

    # RC4 已知向量
    ok = rc4(b'Plaintext', b'Key') == bytes.fromhex('BBF316E8D940AF0AD3')
    print(f'[RC4 向量] {"PASS" if ok else "FAIL"}')
    fails += not ok

    # LCG 恢复
    m = 2 ** 31 - 1; aa, cc, x = 1103515245, 12345, 42
    seq = []
    for _ in range(6):
        x = (aa * x + cc) % m
        seq.append(x)
    r = lcg_recover(seq, m)
    ok = r == (aa, cc)
    print(f'[LCG 恢复] {"PASS" if ok else "FAIL"}  a={r[0] if r else None} c={r[1] if r else None}')
    fails += not ok
    print(f'  预测下 3 个: {lcg_predict(seq, m, 3)}')

    # MD5
    ok = md5_batch([b'abc'])[0].hex() == '900150983cd24fb0d6963f7d28e17f72'
    print(f'[MD5 向量] {"PASS" if ok else "FAIL"}')
    fails += not ok
    t0 = time.time()
    hit, tried, dt = md5_crack(hashlib.md5(b'abc').hexdigest(), string.ascii_lowercase, 3)
    ok = hit == b"abc"
    print(f'[MD5 爆破] {"PASS" if ok else "FAIL"}  {tried} 次 / {dt:.3f}s')
    fails += not ok

    # RSA：构造小密钥自测
    p, q = 1000003, 1000033
    n = p * q; e = 65537
    msg = 0x1234
    c = pow(msg, e, n)
    meth, dec = rsa_attack(n, e, c, verbose=False)
    ok = dec == msg
    print(f'[RSA 攻击] {"PASS" if ok else "FAIL"}  方法={meth}')
    fails += not ok

    # RSA 共模攻击（Bézout 系数 -1 的路径，旧实现返回错误结果）
    n_cm = 10007 * 1000033
    c1, c2 = pow(42, 3, n_cm), pow(42, 65537, n_cm)
    ok = rsa_common_modulus(n_cm, 3, c1, 65537, c2) == 42
    print(f'[RSA 共模] {"PASS" if ok else "FAIL"}')
    fails += not ok

    # RSA 低指数广播（m 远大于 2^53，检验整数开方精度）
    m_bc = 0xC0FFEE
    pairs = [(10007 * 10009, 0), (10037 * 10039, 0), (10061 * 10067, 0)]
    pairs = [(ni, pow(m_bc, 3, ni)) for ni, _ in pairs]
    ok = rsa_broadcast(pairs, 3) == m_bc
    print(f'[RSA 广播] {"PASS" if ok else "FAIL"}')
    fails += not ok

    # RSA p-1（p-1 = 2^15 * 94307 含素数幂，旧实现只累乘一次小素数会失败）
    p_pm1 = 2 ** 15 * 94307 + 1         # = 3090251777，素数
    ok = rsa_pm1(p_pm1 * 1000003) == p_pm1
    print(f'[RSA p-1 素数幂] {"PASS" if ok else "FAIL"}')
    fails += not ok

    # 编码自动解码
    import base64
    payload = base64.b64encode(b'666c61677b7478747d')     # b64(hex(flag{...}))
    got, path = auto_decode(payload, verbose=False)
    ok = bool(got)
    print(f'[自动解码] {"PASS" if ok else "FAIL"}  路径={path} 结果={got[:24] if got else None}')
    fails += not ok
    return fails


def cmd_tea(a):
    data = _hex_arg(a.data_hex, '--data-hex')
    key = _hex_arg(a.key_hex, '--key-hex')
    if len(key) != 16:
        _err(f'--key-hex 必须是 16 字节（128 位）密钥，收到 {len(key)} 字节')
    if a.mode in ('tea', 'xtea'):
        if len(data) % 8:
            _err(f'{a.mode.upper()} 的 --data-hex 必须是 8 字节块的整数倍，收到 {len(data)} 字节')
    else:                               # xxtea：加密自动按 4 字节补齐，解密要求密文合法
        if a.op == 'decrypt':
            if len(data) % 4:
                _err(f'XXTEA 的 --data-hex 必须是 4 字节的整数倍，收到 {len(data)} 字节')
            if 0 < len(data) < 8:
                _err(f'XXTEA 密文至少 8 字节，收到 {len(data)} 字节')
    k = [int(x) for x in np.frombuffer(key, dtype='<u4')]   # 必须转 Python int
    if a.mode == 'tea':
        f = tea_encrypt_block if a.op == 'encrypt' else tea_decrypt_block
    elif a.mode == 'xtea':
        f = xtea_encrypt_block if a.op == 'encrypt' else xtea_decrypt_block
    else:
        out = xxtea_encrypt(data, key) if a.op == 'encrypt' else xxtea_decrypt(data, key)
        print(f'[{a.mode} {a.op}] {out.hex()}')
        return
    out = bytearray()
    for i in range(0, len(data) - 7, 8):
        v0, v1 = int.from_bytes(data[i:i + 4], 'little'), int.from_bytes(data[i + 4:i + 8], 'little')
        r0, r1 = f(v0, v1, k)
        out += r0.to_bytes(4, 'little') + r1.to_bytes(4, 'little')
    print(f'[{a.mode} {a.op}] {bytes(out).hex()}')


def cmd_rc4(a):
    key = _hex_arg(a.key_hex, '--key-hex')
    data = _hex_arg(a.data_hex, '--data-hex')
    if not key:
        _err('--key-hex 不能为空（RC4 密钥长度 1..256 字节）')
    if len(key) > 256:
        _err(f'--key-hex 最长 256 字节，收到 {len(key)} 字节')
    print(f'[rc4] {rc4(data, key).hex()}')


def cmd_lcg(a):
    try:
        mod = _parse_mod(a.mod)
    except ValueError as ex:
        _err(str(ex))
    if mod < 2:
        _err(f'--mod 必须 >= 2，收到 {mod}')
    xs = []
    for part in str(a.outs).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            xs.append(_parse_int(part))
        except ValueError as ex:
            _err(f'--outs 解析失败: {ex}')
    if len(xs) < 3:
        _err(f'--outs 至少需要 3 个连续输出，收到 {len(xs)} 个')
    if a.predict < 0:
        _err(f'--predict 不能为负，收到 {a.predict}')
    xs = [x % mod for x in xs]          # C 的有符号打印也能正确当余数处理
    r = lcg_recover(xs, mod)
    print(f'[lcg] 恢复: a={r[0] if r else None} c={r[1] if r else None}')
    if r:
        print(f'  预测 {a.predict} 个: {lcg_predict(xs, mod, a.predict)}')


def cmd_md5(a):
    tgt = str(a.target).strip().replace(' ', '')
    if len(tgt) != 32 or any(ch not in string.hexdigits for ch in tgt):
        _err(f'--target 必须是 32 个十六进制字符的 MD5 值，收到 {a.target!r}')
    if a.max_len < 0:
        _err(f'--max-len 不能为负，收到 {a.max_len}')
    hit, tried, dt = md5_crack(tgt, a.charset, a.max_len)
    print(f'[md5] 尝试 {tried:,} 次 / {dt:.2f}s  {"命中: " + repr(hit) if hit is not None else "未命中"}')


def _parse_int(s):
    """解析十进制/0x 前缀/纯 hex 字符串。"""
    raw = str(s).strip()
    s = raw.replace('_', '').replace(' ', '')
    neg = s.startswith('-')
    if neg:
        s = s[1:]
    try:
        if not s:
            raise ValueError
        if s.lower().startswith('0x'):
            v = int(s[2:], 16)
        elif all(c in '0123456789' for c in s):
            v = int(s, 10)
        elif all(c in '0123456789abcdefABCDEF' for c in s):
            v = int(s, 16)
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f'无法解析整数: {raw!r}') from None
    return -v if neg else v


def cmd_rsa(a):
    try:
        n = _parse_int(a.n)
        c = _parse_int(a.c)
    except ValueError as ex:
        _err(str(ex))
    if n < 3:
        _err(f'--n 必须是 >= 3 的整数（RSA 模数），收到 {n}')
    if a.e < 3 or a.e % 2 == 0:
        _err(f'--e 必须是 >= 3 的奇数（RSA 公开指数），收到 {a.e}')
    if c < 0:
        _err(f'--c 不能为负，收到 {c}')
    meth, dec = rsa_attack(n, a.e, c)
    if dec is not None:
        b = dec.to_bytes((dec.bit_length() + 7) // 8, 'big')
        print(f'  ★ 方法={meth}  明文 hex={b.hex()}  文本={b!r}')
    else:
        print('  未攻破（试试共模/广播/Coppersmith）')


def cmd_encode(a):
    if a.data_hex is not None:
        data = _hex_arg(a.data_hex, '--data-hex')
    else:
        data = a.data.encode('utf-8')
    got, path = auto_decode(data)
    print(f'\n[decode] 路径={path}')
    print(f'  结果={got!r}')


def main():
    ap = argparse.ArgumentParser(
        description='ctf_ext — CTF 算法族内核（完整说明见 E:\\gpu_crack\\CLI_REFERENCE.md）')
    ap.add_argument('--backend', default='auto', choices=['auto', 'torch', 'numpy'],
                    help='计算后端：auto=torch-cuda→torch-cpu→numpy（默认 auto）')
    sub = ap.add_subparsers(dest='cmd', required=True,
                            metavar='{selftest,tea,speck,rc4,lcg,md5,rsa,encode}')
    sub.add_parser('selftest', help='已知向量自测（失败时退出码=失败项数）')

    p = sub.add_parser('tea', help='TEA / XTEA / XXTEA 加解密')
    p.add_argument('--mode', default='xtea', choices=['tea', 'xtea', 'xxtea'], help='算法，默认 xtea')
    p.add_argument('--op', default='decrypt', choices=['encrypt', 'decrypt'], help='加/解密，默认 decrypt')
    p.add_argument('--key-hex', required=True, help='16 字节密钥 hex')
    p.add_argument('--data-hex', required=True,
                   help='数据裸 hex；tea/xtea 须 8B 倍数，xxtea 解密须 4B 倍数且>=8B')

    p = sub.add_parser('speck', help='Speck32/64、Speck64/128 单块加解密')
    p.add_argument('--key-hex', required=True, help='小端打包的密钥（k0 在最低位），可为整数形式')
    p.add_argument('--data-hex', required=True, help='2*word_bits 位的大端整数（可为整数形式）')
    p.add_argument('--op', default='encrypt', choices=['encrypt', 'decrypt'], help='加/解密，默认 encrypt')
    p.add_argument('--word-bits', type=int, default=32, choices=[16, 32], help='字宽 16/32，默认 32')
    p.add_argument('--key-words', type=int, default=4, help='密钥字数 2/3/4，默认 4')
    p.add_argument('--rounds', type=int, default=None, help='轮数覆盖（自定义组合必填）')

    p = sub.add_parser('rc4', help='RC4 加解密')
    p.add_argument('--key-hex', required=True, help='密钥 hex，1~256 字节')
    p.add_argument('--data-hex', required=True, help='数据裸 hex')

    p = sub.add_parser('lcg', help='LCG 参数恢复 / 状态预测')
    p.add_argument('--mod', default='2**32', help='模数：十进制/0x/裸 hex/2**k，默认 2**32')
    p.add_argument('--outs', required=True, help='>=3 个连续输出，逗号分隔（每项可为 0x/裸 hex）')
    p.add_argument('--predict', type=int, default=3, help='预测个数，默认 3')

    p = sub.add_parser('md5', help='MD5 批量哈希 / 口令爆破')
    p.add_argument('--target', required=True, help='32 字符 MD5 hex')
    p.add_argument('--charset', default=string.ascii_lowercase + string.digits,
                   help='爆破字符集，默认 a-z0-9')
    p.add_argument('--max-len', type=int, default=4, help='口令最大长度（含空串），默认 4')

    p = sub.add_parser('rsa', help='RSA 多攻击自动尝试')
    p.add_argument('--n', required=True, help='模数：十进制/0x/裸 hex（>=3）')
    p.add_argument('--e', type=int, default=65537, help='公开指数，默认 65537（>=3 的奇数）')
    p.add_argument('--c', required=True, help='密文：十进制/0x/裸 hex')

    p = sub.add_parser('encode', help='多层编码 BFS 自动解码')
    enc = p.add_mutually_exclusive_group(required=True)
    enc.add_argument('--data-hex', help='数据以裸 hex 给出')
    enc.add_argument('--data', help='数据以原始文本给出（UTF-8）')

    a = ap.parse_args()
    G.init_backend(a.backend)
    rc = {'selftest': cmd_selftest, 'tea': cmd_tea, 'speck': cmd_speck, 'rc4': cmd_rc4,
          'lcg': cmd_lcg, 'md5': cmd_md5, 'rsa': cmd_rsa, 'encode': cmd_encode}[a.cmd](a)
    # 只有 selftest 会返回 int（失败项数，便于上层集成脚本判断）；其余命令返回 None = 成功
    sys.exit(0 if not isinstance(rc, int) else rc)


if __name__ == '__main__':
    main()
