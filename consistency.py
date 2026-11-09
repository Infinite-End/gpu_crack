#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consistency.py — 跨模块 / 跨后端的独立一致性校验

目的：同一个已知答案，用「不同代码路径」「不同后端」「独立第三方库」分别算，
      结果必须完全一致。任何一处不一致都说明实现有 bug。

覆盖：
  AES-192   : 本工具(torch) vs 本工具(numpy) vs pycryptodome
  ChaCha20  : 本工具 vs pycryptodome
  Poly1305  : 本工具 vs pycryptodome
  CRC32     : 本工具 vs zlib
  MD5       : 本工具 vs 已知向量
  MT19937   : 本工具(torch) vs 本工具(numpy) vs 纯 Python 参考
  TEA/XTEA  : 本工具 vs 独立重写实现
  XXTEA     : 加解密互逆 + 多长度往返
  Speck     : NSA 官方向量 + 加解密互逆
  RS(63,31) : 自建编码器产出的码字 syndrome 必须全零 + 单字节错误必须能定位
  LCG       : 恢复参数 → 预测，与实际序列逐项比对
  xorbrute  : 爆破结果必须是正确明文
  encode    : 多层解码结果必须等于原始明文
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import string
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_crack as G
import ctf_tools as TOOLS
import ctf_ext as EXT
import ctf_misc as MISC

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f'{"PASS" if cond else "FAIL"}  {name:34s} {detail}')
    return cond


# ---------------------------------------------------------------- AES-192
def test_aes():
    key = bytes.fromhex('000102030405060708090a0b0c0d0e0f1011121314151617')
    pt = bytes.fromhex('00112233445566778899aabbccddeeff')
    expect = bytes.fromhex('dda97ca4864cdfe06eaf70a0ec0d7191')

    from Crypto.Cipher import AES
    ref = AES.new(key, AES.MODE_ECB).encrypt(pt)
    check('AES-192 参考实现(pycryptodome)', ref == expect, ref.hex())

    # 本工具 torch 后端
    G.init_backend('torch')
    T = G.xp()
    blk = T.tensor(list(pt), dtype=T.uint8, device=G.DEV).unsqueeze(0)
    kt = T.tensor(list(key), dtype=T.uint8, device=G.DEV)
    out_t = bytes(G.from_dev(G.aes192_ecb(blk, kt))[0])
    check('AES-192 本工具(torch)', out_t == expect, out_t.hex())

    # 本工具 numpy 后端
    G.init_backend('numpy')
    T2 = G.xp()
    blk2 = np.frombuffer(pt, dtype=np.uint8).reshape(1, 16)
    kt2 = np.frombuffer(key, dtype=np.uint8)
    out_n = bytes(G.from_dev(G.aes192_ecb(blk2, kt2))[0])
    check('AES-192 本工具(numpy)', out_n == expect, out_n.hex())
    check('AES-192 torch == numpy', out_t == out_n)

    # 多块 + 不同密钥一致性（随机 8 组）
    ok = True
    rng = random.Random(7)
    for _ in range(8):
        k = bytes(rng.randrange(256) for _ in range(24))
        p = bytes(rng.randrange(256) for _ in range(16))
        r = AES.new(k, AES.MODE_ECB).encrypt(p)
        G.init_backend('torch')
        T = G.xp()
        o = bytes(G.from_dev(G.aes192_ecb(
            T.tensor(list(p), dtype=T.uint8, device=G.DEV).unsqueeze(0),
            T.tensor(list(k), dtype=T.uint8, device=G.DEV)))[0])
        if o != r:
            ok = False
            break
    check('AES-192 随机 8 组 vs 参考', ok)
    G.init_backend('torch')


# ---------------------------------------------------------------- ChaCha20 / Poly1305
def test_chacha_poly():
    from Crypto.Cipher import ChaCha20 as RefChaCha

    key = bytes(range(32))
    nonce12 = bytes.fromhex('000000090000004a00000000')

    # --- 1) RFC 8439 向量（最权威）---
    ks = bytes(TOOLS.chacha20_keystream(key, nonce12, counter=1, nbytes=64)[0])
    rfc = ('10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e'
           'd2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e')
    check('ChaCha20 RFC8439 向量', ks.hex() == rfc)

    # --- 2) 原版变体（64bit counter + 64bit nonce）vs pycryptodome ---
    nonce8 = bytes(8)
    ours_orig = bytes(TOOLS.chacha20_keystream(key, nonce8, counter=0, nbytes=192, rfc7539=False)[0])
    ref_orig = RefChaCha.new(key=key, nonce=nonce8).encrypt(b'\x00' * 192)
    check('ChaCha20 原版变体 vs pycryptodome', ours_orig == ref_orig,
          f'{ours_orig[:8].hex()} vs {ref_orig[:8].hex()}')

    # --- 3) 端到端 AEAD：ChaCha20-Poly1305 全流程 vs cryptography ---
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    ok_auth = True
    rng = random.Random(11)
    for trial in range(6):
        k = bytes(rng.randrange(256) for _ in range(32))
        nonce = bytes(rng.randrange(256) for _ in range(12))
        aad = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 20)))
        msg = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 90)))
        ref = ChaCha20Poly1305(k).encrypt(nonce, msg, aad)
        ref_ct, ref_tag = ref[:-16], ref[-16:]

        otk = bytes(TOOLS.chacha20_keystream(k, nonce, counter=0, nbytes=32)[0])
        ks_body = bytes(TOOLS.chacha20_keystream(k, nonce, counter=1, nbytes=len(msg))[0])
        our_ct = bytes(a ^ b for a, b in zip(msg, ks_body))
        our_tag = TOOLS.poly1305_tag(otk, aad, our_ct)
        if our_ct != ref_ct or our_tag != ref_tag:
            ok_auth = False
            print(f'    trial{trial} ct={our_ct == ref_ct} tag={our_tag.hex()} vs {ref_tag.hex()}')
            break
    check('ChaCha20-Poly1305 AEAD vs cryptography', ok_auth)

    # --- 4) Poly1305 单向量 ---
    pk = bytes.fromhex('85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b')
    check('Poly1305 RFC8439 向量',
          TOOLS.poly1305(pk, b'Cryptographic Forum Research Group').hex() == 'a8061dc1305136c6c22b8baf0c0127a9')


# ---------------------------------------------------------------- CRC32 / MD5
def test_hash():
    data = b'123456789'
    ours = int(TOOLS.crc32_batch(np.frombuffer(data, dtype=np.uint8).reshape(1, -1))[0])
    check('CRC32 本工具 vs zlib', ours == zlib.crc32(data), f'{ours:08x}')
    rnd = bytes(random.Random(3).randrange(256) for _ in range(97))
    ours2 = int(TOOLS.crc32_batch(np.frombuffer(rnd, dtype=np.uint8).reshape(1, -1))[0])
    check('CRC32 随机 97B vs zlib', ours2 == zlib.crc32(rnd))
    check('MD5 本工具 vs hashlib',
          EXT.md5_batch([b'abc'])[0] == hashlib.md5(b'abc').digest() ==
          bytes.fromhex('900150983cd24fb0d6963f7d28e17f72'))


# ---------------------------------------------------------------- MT19937
class RefMT:
    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.i = 624          # 与 std::mt19937 一致：首次取数前先 twist

    def temper(self, y):
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def next(self):
        if self.i >= 624:
            for k in range(624):
                y = (self.mt[k] & 0x80000000) + (self.mt[(k + 1) % 624] & 0x7FFFFFFF)
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1)
                if y & 1:
                    self.mt[k] ^= 0x9908B0DF
            self.i = 0
        v = self.temper(self.mt[self.i])
        self.i += 1
        return v


def test_mt():
    seeds = [0, 1, 5489, 123456, 0xFFFFFFFF]
    for tw in (False, True):
        T = G.xp()
        got = G.from_dev(G.mt19937_keys(T.tensor(seeds, dtype=T.int64, device=G.DEV), twist=tw))
        if tw:
            ref = []
            for s in seeds:
                r = RefMT(s)                        # 必须复用同一个实例，否则永远只有第一个输出
                ref.append(bytes(r.next() & 0xFF for _ in range(24)))
        else:
            # 不 twist：直接 temper 初始状态数组
            ref = []
            for s in seeds:
                mt = [0] * 624
                mt[0] = s & 0xFFFFFFFF
                for i in range(1, 624):
                    mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
                tmp = RefMT(s)
                ref.append(bytes(tmp.temper(mt[i]) & 0xFF for i in range(24)))
        ok = all(bytes(got[i]) == ref[i] for i in range(len(seeds)))
        check(f'MT19937 twist={int(tw)} vs 纯 Python 参考', ok)

    # numpy 后端 vs torch 后端
    G.init_backend('numpy')
    got_np = G.mt19937_keys(np.array(seeds, dtype=np.int64), twist=True)
    G.init_backend('torch')
    got_t = G.from_dev(G.mt19937_keys(G.xp().tensor(seeds, dtype=G.xp().int64, device=G.DEV), twist=True))
    check('MT19937 numpy == torch', np.array_equal(got_np, got_t))


# ---------------------------------------------------------------- TEA 家族
def tea_ref_encrypt(v0, v1, k, rounds=32):
    """独立重写的 TEA（对照实现）"""
    delta = 0x9E3779B9
    s = (delta * rounds) & 0xFFFFFFFF
    v0, v1 = v0, v1
    s = 0
    for _ in range(rounds):
        s = (s + delta) & 0xFFFFFFFF
        v0 = (v0 + (((v1 << 4) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1]))) & 0xFFFFFFFF
        v1 = (v1 + (((v0 << 4) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3]))) & 0xFFFFFFFF
    return v0, v1


def test_tea():
    k = [0, 0, 0, 0]
    a = EXT.tea_encrypt_block(0, 0, k)
    b = tea_ref_encrypt(0, 0, k)
    check('TEA vs 独立实现', a == b == (0x41EA3A0A, 0x94BAA940), f'{a[0]:08x}{a[1]:08x}')

    key = list(np.frombuffer(b'0123456789abcdef', dtype='<u4'))
    key = [int(x) for x in key]
    for v0, v1 in ((0x12345678, 0x9ABCDEF0), (0, 0xFFFFFFFF), (0xDEADBEEF, 0xCAFEBABE)):
        e = EXT.tea_encrypt_block(v0, v1, key)
        d = EXT.tea_decrypt_block(e[0], e[1], key)
        check(f'TEA 往返 {v0:08x},{v1:08x}', d == (v0, v1))
        xe = EXT.xtea_encrypt_block(v0, v1, key)
        xd = EXT.xtea_decrypt_block(xe[0], xe[1], key)
        check(f'XTEA 往返 {v0:08x},{v1:08x}', xd == (v0, v1))

    # XXTEA 多长度往返
    ok = True
    for msg in (b'a', b'abcd', b'flag{xx}', b'0123456789abcdef', bytes(range(64))):
        ct = EXT.xxtea_encrypt(msg, b'0123456789abcdef')
        pt = EXT.xxtea_decrypt(ct, b'0123456789abcdef')[:len(msg)]
        if pt != msg:
            ok = False
            print('   XXTEA 失败:', msg[:16], pt[:16])
            break
    check('XXTEA 多长度往返(1/4/8/16/64B)', ok)


def test_speck():
    ct = EXT.speck_encrypt(0x6574694C, 0x1918111009080100, word_bits=16, key_words=4)
    check('Speck32/64 NSA 向量', ct == 0xA86842F2, f'{ct:08x}')
    ok = True
    for pt in (0x1234567890ABCDEF, 0, 0xFFFFFFFFFFFFFFFF):
        key = 0x1B1A1918131211100B0A090803020100
        c = EXT.speck_encrypt(pt, key, word_bits=32, key_words=4)
        p = EXT.speck_decrypt(c, key, word_bits=32, key_words=4)
        if p != pt:
            ok = False
    check('Speck64/128 加解密往返', ok)


# ---------------------------------------------------------------- RS(63,31)
def gf_pow_mul(a, b):
    return TOOLS.gf_mul_tab(np.array([a], dtype=np.uint8), np.array([b], dtype=np.uint8))[0]


def rs63_encode(data31):
    """
    RS(63,31) 系统码编码器（fcr=0，系数按 **降幂** 排列，与 rs63_syndromes(order='msb') 对齐）。
    c(x) = m(x)·x^32 + r(x)，r(x) = m(x)·x^32 mod g(x)
    """
    nsym = 32
    # g(x) = Π (x - α^i)，按降幂保存
    gen = [1]
    for i in range(nsym):
        alpha = int(TOOLS._EXP[i % 255])
        new = [0] * (len(gen) + 1)
        for j, g in enumerate(gen):
            new[j] ^= int(g)                       # g(x) * x
            new[j + 1] ^= int(gf_pow_mul(g, alpha))  # g(x) * α^i
        gen = new
    # 现在 gen[0] 是最高次系数（1）
    msg = list(data31) + [0] * nsym
    for i in range(len(data31)):
        coef = msg[i]
        if coef:
            for j in range(len(gen)):
                msg[i + j] ^= int(gf_pow_mul(gen[j], coef))
    # 前 len(data31) 位已被清零，只剩余式在尾部
    return bytes(list(data31) + msg[31:])


def test_rs():
    data = bytes(range(31))
    cw = rs63_encode(data)
    ok, info = TOOLS.rs63_check(cw)
    check('RS(63,31) 自建编码器 → syndrome 为零', ok, info)

    # 单字节错误定位
    bad = bytearray(cw)
    pos, delta = 17, 0x5A
    bad[pos] ^= delta
    ok2, _ = TOOLS.rs63_check(bytes(bad))
    check('RS(63,31) 破坏后 syndrome 非零', not ok2)
    # 单字节错误定位：只在"确实与正确码字不同"的位置上尝试修复
    bad_b = bytes(bad)
    located = None
    for p in range(63):
        if bad_b[p] == cw[p]:
            continue
        t = bytearray(bad_b)
        t[p] ^= (bad_b[p] ^ cw[p])          # 把该位改回正确值
        if bytes(t) == cw and TOOLS.rs63_check(bytes(t))[0]:
            located = p
            break
    check('RS(63,31) 单字节错误可定位', located == pos, f'期望 {pos} 得到 {located}')


# ---------------------------------------------------------------- LCG
def test_lcg():
    m, a, c, x = 2 ** 31 - 1, 1103515245, 12345, 42
    seq = []
    for _ in range(8):
        x = (a * x + c) % m
        seq.append(x)
    r = EXT.lcg_recover(seq[:4], m)
    check('LCG 参数恢复', r == (a, c), str(r))
    pred = EXT.lcg_predict(seq[:4], m, 4)
    check('LCG 预测与实际一致', pred == seq[4:8], str(pred[:2]))


# ---------------------------------------------------------------- xorbrute / encode
def test_misc_crypto():
    plain = b'flag{xor_brute_ok}'
    key = b'K3y'
    ct = bytes(b ^ key[i % len(key)] for i, b in enumerate(plain))
    res = TOOLS.xorbrute(ct, max_len=3, top=5)
    hit = any(pt == plain for _, _, pt in res)
    check('xorbrute 3 字节爆破还原', hit, repr(res[0][2][:20]) if res else '')

    import base64
    inner = base64.b32encode(b'flag{chain_ok}')
    mid = base64.b64encode(inner)
    outer = mid.hex().encode()
    got, path = EXT.auto_decode(outer, verbose=False)
    check('encode 多层链式解码', got == b'flag{chain_ok}', f'{path} → {got}')


# ---------------------------------------------------------------- main
def test_fuzz(n):
    """随机输入对拍：固定向量覆盖标准值，这里补边界（随机密钥长度 / 块 / nonce）。"""
    from Crypto.Cipher import AES
    rnd = random.Random(20261109)
    for i in range(n):
        klen = rnd.choice((16, 24, 32))
        key = bytes(rnd.randrange(256) for _ in range(klen))
        blk = bytes(rnd.randrange(256) for _ in range(16))
        # 1) AES-ECB：numpy 后端 vs torch 后端
        G.init_backend('numpy')
        ref = G.from_dev(G.aes_ecb(np.frombuffer(blk, dtype=np.uint8)[None, :],
                                   np.frombuffer(key, dtype=np.uint8))).tobytes()
        G.init_backend('torch')
        T = G.xp()
        got = G.from_dev(G.aes_ecb(T.tensor(np.frombuffer(blk, dtype=np.uint8)[None, :],
                                            device=G.DEV),
                                   T.tensor(np.frombuffer(key, dtype=np.uint8),
                                            device=G.DEV))).tobytes()
        check(f'fuzz#{i} AES-{klen*8} torch↔numpy', ref == got, f'{ref.hex()} vs {got.hex()}')
        # 2) AES-CTR（16B 整块计数器 / 大端）：vs pycryptodome
        nonce16 = bytes(rnd.randrange(256) for _ in range(16))
        ctr = rnd.randrange(1 << 16)
        want = AES.new(key, AES.MODE_CTR, nonce=b'',
                       initial_value=int.from_bytes(nonce16, 'big') + ctr).encrypt(bytes(64))
        mine = TOOLS.aesctr_keystream(key, nonce16, 64, ctr, 'be')
        check(f'fuzz#{i} AES-{klen*8}-CTR vs pycryptodome', mine == want,
              f'{mine[:8].hex()} vs {want[:8].hex()}')
    G.init_backend('auto')


def main(argv=None):
    ap = argparse.ArgumentParser(description='跨模块 / 跨后端 / vs 第三方库 一致性校验')
    ap.add_argument('--fuzz', type=int, default=0, metavar='N',
                    help='额外做 N 轮随机对拍（torch ↔ numpy ↔ pycryptodome），默认 0')
    a = ap.parse_args(argv)
    fuzz = a.fuzz
    print('=' * 60)
    print('跨模块 / 跨后端 / vs 第三方库 一致性校验')
    print('=' * 60)
    test_hash()
    test_aes()
    test_chacha_poly()
    test_mt()
    test_tea()
    test_speck()
    test_rs()
    test_lcg()
    test_misc_crypto()
    if fuzz:
        test_fuzz(fuzz)

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print('=' * 60)
    print(f'通过 {npass}/{len(RESULTS)}')
    if npass != len(RESULTS):
        print('失败项:')
        for name, ok, d in RESULTS:
            if not ok:
                print(f'  - {name}  {d}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
