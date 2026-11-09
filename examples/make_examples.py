#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""examples/make_examples.py — 生成可直接跑通的示例输入（离线、可重复）。

用法：
    python examples/make_examples.py            # 写入 examples/out/
    python examples/make_examples.py --clean    # 先清空再生成

生成物：
    mt19937_keystream.txt   种子 987654 的前 24 字节密钥（mtscan 用）
    rsa_fermat.json         p、q 相近的 RSA 题（rsa --fermat 用）
    ecc_small_k.json        secp256k1 私钥 1337 的公钥（eccscan 用）
    lfsr_bits.txt           x^16+x^14+x^13+x^11+1 的 128 拍输出（bm 用）
    lsb_flag.png            LSB 里藏了 flag 的图（lsb 用）
    wordlist.txt            8 行假口令字典（zipbrute 用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'out')
sys.path.insert(0, ROOT)

import numpy as np                      # noqa: E402


def w(name, data):
    path = os.path.join(OUT, name)
    mode = 'wb' if isinstance(data, bytes) else 'w'
    with open(path, mode, **({} if mode == 'wb' else {'encoding': 'utf-8'})) as fh:
        fh.write(data)
    print(f'  wrote {name:22} {os.path.getsize(path):>7} B')


def gen_mt19937():
    import gpu_crack as G
    G.init_backend('numpy')
    key = G.from_dev(G.mt19937_keys(np.array([987654]), twist=False))[0][:24].tobytes()
    w('mt19937_keystream.txt', key.hex() + '\n')
    return key.hex()


def gen_rsa():
    p, q, e, m = 1000000007, 1000000009, 65537, 0x435446
    n = p * q
    c = pow(m, e, n)
    data = {'p': p, 'q': q, 'n': n, 'e': e, 'm': m, 'c': c,
            'hint': 'p 与 q 相近（差 2），rsa --fermat 秒解；也可以 gcd(n, p*q) 试除'}
    w('rsa_fermat.json', json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def gen_ecc():
    import ctf_ext as E
    k = 1337
    px, py = E.ecc_pub('secp256k1', k)
    pub = b'\x04' + px.to_bytes(32, 'big') + py.to_bytes(32, 'big')
    data = {'curve': 'secp256k1', 'pub_hex': pub.hex(), 'private_key': k,
            'hint': '私钥 1337，eccscan --start 0 --count 5000 命中'}
    w('ecc_small_k.json', json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def gen_lfsr():
    """Galois 型 16 位 LFSR（抽头掩码 0xB400 = x^16+x^14+x^13+x^11+1），输出 128 拍。

    用 Galois 形式可以保证序列的线性复杂度就是 16，bm 必然还原出 L=16。
    """
    reg = 0xACE1
    bits = []
    for _ in range(128):
        out = reg & 1
        bits.append(out)
        reg >>= 1
        if out:
            reg ^= 0xB400
    seq = ''.join(str(b) for b in bits)
    import ctf_ext as E
    taps, L, _pred = E.berlekamp_massey(seq)
    w('lfsr_bits.txt', seq + '\n')
    w('lfsr_bits_meta.txt',
      f'Galois 右移，抽头掩码 0xB400，初值 0xACE1，128 拍\n'
      f'bm 自检：L={L}，taps={taps}\n')


def gen_lsb_png():
    from PIL import Image
    msg = b'flag{example_lsb_ok}'
    payload = msg + b'\x00'
    bits = [(byte >> (7 - i)) & 1 for byte in payload for i in range(8)]
    size = 64
    rnd = np.random.RandomState(2026)
    px = rnd.randint(0, 256, (size, size, 3), dtype=np.uint8)
    flat = px[:, :, 0].reshape(-1)
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | bit
    px[:, :, 0] = flat.reshape(size, size)
    img = Image.fromarray(px, 'RGB')
    path = os.path.join(OUT, 'lsb_flag.png')
    img.save(path)
    print(f'  wrote {"lsb_flag.png":22} {os.path.getsize(path):>7} B')
    w('lsb_flag_meta.txt', f'flag 藏在 R 通道最低位，前 {len(bits)} 个像素，MSB-first\n')


def gen_wordlist():
    words = ['123456', 'password', 'qwerty', 'admin', 'letmein',
             'ctf2026', 'gpu_crack', 'hunter2']
    w('wordlist.txt', '\n'.join(words) + '\n')


def main():
    ap = argparse.ArgumentParser(description='生成 examples/out/ 下的示例输入')
    ap.add_argument('--clean', action='store_true', help='先清空 examples/out/')
    a = ap.parse_args()

    if a.clean and os.path.isdir(OUT):
        for name in os.listdir(OUT):
            os.remove(os.path.join(OUT, name))
    os.makedirs(OUT, exist_ok=True)

    print(f'[examples] 输出目录 {OUT}')
    gen_mt19937()
    gen_rsa()
    gen_ecc()
    gen_lfsr()
    gen_wordlist()
    try:
        gen_lsb_png()
    except ImportError:
        print('  skip  lsb_flag.png（未安装 Pillow）')
    print('[examples] 完成；具体跑法见 examples/README.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
