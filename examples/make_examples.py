#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""examples/make_examples.py — 生成可直接跑通的示例输入（离线、可重复）。

用法：
    python examples/make_examples.py            # 写入 examples/out/
    python examples/make_examples.py --clean    # 先清空再生成
    python examples/make_examples.py --check    # 生成后逐条复跑 README 里的示例命令并汇总 rc

生成物：
    mt19937_keystream.txt   种子 987654 的前 24 字节密钥（mtscan 用）
    rsa_fermat.json         p、q 相近的 RSA 题（rsa 用）
    ecc_small_k.json        secp256k1 私钥 1337 的公钥（eccscan 用）
    lfsr_bits.txt           x^16+x^14+x^13+x^11+1 的 128 拍输出（bm 用）
    lsb_flag.png            LSB 里藏了 flag 的图（lsb 用）
    wordlist.txt            8 行假口令字典（zipbrute 用）
    locked.zip              ZipCrypto 加密的 zip（口令 hunter2，zipbrute 用）

--check 模式用 sys.executable 子进程逐条执行示例命令（默认 --backend numpy，
无 GPU 的 CI 环境也能跑），任一失败则退出码非 0。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'out')
CTF = os.path.join(ROOT, 'skill', 'scripts', 'ctf.py')
sys.path.insert(0, ROOT)

import numpy as np                      # noqa: E402

ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')


def w(name, data):
    path = os.path.join(OUT, name)
    mode = 'wb' if isinstance(data, bytes) else 'w'
    with open(path, mode, **({} if mode == 'wb' else {'encoding': 'utf-8'})) as fh:
        fh.write(data)
    print(f'  wrote {name:22} {os.path.getsize(path):>7} B')


# ---------------------------------------------------------------- 生成器

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
            'hint': 'p 与 q 相近（差 2），Fermat 分解秒解'}
    w('rsa_fermat.json', json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    return data


def gen_ecc():
    import ctf_ext as E
    k = 1337
    px, py = E.ecc_pub('secp256k1', k)
    pub = b'\x04' + px.to_bytes(32, 'big') + py.to_bytes(32, 'big')
    data = {'curve': 'secp256k1', 'pub_hex': pub.hex(), 'private_key': k,
            'hint': '私钥 1337，eccscan --start 0 --count 5000 命中'}
    w('ecc_small_k.json', json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    return data


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
    return seq


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
    return words


# ---------------------------------------------------------------- ZipCrypto
# 标准 zipfile 不能写加密 zip；这里按 PKWARE APPNOTE 的 ZipCrypto（传统加密）手工打包，
# 让 examples 的 zipbrute 示例真正可跑，而不是要求用户自备加密 zip。

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (0xEDB88320 if _c & 1 else 0)
    _CRC_TABLE.append(_c)


def _crc32_byte(crc, b):
    return (crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF]


class _ZipCrypto:
    """ZipCrypto 加密器：密钥流由口令初始化，头部 12 字节 + 数据连续加密。"""

    def __init__(self, password: bytes):
        self.k0, self.k1, self.k2 = 0x12345678, 0x23456789, 0x34567890
        for b in password:
            self._update(b)

    def _update(self, b):
        self.k0 = _crc32_byte(self.k0, b)
        self.k1 = (self.k1 + (self.k0 & 0xFF)) & 0xFFFFFFFF
        self.k1 = (self.k1 * 134775813 + 1) & 0xFFFFFFFF
        self.k2 = _crc32_byte(self.k2, (self.k1 >> 24) & 0xFF)

    def _stream_byte(self):
        temp = (self.k2 | 2) & 0xFFFF
        return ((temp * (temp ^ 1)) >> 8) & 0xFF

    def encrypt(self, data: bytes) -> bytes:
        out = bytearray()
        for b in data:
            out.append(b ^ self._stream_byte())
            self._update(b)
        return bytes(out)


def _make_zipcrypto_zip(path: str, entry_name: str, payload: bytes, password: bytes) -> None:
    """写一个单条目、store 方式、ZipCrypto 加密的 zip（确定性：固定 11 字节头随机数）。"""
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = bytes(range(1, 12)) + bytes([(crc >> 24) & 0xFF])
    crypt = _ZipCrypto(password)
    enc = crypt.encrypt(header + payload)
    name = entry_name.encode('utf-8')
    dos_time, dos_date = 0, ((2026 - 1980) << 9) | (1 << 5) | 1
    flags = 0x0001                              # bit0 = encrypted
    local = struct.pack('<IHHHHHIIIHH', 0x04034B50, 20, flags, 0, dos_time, dos_date,
                        crc, len(enc), len(payload), len(name), 0) + name + enc
    central = struct.pack('<IHHHHHHIIIHHHHHII', 0x02014B50, 20, 20, flags, 0, dos_time,
                          dos_date, crc, len(enc), len(payload), len(name), 0, 0, 0, 0,
                          0, 0) + name
    eocd = struct.pack('<IHHHHIIH', 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)
    with open(path, 'wb') as fh:
        fh.write(local + central + eocd)


def gen_locked_zip():
    payload = b'flag{zip_example_ok}\n'
    _make_zipcrypto_zip(os.path.join(OUT, 'locked.zip'), 'flag.txt', payload, b'hunter2')
    print(f'  wrote {"locked.zip":22} {os.path.getsize(os.path.join(OUT, "locked.zip")):>7} B')
    w('locked_zip_meta.txt', 'ZipCrypto 加密，条目 flag.txt，口令 hunter2（在 wordlist.txt 第 8 行）\n')


# ---------------------------------------------------------------- --check

def _load_json(name):
    with open(os.path.join(OUT, name), encoding='utf-8') as fh:
        return json.load(fh)


def _read_text(name):
    with open(os.path.join(OUT, name), encoding='utf-8') as fh:
        return fh.read().strip()


def build_checks():
    """README 里逐条示例命令（单一真源；README 与 --check 必须一致）。"""
    mt = _read_text('mt19937_keystream.txt')
    rsa = _load_json('rsa_fermat.json')
    ecc = _load_json('ecc_small_k.json')
    lfsr = _read_text('lfsr_bits.txt')
    checks = [
        ('mt19937 种子爆破',
         ['--backend', 'numpy', 'mtscan', '--key-hex', mt, '--seed-start', '900000',
          '--seed-count', '200000'],
         ['命中 seed=987654']),
        ('RSA Fermat 分解',
         ['--backend', 'numpy', 'rsa', '--n', str(rsa['n']), '--e', str(rsa['e']),
          '--c', str(rsa['c'])],
         ['明文 hex=435446']),
        ('ECC 小私钥',
         ['--backend', 'numpy', 'eccscan', '--curve', 'secp256k1', '--pub-hex', ecc['pub_hex'],
          '--start', '0', '--count', '5000'],
         ['私钥 k = 1337']),
        ('LFSR 反推（bm）',
         ['--backend', 'numpy', 'bm', '--bits', lfsr],
         ['L=16']),
        ('LSB 隐写提取',
         ['lsb', '--img', os.path.join('examples', 'out', 'lsb_flag.png')],
         ['flag{example_lsb_ok}']),
        ('ZIP 口令爆破',
         ['--backend', 'numpy', 'zipbrute', '--zip', os.path.join('examples', 'out', 'locked.zip'),
          '--wordlist', os.path.join('examples', 'out', 'wordlist.txt'), '--jobs', '1'],
         ["口令 = 'hunter2'"]),
    ]
    return checks


def run_checks() -> int:
    """逐条用子进程执行示例命令，验证 rc==0 且输出含预期标记；汇总失败数。"""
    checks = build_checks()
    print(f'[examples] --check：逐条复跑 {len(checks)} 条示例命令（cwd={ROOT}）')
    failed = []
    for name, args, markers in checks:
        cmd = [sys.executable, '-X', 'utf8', CTF] + args
        shown = ' '.join(args)
        if len(shown) > 120:
            shown = shown[:117] + '...'
        try:
            r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=300)
        except (subprocess.TimeoutExpired, OSError) as exc:
            failed.append(name)
            print(f'  FAIL {name}: {exc}')
            continue
        out = (r.stdout or '') + (r.stderr or '')
        missing = [m for m in markers if m not in out]
        if r.returncode != 0 or missing:
            failed.append(name)
            print(f'  FAIL {name}: rc={r.returncode} missing={missing}')
            print('    cmd: python skill/scripts/ctf.py ' + shown)
            tail = out.strip().splitlines()[-6:]
            for ln in tail:
                print('    | ' + ln[:200])
        else:
            print(f'  PASS {name}')
    print(f'[examples] --check: {len(checks) - len(failed)}/{len(checks)} PASS')
    if failed:
        print('[examples] --check: FAIL -> ' + ', '.join(failed))
        return 1
    return 0


def verify_idempotent() -> int:
    """重复生成并比较文件哈希，确保示例生成是确定性的（CI/复跑不会无故 diff）。"""
    names = ['mt19937_keystream.txt', 'rsa_fermat.json', 'ecc_small_k.json', 'lfsr_bits.txt',
             'lfsr_bits_meta.txt', 'lsb_flag.png', 'lsb_flag_meta.txt', 'wordlist.txt',
             'locked.zip', 'locked_zip_meta.txt']

    def digest():
        out = {}
        for n in names:
            p = os.path.join(OUT, n)
            if os.path.exists(p):
                with open(p, 'rb') as fh:
                    out[n] = hashlib.sha256(fh.read()).hexdigest()
        return out

    first = digest()
    gen_all(clean=False, verbose=False)
    second = digest()
    bad = [n for n in names if first.get(n) != second.get(n)]
    print(f'[examples] 幂等性：{"PASS" if not bad else "FAIL"}（重复生成哈希一致）')
    if bad:
        print('  不一致文件:', bad)
        return 1
    return 0


# ---------------------------------------------------------------- main

def gen_all(clean: bool = False, verbose: bool = True) -> None:
    if clean and os.path.isdir(OUT):
        for name in os.listdir(OUT):
            os.remove(os.path.join(OUT, name))
    os.makedirs(OUT, exist_ok=True)

    if verbose:
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
    gen_locked_zip()
    if verbose:
        print('[examples] 完成；具体跑法见 examples/README.md')


def main():
    ap = argparse.ArgumentParser(description='生成 examples/out/ 下的示例输入')
    ap.add_argument('--clean', action='store_true', help='先清空 examples/out/')
    ap.add_argument('--check', action='store_true',
                    help='生成后逐条复跑 README 示例命令并汇总 rc（无 GPU 用 numpy 后端）')
    a = ap.parse_args()

    gen_all(clean=a.clean)
    rc = 0
    if a.check:
        rc |= verify_idempotent()
        rc |= run_checks()
    print(f'[examples] {"ALL PASS" if rc == 0 else "FAIL"}')
    return 0 if rc == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
