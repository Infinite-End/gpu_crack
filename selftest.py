#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_crack.py 自测：正确性 + 性能。用 GPU 环境运行。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_crack as G

name, dev = G.init_backend('torch')
print(f'[backend] {name} {dev}\n')
T = G.xp()

# ---------- 1) AES-192 NIST 向量 ----------
key = bytes.fromhex('000102030405060708090a0b0c0d0e0f1011121314151617')
pt  = bytes.fromhex('00112233445566778899aabbccddeeff')
exp = bytes.fromhex('dda97ca4864cdfe06eaf70a0ec0d7191')
blk = T.tensor(list(pt), dtype=T.uint8, device=G.DEV).unsqueeze(0)
kt  = T.tensor(list(key), dtype=T.uint8, device=G.DEV)
out = bytes(G.from_dev(G.aes192_ecb(blk, kt))[0])
print(f'[AES-192 NIST] {"PASS" if out == exp else "FAIL"}  got={out.hex()}')

# ---------- 2) mt19937 vs 纯 Python 参考 ----------
class RefMT:
    def __init__(self, seed):
        self.mt=[0]*624; self.mt[0]=seed & 0xFFFFFFFF
        for i in range(1,624):
            self.mt[i]=(1812433253*(self.mt[i-1]^(self.mt[i-1]>>30))+i)&0xFFFFFFFF
    def keys(self,n=24):
        o=[]
        for k in range(n):
            y=self.mt[k]; y^=y>>11; y^=(y<<7)&0x9D2C5680; y^=(y<<15)&0xEFC60000; y^=y>>18
            o.append(y&0xFF)
        return bytes(o)

seeds = list(range(16)) + [123456, 0xDEADBEEF, 0xFFFFFFFF]
gpu = G.from_dev(G.mt19937_keys(T.tensor(seeds, dtype=T.int64, device=G.DEV), twist=False))
ok = all(bytes(gpu[i]) == RefMT(s).keys() for i, s in enumerate(seeds))
print(f'[mt19937 vs PyRef] {"PASS" if ok else "FAIL"}  ({len(seeds)} 个种子 × 24B)')

# ---------- 3) 种植-找回测试：mtscan ----------
PLANT = 987654
plant_key = RefMT(PLANT).keys(24).hex()
print(f'\n[测试] mtscan 目标 key={plant_key[:16]}...  种子 = {PLANT}')
class A: pass
a = A(); a.key_hex = plant_key; a.seed_start = PLANT - 1000; a.seed_count = 2000; a.chunk = 1024; a.twist = 0
t0 = time.time(); G.cmd_mtscan(a); print(f'  用时 {time.time()-t0:.2f}s')

# ---------- 4) 种植-找回测试：sboxscan ----------
A_, B_ = 0x42, 0x99
ai = T.tensor([[A_]], dtype=T.uint8, device=G.DEV)
bv = T.tensor([[int(G.INV_SBOX[B_])]], dtype=T.uint8, device=G.DEV)
plain = b'gylike{12345678}'
pb = T.tensor(list(plain), dtype=T.uint8, device=G.DEV).unsqueeze(0)
plant_ct = bytes(G.from_dev(G.aes192_ecb(pb, kt, None, ai, bv))[0])
print(f'\n[测试] sboxscan 植入 (a,b)=({A_:#x},{B_:#x})  ct={plant_ct.hex()}')
a2 = A(); a2.plain_hex = plain.hex(); a2.cipher_hex = plant_ct.hex(); a2.key_hex = key.hex(); a2.chunk = 1 << 14
t0 = time.time(); G.cmd_sboxscan(a2); print(f'  用时 {time.time()-t0:.2f}s')

# ---------- 5) 性能 ----------
print('\n[性能]')
for n in (1 << 16, 1 << 18):
    s = T.arange(n, dtype=T.int64, device=G.DEV)
    t0 = time.time(); G.mt19937_keys(s, twist=False); dt = time.time() - t0
    print(f'  mt19937 : {n/dt/1e6:8.2f} M keys/s  ({n} 个, {dt*1e3:.0f} ms)')
for n in (1 << 14, 1 << 16):
    b = T.zeros((n, 16), dtype=T.uint8, device=G.DEV)
    a_idx = T.full((n, 1), 0x42, dtype=T.uint8, device=G.DEV)
    b_val = T.full((n, 1), 0x99, dtype=T.uint8, device=G.DEV)
    t0 = time.time(); G.aes192_ecb(b, kt, None, a_idx, b_val); dt = time.time() - t0
    print(f'  AES-192 : {n/dt/1e6:8.2f} M blocks/s ({n} 块, {dt*1e3:.0f} ms)')
