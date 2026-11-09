#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_crack.py — 通用 GPU 批量爆破 / 校验框架（CTF toolkit 主框架）

设计原则
--------
1. 一切按 **批次（batch）** 思考：把"要试 N 个候选"变成"同时算 N 个"。
2. 先算 **廉价判据**（XOR 后的 popcount / 前缀匹配），再算昂贵的（完整 tag/AES）。
3. 分块（chunk）执行，显存不够就调小 --chunk，避免 OOM。
4. 早停：找到即返回，不用等全量跑完。
5. 后端可切换：torch(CUDA) → torch(CPU) → numpy。

子命令
------
  bench      基准测试（摸清本机吞吐，正式搜索前先跑）
  mtscan     暴力 mt19937 种子 → 24B 密钥，与目标密钥流前缀比对
  sboxscan   AES-128/192/256 任意 S-box 候选批量加密，匹配指定明文/密文对
  ksscore    候选密钥流 × 多样本 的比特一致度打分（统计淘汰）
  lfsrscan   多条 LFSR（抽头可不同）并行跑 N 拍，匹配目标比特流

全局参数（写在子命令之前或之后均可）
------------------------------------
  --backend {auto,torch,numpy}   默认 auto（torch-cuda → torch-cpu → numpy 回退）
  --chunk N                      每批候选数，默认 65536；显存不足时减半
  --json / --quiet / --no-progress / --checkpoint FILE / --reset-checkpoint

退出码
------
  0  成功（搜索未命中同样是 0，看输出里的 "未命中"/"★ 命中"）
  2  参数或输入格式错误（stderr 一行 "[错误] ..."，不抛 traceback）

用法示例
--------
  python gpu_crack.py bench
  python gpu_crack.py mtscan  --seed-start 0 --seed-count 100000000 --key-hex ac2f75c0...
  python gpu_crack.py sboxscan --plain-hex 616263... --cipher-hex 0011... --key-hex 0011...
  python gpu_crack.py ksscore --target-hex 3836c93c...
  python gpu_crack.py lfsrscan --taps 25:10,11,24 --bits 1011...

完整参数/输出/退出码见 CLI_REFERENCE.md。
依赖：torch（CUDA 版最好）；无 torch 时自动退回 numpy（慢但能跑）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import ctf_common as C

# ---------------------------------------------------------------- 后端选择
BACKEND = None
torch = None
DEV = None


def init_backend(prefer: str = "auto"):
    """初始化计算后端，返回 (backend_name, device_str)。"""
    global BACKEND, torch, DEV
    if prefer in ("auto", "torch"):
        try:
            import torch as _t
            torch = _t
            if _t.cuda.is_available():
                DEV = _t.device("cuda")
                BACKEND = "torch-cuda"
            else:
                DEV = _t.device("cpu")
                BACKEND = "torch-cpu"
        except Exception:
            BACKEND = None
    if prefer == "numpy" or BACKEND is None:
        BACKEND = "numpy"
        DEV = None
    return BACKEND, str(DEV)


def xp():
    """返回当前张量库（torch 或 numpy）。"""
    if BACKEND.startswith("torch"):
        return torch
    return np


def to_dev(a):
    if BACKEND.startswith("torch"):
        return a.to(DEV, non_blocking=True)
    return a


def from_dev(a):
    if BACKEND.startswith("torch"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def hit_indices(mask):
    """bool 命中掩码 -> int 索引数组。

    注意 torch 的 tensor.nonzero() 返回 [K, 1]（与 numpy 的 tuple-of-arrays 不同），
    旧代码 mask.nonzero()[0] 在 torch 后端只会取到"第一个命中"，导致每块漏报其余命中。
    """
    return np.nonzero(np.asarray(from_dev(mask)))[0]


def M32():
    """uint32 掩码张量（torch 用 int64 模拟 uint32 回绕）。"""
    return 0xFFFFFFFF


def parse_hex(text, opt):
    """解析命令行裸 hex 参数；非法时返回 None（调用方转成退出码 2，不抛 traceback）。"""
    if not isinstance(text, str):
        return None
    try:
        return bytes.fromhex(text)
    except ValueError:
        return None


# ---------------------------------------------------------------- 通用工具
_PCTAB_CACHE = {}


def _pctab(T):
    """256 项 popcount 查找表（缓存在对应设备上）。"""
    key = str(getattr(T, "__name__", T)) + str(DEV)
    if key not in _PCTAB_CACHE:
        v = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
        _PCTAB_CACHE[key] = torch.from_numpy(v).to(DEV) if BACKEND.startswith("torch") else v
    return _PCTAB_CACHE[key]


def popcount_matrix(x):
    """
    按位统计 1 的个数：x 为整型张量（任意 shape）。
    GPU 走 256 项查表（4 字节 / 8 字节各查一次），比逐位移位快一个数量级且无同步。
    """
    if BACKEND.startswith("torch"):
        x = x.to(torch.int64)
        tab = _pctab(torch)
        c = tab[x & 0xFF]
        c += tab[(x >> 8) & 0xFF]
        c += tab[(x >> 16) & 0xFF]
        c += tab[(x >> 24) & 0xFF]
        c += tab[(x >> 32) & 0xFF]
        c += tab[(x >> 40) & 0xFF]
        c += tab[(x >> 48) & 0xFF]
        c += tab[(x >> 56) & 0xFF]
        return c
    x = np.asarray(x).astype(np.uint64)
    tab = _pctab(np)
    c = tab[x & 0xFF]
    for s in (8, 16, 24, 32, 40, 48, 56):
        c = c + tab[(x >> s) & 0xFF]
    return c


def bit_agreement(cand_bytes, target_bytes):
    """
    候选密钥流 vs 目标密钥流的比特一致度。
    cand_bytes  : [N, L] uint8
    target_bytes: [L]    uint8  或 [M, L]
    返回 [N] 或 [N, M] 的 0..L*8 一致比特数。
    """
    T = xp()
    if BACKEND.startswith("torch"):
        c = cand_bytes.to(torch.int64)
        t = target_bytes.to(torch.int64)
        if t.dim() == 1:
            x = c ^ t.unsqueeze(0)
            return (8 * c.shape[1]) - popcount_matrix(x).sum(-1)
        if c.shape[0] == 0:                              # 空候选：避免 torch.cat([]) 报错
            return torch.zeros((0, t.shape[0]), dtype=torch.int64, device=c.device)
        # [N,L] vs [M,L] -> 分块避免 N*M 爆显存
        outs = []
        step = max(1, 2 ** 26 // max(1, t.shape[0] * c.shape[1]))
        for i in range(0, c.shape[0], step):
            blk = c[i:i + step]
            x = blk.unsqueeze(1) ^ t.unsqueeze(0)          # [n,M,L]
            outs.append((8 * c.shape[1]) - popcount_matrix(x).sum(-1))
        return torch.cat(outs, 0)
    c = np.asarray(cand_bytes)
    t = np.asarray(target_bytes)
    if t.ndim == 1:
        x = c ^ t[None, :]
    else:
        x = c[:, None, :] ^ t[None, :, :]
    # 对字节内 8 bit 与全部 L 字节求和：1D 目标得 [N]，2D 目标得 [N,M]
    return 8 * c.shape[1] - popcount_matrix(x).astype(np.int64).sum(-1)


# ---------------------------------------------------------------- MT19937
def mt19937_keys(seeds, twist=True, nbytes=24):
    """
    给定种子批量生成 std::mt19937 的前 nbytes 个输出低字节（默认 24B = AES-192 密钥）。
    seeds  : [N] 整型张量/数组（种子按 uint32 取模，负数等价于 2^32 - |seed| 附近回绕）
    twist  : False=init_genrand 后直接 temper 初始状态（魔改版常见）；
             True=标准 std::mt19937（取数前先 twist）
    nbytes : 需要的密钥字节数 1~24；只取前几字节时按需截断，不做无用功
    返回   : [N, nbytes] uint8
    """
    T = xp()
    mask = M32()
    n = int(nbytes)
    if not 1 <= n <= 24:
        raise ValueError(f'nbytes 需在 1..24，收到 {n}')
    if BACKEND.startswith("torch"):
        s = seeds.to(torch.int64) & mask
        N = s.shape[0]
        rows = 624 if twist else n          # 不 twist 时只需前 n 个状态（串行依赖）
        st = torch.empty((N, rows), dtype=torch.int64, device=DEV)
        st[:, 0] = s
        for i in range(1, rows):
            prev = st[:, i - 1]
            st[:, i] = ((1812433253 * (prev ^ (prev >> 30))) + i) & mask
        if twist:
            for i in range(624):
                y = (st[:, i] & 0x80000000) + (st[:, (i + 1) % 624] & 0x7FFFFFFF)
                v = st[:, (i + 397) % 624] ^ (y >> 1)
                v = torch.where((y & 1) == 1, v ^ 0x9908B0DF, v)
                st[:, i] = v & mask
        y = st[:, :n].clone()
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        y &= mask
        return (y & 0xFF).to(torch.uint8)

    # numpy 版：跨种子向量化（旧版逐种子纯 Python 循环，慢 ~1000 倍）+ 行优先布局
    s = np.asarray(seeds)
    if s.ndim != 1:
        raise ValueError('seeds 需为 1D 整型数组')
    u = s.astype(np.uint64) & np.uint64(mask)     # int64 -> uint64 走补码回绕，语义同 torch
    N = u.shape[0]
    rows = 624 if twist else n
    st = np.empty((rows, N), dtype=np.uint64)
    st[0] = u
    for i in range(1, rows):
        prev = st[i - 1]
        st[i] = (np.uint64(1812433253) *
                 (prev ^ (prev >> np.uint64(30))) + np.uint64(i)) & np.uint64(mask)
    if twist:
        for i in range(624):
            y = (st[i] & np.uint64(0x80000000)) + (st[(i + 1) % 624] & np.uint64(0x7FFFFFFF))
            v = st[(i + 397) % 624] ^ (y >> np.uint64(1))
            v = v ^ np.where((y & np.uint64(1)) == np.uint64(1),
                             np.uint64(0x9908B0DF), np.uint64(0))
            st[i] = v & np.uint64(mask)
    y = st[:n].copy()
    y ^= y >> np.uint64(11)
    y ^= (y << np.uint64(7)) & np.uint64(0x9D2C5680)
    y ^= (y << np.uint64(15)) & np.uint64(0xEFC60000)
    y ^= y >> np.uint64(18)
    y &= np.uint64(mask)
    return (y & np.uint64(0xFF)).astype(np.uint8).T.copy()


# ---------------------------------------------------------------- AES-192
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16],
    dtype=np.uint8)
INV_SBOX = np.zeros(256, dtype=np.uint8)
INV_SBOX[SBOX] = np.arange(256, dtype=np.uint8)
RCON = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36,0x6c,0xd8,0xab,0x4d]


def _xtime(a):
    """GF(2^8) 乘 2（张量版）。"""
    hi = (a >> 7) & 1
    return ((a << 1) ^ (0x1b * hi)) & 0xFF


def _sub_bytes(s, sbox_table, a_idx=None, b_val=None):
    """
    SubBytes。若给了 a_idx/b_val（每候选不同的 S-box 破坏），
    则对 s[x]==a_idx 的位置替换为 b_val。
    s        : [B,16] uint8(张量)
    sbox_table: 256 uint8 张量
    a_idx    : [B,1] 或 None
    b_val    : [B,1] 或 None
    """
    T = xp()
    if BACKEND.startswith("torch"):
        y = sbox_table[s.long()]
        if a_idx is not None:
            y = torch.where(s == a_idx, b_val.expand_as(s), y)
        return y
    y = sbox_table[s]
    if a_idx is not None:
        y = np.where(s == a_idx, b_val, y)
    return y


def _expand_key_np(keybytes):
    """numpy 标准 S-box 密钥扩展（AES-128/192/256），返回 [Nr+1,16] uint8。"""
    kb = np.frombuffer(keybytes, dtype=np.uint8)
    Nk = len(kb) // 4
    if Nk not in (4, 6, 8):
        raise ValueError(f'AES 密钥长度需为 16/24/32 字节，收到 {len(kb)} 字节')
    Nr = Nk + 6
    w = [kb[4 * i:4 * i + 4].copy() for i in range(Nk)]
    for i in range(Nk, 4 * (Nr + 1)):
        tmp = w[i - 1].copy()
        if i % Nk == 0:
            tmp = np.concatenate([tmp[1:], tmp[:1]])
            tmp = SBOX[tmp]
            tmp[0] ^= RCON[i // Nk - 1]
        elif Nk > 6 and i % Nk == 4:
            tmp = SBOX[tmp]
        w.append(w[i - Nk] ^ tmp)
    return np.concatenate(w, axis=0).reshape(Nr + 1, 16)


def aes_ecb(block, key, sbox_table=None, a_idx=None, b_val=None):
    """
    批量 AES-128/192/256 ECB 加密单块（无 padding）。
    block : [B,16] uint8      key : [16|24|32] 或 [B,16|24|32]
    sbox_table : 256 uint8（默认标准 S-box）
    a_idx/b_val: [B,1] 指定"某位置被破坏"（cryp-12025 的 bug 模型）
    返回  : [B,16] uint8
    """
    T = xp()
    is_t = BACKEND.startswith("torch")
    klen = int(key.shape[1] if key.ndim == 2 else key.shape[0])
    Nk = klen // 4
    if Nk not in (4, 6, 8):
        raise ValueError(f'AES 密钥需 16/24/32 字节，收到 {klen} 字节')
    Nr = Nk + 6
    # S-box 表统一转成后端原生 256 项 uint8 表（允许调用方传 numpy 表/torch CPU 表）
    if sbox_table is None:
        sbox_table = torch.from_numpy(SBOX).to(DEV) if is_t else SBOX
    if is_t:
        if isinstance(sbox_table, np.ndarray):
            sbox_table = torch.from_numpy(
                np.ascontiguousarray(sbox_table, dtype=np.uint8)).to(DEV)
        else:
            sbox_table = sbox_table.to(device=DEV, dtype=torch.uint8)
        if int(sbox_table.numel()) != 256:
            raise ValueError(f'sbox_table 需为 256 项，收到 {int(sbox_table.numel())} 项')
    else:
        sbox_table = np.asarray(sbox_table, dtype=np.uint8)
        if sbox_table.size != 256:
            raise ValueError(f'sbox_table 需为 256 项，收到 {sbox_table.size} 项')
    B = block.shape[0]

    # ---- 快速路径：单一密钥（批量候选只差 S-box）时，round key 在 CPU 上算一次 ----
    fast_rk = None
    if key.ndim == 1:
        kb = from_dev(key).astype(np.uint8).tobytes() if is_t else np.asarray(key, dtype=np.uint8).tobytes()
        fast_rk = _expand_key_np(kb)                       # [Nr+1,16] uint8 ndarray
        if is_t:
            fast_rk = torch.from_numpy(fast_rk).to(DEV)
        rk_flat = fast_rk.reshape(-1) if is_t else fast_rk.reshape(-1)
        s = block ^ rk_flat[:16]
        # Nr 已在函数入口按密钥长度算好（128/192/256 → 10/12/14 轮）
        for rnd in range(1, Nr + 1):
            s = _sub_bytes(s, sbox_table, a_idx, b_val)
            if is_t:
                m = s.reshape(B, 4, 4)
                m = torch.stack([torch.roll(m[:, :, r], shifts=-r, dims=1) for r in range(4)], dim=2)
                s = m.reshape(B, 16)
            else:
                m = s.reshape(B, 4, 4)
                m = np.stack([np.roll(m[:, :, r], shift=-r, axis=1) for r in range(4)], axis=2)
                s = m.reshape(B, 16)
            if rnd != Nr:
                c0, c1, c2, c3 = s[:, 0::4], s[:, 1::4], s[:, 2::4], s[:, 3::4]
                n0 = _xtime(c0) ^ (_xtime(c1) ^ c1) ^ c2 ^ c3
                n1 = c0 ^ _xtime(c1) ^ (_xtime(c2) ^ c2) ^ c3
                n2 = c0 ^ c1 ^ _xtime(c2) ^ (_xtime(c3) ^ c3)
                n3 = (_xtime(c0) ^ c0) ^ c1 ^ c2 ^ _xtime(c3)
                s = (torch.stack([n0, n1, n2, n3], dim=2).reshape(B, 16) if is_t
                     else np.stack([n0, n1, n2, n3], axis=2).reshape(B, 16))
            s = s ^ rk_flat[16 * rnd: 16 * (rnd + 1)]
        return s

    # ---- 通用路径：密钥逐候选不同 ----
    if key.ndim == 1:
        key = key.unsqueeze(0).expand(B, -1) if is_t else np.tile(key, (B, 1))
    # ---- key expansion（Nr 轮 → Nr+1 个 round key）----
    w = [key[:, 4 * i:4 * i + 4] for i in range(Nk)]
    for i in range(Nk, 4 * (Nr + 1)):
        tmp = w[i - 1]
        if i % Nk == 0:
            if is_t:
                rot = torch.cat([tmp[:, 1:], tmp[:, :1]], dim=1)
                sub = sbox_table[rot.long()]
                sub = torch.cat([(sub[:, :1] ^ RCON[i // Nk - 1]), sub[:, 1:]], dim=1)
            else:
                rot = np.concatenate([tmp[:, 1:], tmp[:, :1]], axis=1)
                sub = sbox_table[rot]
                sub[:, 0] ^= RCON[i // Nk - 1]
        elif Nk > 6 and i % Nk == 4:
            sub = (sbox_table[tmp.long()] if is_t else sbox_table[tmp])
        else:
            sub = tmp
        w.append(w[i - Nk] ^ sub)
    rk = [None] * (Nr + 1)
    for r in range(Nr + 1):
        rk[r] = (torch.cat([w[4 * r + 0], w[4 * r + 1], w[4 * r + 2], w[4 * r + 3]], dim=1)
                 if is_t else
                 np.concatenate([w[4 * r + 0], w[4 * r + 1], w[4 * r + 2], w[4 * r + 3]], axis=1))
    s = block ^ rk[0]
    for rnd in range(1, Nr + 1):
        s = _sub_bytes(s, sbox_table, a_idx, b_val)
        # ShiftRows（列主序 [c*4+r]）
        if is_t:
            m = s.reshape(B, 4, 4)                      # [B, c, r]
            m = torch.stack([torch.roll(m[:, :, r], shifts=-r, dims=1) for r in range(4)], dim=2)
            s = m.reshape(B, 16)
        else:
            m = s.reshape(B, 4, 4)
            m = np.stack([np.roll(m[:, :, r], shift=-r, axis=1) for r in range(4)], axis=2)
            s = m.reshape(B, 16)
        if rnd != Nr:
            c0, c1, c2, c3 = s[:, 0::4], s[:, 1::4], s[:, 2::4], s[:, 3::4]
            # MixColumns
            def xt(a): return _xtime(a)
            n0 = xt(c0) ^ (xt(c1) ^ c1) ^ c2 ^ c3
            n1 = c0 ^ xt(c1) ^ (xt(c2) ^ c2) ^ c3
            n2 = c0 ^ c1 ^ xt(c2) ^ (xt(c3) ^ c3)
            n3 = (xt(c0) ^ c0) ^ c1 ^ c2 ^ xt(c3)
            if is_t:
                s = torch.stack([n0, n1, n2, n3], dim=2).reshape(B, 16)
            else:
                s = np.stack([n0, n1, n2, n3], axis=2).reshape(B, 16)
        s = s ^ rk[rnd]
    return s


# ---------------------------------------------------------------- 子命令
# 兼容旧名：ctf_tools.py 以 24 字节密钥调用，语义不变
aes192_ecb = aes_ecb


def _cuda_sync():
    """只在真正用 CUDA 时同步；torch-cpu 后端调用 torch.cuda.* 会抛 RuntimeError。"""
    if BACKEND == "torch-cuda":
        torch.cuda.synchronize()


def _bench_text_lines(rows, markdown=False):
    """bench 人读/Markdown 输出行（纯函数，便于自测输出契约）。"""
    if markdown:
        lines = ['| 内核 | 吞吐 | 批量 | 耗时 |', '|---|---|---|---|']
        lines += [f'| `{label}` | {rate} {unit} | {cnt} | {sec*1e3:.0f} ms |'
                  for label, rate, unit, cnt, sec in rows]
        return lines
    return [f"[{label:<8}] {rate} {unit}  ({sec*1e3:.0f} ms, {cnt})"
            for label, rate, unit, cnt, sec in rows]


def _bench_json_objs(rows):
    """bench --json 的逐内核事件对象（纯函数，便于自测输出契约）。"""
    return [{'event': 'bench', 'kernel': label, 'rate': float(rate), 'unit': unit,
             'count': cnt, 'seconds': round(sec, 6)}
            for label, rate, unit, cnt, sec in rows]


def cmd_bench(a):
    T = xp()
    md = bool(getattr(a, 'markdown', False))
    js = C.json_mode()
    # 横幅属于提示：--markdown 时 stdout 只留 Markdown 表格，--json 时只留 JSON 行
    text_out = not (md or js or C.quiet())
    if text_out:
        print(f"[backend] {BACKEND}  device={DEV}")
        if BACKEND == "torch-cuda":
            print(f"[device ] {torch.cuda.get_device_name(0)}  "
                  f"VRAM {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    def warm(fn, *args):
        """预热：触发 JIT 编译 + 缓存分配，避免首次调用被计入耗时。"""
        for _ in range(3):
            fn(*args)
        _cuda_sync()

    rows = []

    def _timed(fn, *args):
        warm(fn, *args)
        t0 = time.time()
        fn(*args)
        _cuda_sync()
        return time.time() - t0

    # 1) popcount 吞吐
    n = 1 << 22
    x = (T.randint(0, 1 << 31, (n,), dtype=T.int64, device=DEV) if BACKEND.startswith("torch")
         else np.random.randint(0, 1 << 31, n, dtype=np.int64))
    dt = _timed(popcount_matrix, x)
    rows.append(('popcount', f"{n/dt/1e6:.1f}", 'M 个/s', f'{n} 个 32bit', dt))

    # 2) mt19937 密钥生成
    m = 1 << 16
    s = (T.arange(m, dtype=T.int64, device=DEV) if BACKEND.startswith("torch")
         else np.arange(m, dtype=np.int64))
    dt = _timed(mt19937_keys, s, False)
    rows.append(('mt19937', f"{m/dt/1e3:.1f}", 'K keys/s', f'{m} 个种子', dt))

    # 3) AES-128/192/256 单块批量
    for kl in (16, 24, 32):
        blk = (T.zeros((1 << 14, 16), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
               else np.zeros((1 << 14, 16), dtype=np.uint8))
        k = (T.zeros((kl,), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
             else np.zeros((kl,), dtype=np.uint8))
        ai = (T.full((blk.shape[0], 1), 0x42, dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
              else np.full((blk.shape[0], 1), 0x42, dtype=np.uint8))
        dt = _timed(aes_ecb, blk, k, None, ai, ai)
        rows.append((f'aes{kl * 8}', f"{blk.shape[0]/dt/1e3:.1f}", 'K blocks/s',
                     f'{blk.shape[0]} 块', dt))

    if js:
        for obj in _bench_json_objs(rows):
            C.emit(obj)
    else:
        for line in _bench_text_lines(rows, md):
            print(line)


def cmd_mtscan(a):
    """暴力 mt19937 种子 → 24B 密钥，比对目标。"""
    T = xp()
    target = parse_hex(a.key_hex, '--key-hex')
    if target is None:
        return C.fail(f"--key-hex 不是合法裸 hex：{a.key_hex!r}")
    L = len(target)
    if not 1 <= L <= 24:
        return C.fail(f"--key-hex 需 1~24 字节（mt19937 每次只生成 24B 密钥），实际 {L} 字节")
    if a.seed_count <= 0:
        return C.fail("--seed-count 必须 > 0")
    if a.seed_start < -(2 ** 63) or a.seed_start + a.seed_count > 2 ** 63 - 1:
        return C.fail("--seed-start/--seed-count 超出 int64 可表示范围")
    twist = (a.twist == 1)
    tgt_t = (T.tensor(list(target), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
             else np.frombuffer(target, dtype=np.uint8))

    def _key_of(seed):
        """按种子重算完整密钥（命中复核 / 过滤断点里的陈旧命中）。"""
        seeds = (T.tensor([seed], dtype=T.int64, device=DEV) if BACKEND.startswith("torch")
                 else np.array([seed], dtype=np.int64))
        return np.asarray(from_dev(mt19937_keys(seeds, twist=twist, nbytes=L)[0]),
                          dtype=np.uint8)

    def _step(lo, hi):
        cnt = hi - lo
        seeds = (T.arange(lo, hi, dtype=T.int64, device=DEV) if BACKEND.startswith("torch")
                 else np.arange(lo, hi, dtype=np.int64))
        keys = mt19937_keys(seeds, twist=twist, nbytes=L)
        eq = (keys == tgt_t).all(axis=1)
        out = []
        if bool(eq.any()):
            out = [int(lo) + int(i) for i in hit_indices(eq)]
        return cnt, out

    total, found, dt = C.chunked(_step, a.seed_count, start=a.seed_start, chunk=a.chunk,
                                 label='mtscan', checkpoint=getattr(a, 'checkpoint', ''),
                                 reset=getattr(a, 'reset_checkpoint', False),
                                 progress=not getattr(a, 'no_progress', False),
                                 ckpt_meta={'kind': 'mtscan', 'key_hex': target.hex(),
                                            'twist': int(a.twist)},
                                 on_hit=lambda h: True)
    C.info(f"[done] {total:,} 个种子 / {dt:.2f}s = {total/max(dt,1e-9)/1e6:.1f} M seeds/s")
    # 断点里可能残留其它任务/其它目标的历史命中：只承认与当前目标真正一致的种子
    verified, stale = [], 0
    seen = set()
    for s in found:
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
            stale += 1
            continue
        s = int(s)
        if s in seen:
            continue
        seen.add(s)
        if _key_of(s).tobytes() == target:
            verified.append(s)
        else:
            stale += 1
    if stale:
        C.info(f"[warn] 忽略断点中 {stale} 条与当前 --key-hex 不匹配的历史命中")
    if verified:
        for s in verified:
            k = bytes(_key_of(s)[:L]).hex()
            C.emit({'event': 'hit', 'kind': 'mtscan', 'seed': s, 'key': k,
                    'text': f"  ★ 命中 seed={s}  key={k}"})
    else:
        C.emit({'event': 'result', 'kind': 'mtscan', 'hits': 0, 'scanned': total,
                'text': "  未命中"})


def cmd_sboxscan(a):
    """枚举 S-box 破坏 (a,b) → AES-128/192/256 加密，比对目标密文。"""
    T = xp()
    plain = parse_hex(a.plain_hex, '--plain-hex')
    target = parse_hex(a.cipher_hex, '--cipher-hex')
    key = parse_hex(a.key_hex, '--key-hex')
    for opt, val, raw in (('--plain-hex', plain, a.plain_hex),
                          ('--cipher-hex', target, a.cipher_hex),
                          ('--key-hex', key, a.key_hex)):
        if val is None:
            return C.fail(f"{opt} 不是合法裸 hex：{raw!r}")
    if len(plain) == 0 or len(plain) % 16 or len(plain) != len(target):
        return C.fail(f"明文/密文须等长、16 字节整数倍且非空（明文 {len(plain)}B，密文 {len(target)}B）")
    if len(key) not in (16, 24, 32):
        return C.fail(f"AES 密钥需 16/24/32 字节，实际 {len(key)} 字节")

    pairs = [(x, y) for x in range(256) for y in range(256) if x != y]
    pairs = np.array(pairs, dtype=np.int64)
    n = pairs.shape[0]
    C.info(f"[scan] {n} 种 (a,b) 组合，{len(plain)//16} 块")

    blk0 = (T.tensor(list(plain[:16]), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
            else np.frombuffer(plain[:16], dtype=np.uint8))
    tgt0 = (T.tensor(list(target[:16]), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
            else np.frombuffer(target[:16], dtype=np.uint8))
    key_t = (T.tensor(list(key), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
             else np.frombuffer(key, dtype=np.uint8))

    def _step(lo, hi):
        p = pairs[lo:hi]
        ai = (T.tensor(p[:, 0], dtype=T.uint8, device=DEV).unsqueeze(1) if BACKEND.startswith("torch")
              else p[:, 0].astype(np.uint8)[:, None])
        bv = (T.tensor(INV_SBOX[p[:, 1]], dtype=T.uint8, device=DEV).unsqueeze(1) if BACKEND.startswith("torch")
              else INV_SBOX[p[:, 1]][:, None])
        blk = blk0.unsqueeze(0).expand(p.shape[0], 16) if BACKEND.startswith("torch") \
            else np.tile(blk0, (p.shape[0], 1))
        out = aes192_ecb(blk, key_t, None, ai, bv)
        eq = (out == tgt0).all(axis=1)
        got = []
        if bool(eq.any()):
            for j in hit_indices(eq):
                got.append([int(p[j][0]), int(p[j][1])])
        return hi - lo, got

    total, hits, dt = C.chunked(_step, n, chunk=a.chunk, label='sboxscan',
                                checkpoint=getattr(a, 'checkpoint', ''),
                                reset=getattr(a, 'reset_checkpoint', False),
                                progress=not getattr(a, 'no_progress', False),
                                ckpt_meta={'kind': 'sboxscan', 'plain_hex': plain.hex(),
                                           'cipher_hex': target.hex(), 'key_hex': key.hex()})
    C.info(f"[done] {total} 组合 / {dt:.2f}s = {total/max(dt,1e-9)/1e3:.1f} K/s")

    # 用全部块验证；顺带过滤断点文件里可能残留的其它任务/其它目标命中。
    # 批量验证（每批最多 8192 个候选 × 每块一次 aes_ecb），避免逐候选 Python 循环。
    hits_clean, dropped = [], 0
    seen = set()
    for h in hits:
        if not isinstance(h, (list, tuple)) or len(h) != 2:
            dropped += 1
            continue
        try:
            x, y = int(h[0]), int(h[1])
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (0 <= x < 256 and 0 <= y < 256) or x == y:
            dropped += 1
            continue
        if (x, y) in seen:
            continue
        seen.add((x, y))
        hits_clean.append((x, y))
    tgt_np = np.frombuffer(target, dtype=np.uint8)
    verified = []
    vbatch = 8192
    for i in range(0, len(hits_clean), vbatch):
        blk_hits = hits_clean[i:i + vbatch]
        H = len(blk_hits)
        xs = np.array([h[0] for h in blk_hits], dtype=np.uint8)
        ys = np.array([h[1] for h in blk_hits], dtype=np.uint8)
        keep = np.ones(H, dtype=bool)
        for off in range(0, len(plain), 16):
            pb = np.frombuffer(plain[off:off + 16], dtype=np.uint8)
            blk = (T.tensor(np.tile(pb, (H, 1)), dtype=T.uint8, device=DEV)
                   if BACKEND.startswith("torch") else np.tile(pb, (H, 1)))
            ai = (T.tensor(xs, dtype=T.uint8, device=DEV).unsqueeze(1)
                  if BACKEND.startswith("torch") else xs[:, None])
            bv = (T.tensor(INV_SBOX[ys], dtype=T.uint8, device=DEV).unsqueeze(1)
                  if BACKEND.startswith("torch") else INV_SBOX[ys][:, None])
            out = np.asarray(from_dev(aes192_ecb(blk, key_t, None, ai, bv)))
            keep &= (out == tgt_np[off:off + 16]).all(axis=1)
            if not keep.any():
                break
        dropped += int(H - int(keep.sum()))
        verified.extend([h for h, v in zip(blk_hits, keep) if v])
    if dropped:
        C.info(f"[warn] 丢弃 {dropped} 条未通过全块验证/非法的候选（断点可能来自其它目标）")
    for (x, y) in verified:
        C.emit({'event': 'hit', 'kind': 'sboxscan', 'a': x, 'b': y, 'verified': True,
                'text': f"  ★ (a,b)=({x},{y})  全块验证=通过"})
    if not verified:
        C.emit({'event': 'result', 'kind': 'sboxscan', 'hits': 0, 'text': "  未命中"})


def cmd_ksscore(a):
    """候选密钥流 vs 目标/样本 的比特一致度打分（统计淘汰）。"""
    T = xp()
    tgt = parse_hex(a.target_hex, '--target-hex')
    if tgt is None:
        return C.fail(f"--target-hex 不是合法裸 hex：{a.target_hex!r}")
    L = len(tgt)
    if L == 0:
        return C.fail("--target-hex 不能为空")
    if a.candidates <= 0:
        return C.fail("--candidates 必须 > 0")
    tgt_t = (T.tensor(list(tgt), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
             else np.frombuffer(tgt, dtype=np.uint8))
    best = {'score': -1, 'cand': None}

    def _step(lo, hi):
        cnt = hi - lo
        cand = (T.randint(0, 256, (cnt, L), dtype=T.int64, device=DEV).to(T.uint8)
                if BACKEND.startswith("torch") else
                np.random.randint(0, 256, (cnt, L), dtype=np.uint8))
        sc = bit_agreement(cand, tgt_t)
        sc_np = from_dev(sc)
        mx = int(sc_np.max())
        if mx > best['score']:
            i = int(np.argmax(sc_np))
            best['score'] = mx
            best['cand'] = from_dev(cand)[i].tobytes().hex()
        return cnt, []

    done, _hits, dt = C.chunked(_step, a.candidates, chunk=a.chunk, label='ksscore',
                                checkpoint=getattr(a, 'checkpoint', ''),
                                reset=getattr(a, 'reset_checkpoint', False),
                                progress=not getattr(a, 'no_progress', False),
                                ckpt_meta={'kind': 'ksscore', 'target_hex': tgt.hex()})
    C.info(f"[done] {done:,} 候选 × {L}B / {dt:.2f}s = {done/max(dt,1e-9)/1e6:.2f} M cand/s")
    C.emit({'event': 'result', 'kind': 'ksscore', 'scanned': done, 'bits': L * 8,
            'best_score': best['score'], 'best_candidate': best['cand'],
            'text': f"  最佳比特一致度 {best['score']}/{L*8}   候选 {best['cand']}"})


def cmd_lfsrscan(a):
    """
    并行跑多条 Fibonacci LFSR，匹配目标比特流。
    --taps  "25:10,11,24" 或 "25:10,11,24;28:12,22,25;27:1,4,19,21,23"（多组）
    """
    T = xp()
    if not a.bits or set(a.bits) - set("01"):
        return C.fail("--bits 只能是 0/1 组成的非空比特串")
    target_bits = np.array([int(c) for c in a.bits], dtype=np.int64)
    groups = []
    for g in a.taps.split(";"):
        g = g.strip()
        parts = g.split(":")
        if len(parts) != 2:
            return C.fail(f"--taps 组格式应为 n:tap1,tap2,...，收到 {g!r}")
        n_str, tap_str = parts
        try:
            n = int(n_str)
            taps = [int(t) for t in tap_str.split(",") if t.strip()]
        except ValueError:
            return C.fail(f"--taps 组 {g!r} 含非整数：n 与每个 tap 都必须是十进制整数")
        if n <= 0 or not taps or any(t < 1 or t > n for t in taps):
            return C.fail(f"--taps 组 {g!r} 非法：状态数 n 需 >0，且每个 tap 必须在 1..n")
        groups.append((n, taps))
    if a.trials <= 0:
        return C.fail("--trials 必须 > 0")
    if a.rng_seed < 0:
        return C.fail("--rng-seed 必须 >= 0")
    Tlen = len(target_bits)
    C.info(f"[scan] {len(groups)} 组 LFSR，目标 {Tlen} 拍")

    rng = np.random.default_rng(a.rng_seed)
    trials = a.trials
    best = -1
    t0 = time.time()
    for g in range(len(groups)):
        n, taps = groups[g]
        try:
            state = rng.integers(0, 2, size=(trials, n), dtype=np.int64)
        except (MemoryError, ValueError) as exc:
            return C.fail(f"初始状态内存不足（trials={trials}, n={n}）：{exc}")
        try:
            if BACKEND.startswith("torch"):
                st = torch.tensor(state, device=DEV)
                taps_t = torch.tensor([t - 1 for t in taps], device=DEV)
                score = torch.zeros(trials, dtype=torch.int64, device=DEV)
                # 逐拍累加一致位数（不再保留 [trials, 拍数] 大矩阵，长比特流也不爆内存）
                for k in range(Tlen):
                    fb = st[:, taps_t[0]].clone()
                    for t in taps_t[1:]:
                        fb ^= st[:, t]
                    score += (fb == target_bits[k]).to(torch.int64)
                    st = torch.cat([fb.unsqueeze(1), st[:, :-1]], dim=1)
                m = int(score.max().item())
            else:
                st = state.copy()
                taps_i = [t - 1 for t in taps]
                score = np.zeros(trials, dtype=np.int64)
                for k in range(Tlen):
                    fb = st[:, taps_i[0]].copy()
                    for t in taps_i[1:]:
                        fb ^= st[:, t]
                    score += (fb == target_bits[k])
                    st = np.concatenate([fb[:, None], st[:, :-1]], axis=1)
                m = int(score.max())
        except MemoryError as exc:
            return C.fail(f"内存/显存不足（trials={trials}, n={n}, bits={Tlen}）：{exc}")
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                return C.fail(f"显存不足（trials={trials}, n={n}, bits={Tlen}）：{exc}")
            raise
        if m > best:
            best = m
        C.emit({'event': 'result', 'kind': 'lfsrscan', 'group': g + 1, 'n': n, 'taps': taps,
                'best': int(m), 'bits': Tlen,
                'text': f"  组{g+1} (n={n}, taps={taps}): 最佳 {m}/{Tlen} 位"})
    dt = time.time() - t0
    C.info(f"[done] {trials*len(groups):,} 组状态 × {Tlen} 拍 / {dt:.2f}s")


# ---------------------------------------------------------------- CLI
# 全局参数（7 个）的唯一定义；主解析器和各子命令解析器都注册一遍，
# 子命令后的副本用 SUPPRESS 默认值，这样"全局参数写在子命令前后都行"。
_GLOBAL_SPECS = (
    (('--backend',), dict(default='auto', choices=['auto', 'torch', 'numpy'],
                          help='计算后端：auto=torch-cuda→torch-cpu→numpy（默认 auto；子命令前后均可）')),
    (('--chunk',), dict(type=int, default=1 << 16,
                        help='每批候选数（默认 65536；显存不足时自动减半；子命令前后均可）')),
    (('--json',), dict(action='store_true', default=False,
                       help='结果行改为单行 JSON（机器可读；子命令前后均可）')),
    (('--quiet',), dict(action='store_true', default=False,
                        help='抑制横幅与进度，只保留结果行（子命令前后均可）')),
    (('--no-progress',), dict(action='store_true', default=False,
                              help='不打印进度/ETA，结果仍输出（子命令前后均可）')),
    (('--checkpoint',), dict(default='', metavar='FILE',
                             help='断点文件：每块落盘一次，中断后从上次位置续跑（子命令前后均可）')),
    (('--reset-checkpoint',), dict(action='store_true', default=False,
                                   help='忽略已有断点，从头开始（子命令前后均可）')),
)


def _add_global_args(parser, *, after_sub=False):
    """注册全局参数；after_sub=True 时默认置 SUPPRESS，避免覆盖子命令前已解析出的值。"""
    for flags, kw in _GLOBAL_SPECS:
        kw = dict(kw)
        if after_sub:
            kw['default'] = argparse.SUPPRESS
        parser.add_argument(*flags, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="gpu_crack — GPU 批量爆破/校验框架（完整说明见 CLI_REFERENCE.md）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    _add_global_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True,
                            metavar="{bench,mtscan,sboxscan,ksscore,lfsrscan}")

    p = sub.add_parser("bench", help="基准测试：当前机器/后端的吞吐（正式搜索前先跑）")
    _add_global_args(p, after_sub=True)
    p.add_argument("--markdown", action="store_true",
                   help="输出 Markdown 表格（可直接写入 docs/bench.md；stdout 只留表格）")

    p = sub.add_parser("mtscan", help="暴力 mt19937 种子 -> 24B 密钥，与目标比对")
    _add_global_args(p, after_sub=True)
    p.add_argument("--key-hex", required=True, help="目标密钥/密钥流前缀，裸 hex，1~24 字节")
    p.add_argument("--seed-start", type=int, default=0, help="起始种子（含），默认 0")
    p.add_argument("--seed-count", type=int, default=1 << 24, help="扫描种子个数，默认 16777216")
    p.add_argument("--twist", type=int, default=0, choices=[0, 1],
                   help="0=init_genrand 后直接读状态（魔改版常见，默认）；1=标准 std::mt19937")

    p = sub.add_parser("sboxscan", help="枚举 AES-128/192/256 S-box 破坏 (a,b)，匹配明文/密文对")
    _add_global_args(p, after_sub=True)
    p.add_argument("--plain-hex", required=True, help="明文块序列，裸 hex，16 字节整数倍且非空")
    p.add_argument("--cipher-hex", required=True, help="目标密文，裸 hex，与明文等长")
    p.add_argument("--key-hex", required=True, help="AES 密钥，裸 hex，16/24/32 字节")

    p = sub.add_parser("ksscore", help="候选密钥流 vs 目标比特一致度打分（统计淘汰）")
    _add_global_args(p, after_sub=True)
    p.add_argument("--target-hex", required=True, help="目标密钥流，裸 hex，>=1 字节")
    p.add_argument("--candidates", type=int, default=1 << 20, help="随机候选个数，默认 1048576")

    p = sub.add_parser("lfsrscan", help="多条 LFSR 并行仿真，匹配目标比特流")
    _add_global_args(p, after_sub=True)
    p.add_argument("--taps", required=True, help='抽头，如 "25:10,11,24;28:12,22,25"（tap 从 1 开始）')
    p.add_argument("--bits", required=True, help="目标输出比特流，仅含 0/1 的字符串")
    p.add_argument("--trials", type=int, default=1 << 16, help="每组随机初始状态数，默认 65536")
    p.add_argument("--rng-seed", type=int, default=1, help="随机种子，默认 1（双后端结果一致）")

    a = ap.parse_args(argv)
    C.apply_output_args(a)
    if a.chunk < 1:
        return C.fail("--chunk 必须 >= 1")
    name, dev = init_backend(a.backend)
    C.info(f"[backend] {name}  device={dev}\n")
    rc = {"bench": cmd_bench, "mtscan": cmd_mtscan, "sboxscan": cmd_sboxscan,
          "ksscore": cmd_ksscore, "lfsrscan": cmd_lfsrscan}[a.cmd](a)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
