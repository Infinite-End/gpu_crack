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
  selftest  已知向量自测（18 项；全 PASS 退出码 0，失败 1）

全局参数（必须写在子命令之前）
------------------------------
  --backend {auto,torch,numpy}   默认 auto
  --chunk N                      每批候选数 / 断点粒度（chacha --brute：GPU 默认 262144，其它 16384）
  --checkpoint FILE              断点文件（chacha --brute、xorbrute 支持续跑）
  --reset-checkpoint             忽略已有断点
  --json                         结果行改为单行 JSON 事件（stdout 只有事件）
  --quiet / --no-progress        抑制横幅与进度/ETA（结果仍输出）

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
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_crack as G                      # noqa: E402
import ctf_common as C                     # noqa: E402


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
    """读取密钥文件：每行一个 hex 密钥；支持 0x 前缀、空行、# 注释行、UTF-8 BOM。"""
    try:
        # utf-8-sig：Windows 记事本/Excel 导出的文件常带 BOM，不剥掉会把首行密钥弄坏
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
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


# ---------------------------------------------------------------- 输出（JSON 契约）
# 文本模式保留原有中文文案；--json 模式只往 stdout 写单行 JSON 事件
# （event=result|hit|error，kind=子命令名），进度/横幅仍走 stderr。
def _json_result(kind, **fields):
    """--json 模式输出一条 result 事件（文本模式请调用方 print 原本文案）。"""
    C.emit({'event': 'result', 'kind': kind, **fields})


def _json_hit(kind, **fields):
    """--json 模式输出一条 hit 事件（★ 命中行）。"""
    C.emit({'event': 'hit', 'kind': kind, **fields})


def _json_locate(kind, **fields):
    """--json 模式输出一条 locate 事件（纠错定位结果）。"""
    C.emit({'event': 'locate', 'kind': kind, **fields})


# =====================================================================
# 1. ChaCha20
# =====================================================================
def _is_gpu():
    """当前后端是否为 torch-CUDA（GPU 快速路径开关）。"""
    return _is_torch() and G.BACKEND == 'torch-cuda'


_CHACHA_CONST = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)


def _rotl(x, n):
    # 兼容旧内部名（numpy 标量路径已不再使用；保留给外部脚本引用）
    return ((x << n) | (x >> (32 - n))) & M32


def _chacha_parse(nonce, counter, nbytes, rfc7539):
    """公共参数校验（两个后端共用），返回 (nonce, counter, nbytes, rfc7539, blocks)。"""
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
    return nonce, counter, nbytes, bool(rfc7539), blocks


def _chacha_words32(byte_mat):
    """[B,4k] uint8 → [B,k] uint32（每 4 字节小端合成 1 个字）。"""
    b = byte_mat.astype(np.uint32).reshape(byte_mat.shape[0], -1, 4)
    return b[:, :, 0] | (b[:, :, 1] << np.uint32(8)) | (b[:, :, 2] << np.uint32(16)) \
        | (b[:, :, 3] << np.uint32(24))


def _rotl32_np(x, n):
    """uint32 ndarray 循环左移（逻辑移位；uint32 运算天然回绕）。"""
    return (x << np.uint32(n)) | (x >> np.uint32(32 - n))


def _rotl32_t(x, n):
    """torch int32 循环左移：算术右移后用掩码消掉符号扩展位。

    注意掩码是 (1<<n)-1（右移 32-n 位后只保留低 n 位），不是 (1<<(32-n))-1。
    """
    return (x << n) | ((x >> (32 - n)) & ((1 << n) - 1))


def _chacha20_ks_numpy(keys, nonce, counter, nbytes, rfc7539):
    """numpy 后端：uint32 状态 + 每行一个连续一维数组 + 所有块一次性并行。"""
    B = keys.shape[0]
    blocks = (nbytes + 63) // 64
    if blocks == 0:
        return np.zeros((B, 0), dtype=np.uint8)
    N = B * blocks
    kw = _chacha_words32(keys)                     # [B,8] uint32
    blk_idx = np.tile(np.arange(blocks, dtype=np.uint64), B)
    nw = _chacha_words32(np.frombuffer(
        bytes(nonce), dtype=np.uint8).reshape(1, len(nonce)))
    st = [np.full(N, np.uint32(v), dtype=np.uint32) for v in _CHACHA_CONST]
    for i in range(8):
        st.append(np.repeat(kw[:, i], blocks).astype(np.uint32))
    if rfc7539:
        st.append(((np.uint64(counter) + blk_idx) & np.uint64(M32)).astype(np.uint32))
        for i in range(3):
            st.append(np.full(N, np.uint32(nw[0, i]), dtype=np.uint32))
    else:
        low = (np.uint64(counter) & np.uint64(M32)) + blk_idx
        st.append((low & np.uint64(M32)).astype(np.uint32))
        hi = ((np.uint64(counter) >> np.uint64(32)) + (low >> np.uint64(32))) & np.uint64(M32)
        st.append(hi.astype(np.uint32))
        for i in range(2):
            st.append(np.full(N, np.uint32(nw[0, i]), dtype=np.uint32))

    x = [a.copy() for a in st]

    def qr(a, b, c, d):
        xa, xb, xc, xd = x[a], x[b], x[c], x[d]
        xa += xb
        xd ^= xa; xd = _rotl32_np(xd, 16)
        xc += xd
        xb ^= xc; xb = _rotl32_np(xb, 12)
        xa += xb
        xd ^= xa; xd = _rotl32_np(xd, 8)
        xc += xd
        xb ^= xc; xb = _rotl32_np(xb, 7)
        x[a], x[b], x[c], x[d] = xa, xb, xc, xd

    for _ in range(10):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    words = np.stack([x[i] + st[i] for i in range(16)], axis=1)   # [N,16] uint32
    raw = words.view(np.uint8).reshape(N, 64)                      # 小端字节序
    return raw.reshape(B, blocks * 64)[:, :nbytes]


def _chacha20_ks_torch(keys_t, nonce, counter, nbytes, rfc7539):
    """torch 后端：int32 状态 + 连续一维字数组 + 所有块一次性并行。

    返回同 device 的 uint8 张量（GPU 路径不落 CPU）：`chacha --brute` 直接拿它
    在 GPU 上与目标比较，避免 keystream 回传再比较的往返。
    """
    T = G.torch
    dev = keys_t.device
    B = int(keys_t.shape[0])
    blocks = (nbytes + 63) // 64
    if blocks == 0:
        return T.zeros((B, 0), dtype=T.uint8, device=dev)
    N = B * blocks
    kw = keys_t.reshape(B, 8, 4).to(T.int32)        # [B,8,4]
    kw = kw[..., 0] | (kw[..., 1] << 8) | (kw[..., 2] << 16) | (kw[..., 3] << 24)
    blk_idx = T.arange(blocks, dtype=T.int64, device=dev).repeat(B)
    nonce_b = bytes(nonce)
    nw = [int.from_bytes(nonce_b[4 * i:4 * i + 4], 'little')
          for i in range(len(nonce_b) // 4)]

    def _i32(v):
        v &= M32
        return v - (1 << 32) if v >= (1 << 31) else v

    st = [T.full((N,), _i32(v), dtype=T.int32, device=dev) for v in _CHACHA_CONST]
    for i in range(8):
        st.append(kw[:, i].repeat_interleave(blocks))
    if rfc7539:
        ctr = (int(counter) + blk_idx) & M32
        st.append(ctr.to(T.int32))
        for i in range(3):
            st.append(T.full((N,), _i32(nw[i]), dtype=T.int32, device=dev))
    else:
        low = (int(counter) & M32) + blk_idx
        st.append((low & M32).to(T.int32))
        hi = ((int(counter) >> 32) + (low >> 32)) & M32
        st.append(hi.to(T.int32))
        for i in range(2):
            st.append(T.full((N,), _i32(nw[i]), dtype=T.int32, device=dev))

    x = [a.clone() for a in st]

    def qr(a, b, c, d):
        xa, xb, xc, xd = x[a], x[b], x[c], x[d]
        xa = xa + xb
        xd = _rotl32_t(xd ^ xa, 16)
        xc = xc + xd
        xb = _rotl32_t(xb ^ xc, 12)
        xa = xa + xb
        xd = _rotl32_t(xd ^ xa, 8)
        xc = xc + xd
        xb = _rotl32_t(xb ^ xc, 7)
        x[a], x[b], x[c], x[d] = xa, xb, xc, xd

    for _ in range(10):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    words = T.stack([x[i] + st[i] for i in range(16)], dim=1)      # [N,16] int32
    raw = words.view(T.uint8).reshape(N, 64)                       # CUDA 小端
    return raw.reshape(B, blocks * 64)[:, :nbytes]


def chacha20_keystream(keys, nonce, counter=0, nbytes=64, rfc7539=None):
    """
    批量生成 ChaCha20 密钥流。

    keys   : [B,32] uint8 张量/ndarray（或 32 字节 bytes，单密钥）
    nonce  : 12 字节（rfc7539）/ 8 字节（原版）
    counter: 起始计数器（rfc7539 为 32 位；原版为 64 位）
    rfc7539: None = 按 nonce 长度自动判断（12B→True，8B→False）
    nbytes : 要生成的字节数（允许 0）
    返回   : [B, nbytes] uint8 (numpy)

    实现：uint32/int32 状态（内存减半）+ 所有块拼成 [B*blocks, 16] 一次跑完；
    torch 张量输入时全程留在原 device，不再绕 numpy。
    """
    nonce, counter, nbytes, rfc7539, blocks = _chacha_parse(nonce, counter, nbytes, rfc7539)
    if isinstance(keys, (bytes, bytearray)):
        kb = np.frombuffer(bytes(keys), dtype=np.uint8)
        if kb.size != 32:
            raise ValueError(f'ChaCha20 密钥必须 32 字节，收到 {kb.size} 字节')
        keys = kb.reshape(1, 32)
    if _is_torch():
        T = G.torch
        if isinstance(keys, np.ndarray):
            keys_t = T.from_numpy(np.ascontiguousarray(keys, dtype=np.uint8)).to(_dev())
        elif T.is_tensor(keys):
            keys_t = keys.detach()
            if keys_t.device != _dev():
                keys_t = keys_t.to(_dev())
            if keys_t.dtype != T.uint8:
                keys_t = keys_t.to(T.uint8)
        else:
            keys_t = T.tensor(np.ascontiguousarray(keys, dtype=np.uint8), device=_dev())
        if keys_t.dim() != 2 or int(keys_t.shape[1]) != 32:
            raise ValueError(f'keys 形状应为 [B,32]，收到 {tuple(keys_t.shape)}')
        if nbytes == 0:
            return np.zeros((int(keys_t.shape[0]), 0), dtype=np.uint8)
        return _chacha20_ks_torch(keys_t, nonce, counter, nbytes, rfc7539).cpu().numpy()
    k_np = np.asarray(keys)
    if k_np.ndim != 2 or k_np.shape[1] != 32:
        raise ValueError(f'keys 形状应为 [B,32]，收到 {k_np.shape}')
    return _chacha20_ks_numpy(np.ascontiguousarray(k_np, dtype=np.uint8),
                              nonce, counter, nbytes, rfc7539)


def _chacha_brute_batch(keys, nonce, counter, target, rfc7539):
    """一批候选密钥的 keystream vs 目标比较，返回批内命中事件列表。

    keys 为 torch 张量时全流程留在 GPU（生成→密钥流→比较→只回传命中的 key）；
    为 numpy 数组时走 numpy 后端。命中事件的 index 是**批内下标**，由调用方加偏移。
    """
    target_b = bytes(target)
    if G.torch is not None and G.torch.is_tensor(keys):
        T = G.torch
        ks = _chacha20_ks_torch(keys, nonce, counter, len(target_b), rfc7539)
        tgt = T.from_numpy(np.frombuffer(target_b, dtype=np.uint8).copy()).to(keys.device)
        eq = (ks == tgt).all(dim=1)
        idx = T.nonzero(eq, as_tuple=False).flatten()
        if int(idx.numel()) == 0:
            return []
        ic = idx.cpu().tolist()
        kc = keys[idx].cpu().numpy()
        return [{'event': 'hit', 'kind': 'chacha', 'index': int(i),
                 'key': kc[j].tobytes().hex()} for j, i in enumerate(ic)]
    keys_np = keys if isinstance(keys, np.ndarray) else np.asarray(keys, dtype=np.uint8)
    ks = chacha20_keystream(keys_np, nonce, counter, len(target_b), rfc7539)
    eq = (ks == np.frombuffer(target_b, dtype=np.uint8)).all(1)
    return [{'event': 'hit', 'kind': 'chacha', 'index': int(i),
             'key': keys_np[i].tobytes().hex()} for i in np.nonzero(eq)[0]]


def chacha_brute(nonce, counter, target, tries, rfc7539, chunk=1 << 14, checkpoint='',
                 reset=False, progress=True, seed=1):
    """ChaCha20 随机密钥搜索演示（不保证可复现，仅演示吞吐）。

    按块调度（ctf_common.chunked）：--chunk 为每块候选数，--checkpoint 可续跑；
    进度/ETA 只写 stderr（--quiet / --no-progress 可关），不污染 stdout 结果流。
    torch-CUDA 时随机密钥在显存里生成、keystream 与比较全在 GPU，只回传命中项。
    返回 (done, hits, seconds)，hits 为 hit 事件 dict（index/key）。
    """
    tries, chunk = int(tries), int(chunk)
    if chunk < 1:
        raise ValueError(f'--chunk 必须 >= 1（chunk={chunk}）')
    if tries < 1:
        raise ValueError(f'--tries 必须 >= 1（tries={tries}）')
    target_b = bytes(target)
    if _is_gpu():
        T = G.torch
        dev = _dev()
        gen = T.Generator(device=dev)
        gen.manual_seed(int(seed))

        def fn(lo, hi):
            n = hi - lo
            keys = T.randint(0, 256, (n, 32), generator=gen, device=dev,
                             dtype=T.uint8)
            got = _chacha_brute_batch(keys, nonce, counter, target_b, rfc7539)
            for h in got:
                h['index'] = int(lo) + h['index']
            return n, got
    else:
        rng = np.random.default_rng(int(seed))

        def fn(lo, hi):
            n = hi - lo
            keys = rng.integers(0, 256, (n, 32), dtype=np.uint8)
            got = _chacha_brute_batch(keys, nonce, counter, target_b, rfc7539)
            for h in got:
                h['index'] = int(lo) + h['index']
            return n, got

    def on_hit(got):
        return True                                    # 命中即早停

    done, hits, dt = C.chunked(fn, tries, 0, chunk=chunk, label='chacha',
                               checkpoint=checkpoint, reset=reset,
                               progress=progress, on_hit=on_hit,
                               # meta 指纹：不同 nonce/counter/目标长度/变体的任务不串用断点
                               ckpt_meta={'cmd': 'chacha', 'nonce': bytes(nonce).hex(),
                                          'counter': int(counter),
                                          'target_len': int(len(target_b)),
                                          'rfc7539': bool(rfc7539),
                                          'target_sha': hashlib.sha256(
                                              target_b).hexdigest()[:16]})
    # 断点文件可能来自别的命令/旧格式：只认本命令的 dict 事件
    hits = [h for h in hits if isinstance(h, dict) and h.get('kind') == 'chacha']
    return done, hits, dt


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


def _xor_split(g, max_len):
    """全局候选序号 → (密钥长度, 该长度起始全局序号, 长度内密钥序号)。"""
    base = 0
    for klen in range(1, int(max_len) + 1):
        total = 256 ** klen
        if g < base + total:
            return klen, base, g - base
        base += total
    raise ValueError(f'XOR 候选序号越界: {g}')


def _xor_keys(klen, first, count):
    """按大端密钥序号向量化生成 [count, klen] uint8 候选密钥。"""
    keys = np.arange(int(first), int(first) + int(count), dtype=np.uint64)
    return np.stack([(keys >> np.uint64(8 * (klen - 1 - i))) & np.uint64(0xFF)
                     for i in range(klen)], axis=1).astype(np.uint8)


# 小写映射表（GPU 打分用）：ASCII 大写 → 小写，其余不变
_LOW256 = np.arange(256, dtype=np.uint8)
_LOW256[(_LOW256 >= 0x41) & (_LOW256 <= 0x5A)] += np.uint8(32)
_FLAG_SEQS = tuple(w + b'{' for w in _FLAG_WORDS)
_XOR_TIE_BASE = 1 << 25          # > 最大候选批大小，用于把「分数 + 下标」编码成单键


def _xor_select(sc, k):
    """批内 top-k：**分数降序、批内候选下标降序**（CPU/GPU 共用的确定性并列规则）。

    旧实现 `np.argsort(sc)[-k:][::-1]` 在并列分数时的次序取决于 numpy 排序实现，
    CPU 与 GPU 无法逐项对拍。这里改成显式确定性规则：把 (score, index) 编码为
    唯一 int64 键，取最小的 k 个，因此 CPU/GPU/不同 chunk 下结果完全一致。
    """
    cnt = int(sc.size)
    tie = -(sc.astype(np.int64) * _XOR_TIE_BASE) - np.arange(cnt, dtype=np.int64)
    if k < cnt:
        idx = np.argpartition(tie, k - 1)[:k]
        return idx[np.argsort(tie[idx])]
    return np.argsort(tie)


def _xor_oom(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return 'outofmemory' in name or 'out of memory' in text or \
        ('cuda error' in text and 'memory' in text)


class _XorScanner:
    """重复 XOR 候选空间的确定性分块扫描器（numpy / CPU 路径）。

    候选按「密钥长度段 + 段内固定批次」切批（批次大小只由内存决定，与 --chunk 无关），
    跨块边界的不完整批次留到下一次 fn 调用凑满再打分。因此：
      * 同一输入/参数下结果与 --chunk / --checkpoint 粒度无关（含并列分数的次序）；
      * --chunk 只影响进度/断点落盘频率，不会拖慢内核（不会把批次切碎）。
    GPU 子类只替换 `_process`：用更大的 wave 一次算多个批次，但**每批仍按同一规则
    各自选 top-k**，所以 CPU/GPU 输出逐项一致。
    """

    def __init__(self, d, n, max_len, top, end=None):
        self.d = d
        self.n = int(n)
        self.max_len = int(max_len)
        self.top = int(top)
        self.end = int(end) if end is not None else None
        self.vec = max(1, 4_000_000 // max(1, self.n))   # 单批 4M 字节单元
        self.pending = None                              # 不完整批次 (klen, base, s, e)
        self.pending_batches = []                        # 已凑满、待打分的批次
        self.wave = 1                                    # 一次 _process 最多几个批次

    def _batch_end(self, pos):
        """返回 pos 所在批次的 (klen, base, b_start, b_end)。"""
        klen, base, _ = _xor_split(pos, self.max_len)
        b_start = base + ((pos - base) // self.vec) * self.vec
        return klen, base, b_start, min(b_start + self.vec, base + 256 ** klen)

    def _score_batch(self, klen, base, s, e):
        cnt = e - s
        kb = _xor_keys(klen, s - base, cnt)
        ks = np.tile(kb, (1, (self.n + klen - 1) // klen))[:, :self.n]
        pt = self.d[None, :] ^ ks
        sc = _score_plaintext(pt)
        idx = _xor_select(sc, min(self.top, cnt))
        return [{'event': 'hit', 'kind': 'xorbrute', 'score': int(sc[i]),
                 'key': kb[i].tobytes().hex(),
                 'plaintext': pt[i].tobytes().hex()} for i in idx]

    def _process(self, batches):
        out = []
        for b in batches:
            out += self._score_batch(*b)
        return out

    def fn(self, lo, hi):
        lo, hi = int(lo), int(hi)
        pos = lo
        if self.pending is not None:
            klen, base, s, e = self.pending
            if hi < e:                       # 批次还没凑满：等下一个块
                return hi - lo, []
            self.pending_batches.append((klen, base, s, e))
            self.pending = None
            pos = e
        else:
            # 断点续跑可能落在批次中间：该批上次未完成（无 hits），从批起点重扫
            _, _, b_start, _ = self._batch_end(pos)
            if b_start < pos:
                pos = b_start
        while pos < hi:
            klen, base, b_start, b_end = self._batch_end(pos)
            if b_end <= hi:
                self.pending_batches.append((klen, base, pos, b_end))
                pos = b_end
            else:                            # 余下的是不完整批次
                self.pending = (klen, base, pos, b_end)
                break
        if not self.pending_batches:
            return hi - lo, []
        flush = (self.end is not None and hi >= self.end) or len(self.pending_batches) >= self.wave
        if not flush:
            return hi - lo, []
        batches = self.pending_batches
        self.pending_batches = []
        try:
            got = self._process(batches)
        except Exception:
            self.pending = None              # 让重试从批边界重新开始
            raise
        return hi - lo, got


class _XorScannerGPU(_XorScanner):
    """torch-CUDA 扫描器：候选键生成 / 明文异或 / 整数打分 / top-k 全在 GPU。

    与 CPU 路径共用同一批次划分与 `_xor_select` 并列规则；每 wave 把若干连续批次
    拼成一个大张量一次算完（提升 GPU 占用率），再按批次切片各自选 top-k。
    """

    def __init__(self, d, n, max_len, top, end=None):
        super().__init__(d, n, max_len, top, end)
        T = G.torch
        self._T = T
        self._dev = _dev()
        self._d_t = T.from_numpy(d.copy()).to(self._dev)
        w8 = _W256.astype(np.int8)
        self._w8g = T.from_numpy(np.ascontiguousarray(w8)).to(self._dev)
        self._lowg = T.from_numpy(np.ascontiguousarray(_LOW256)).to(self._dev)
        self._seqs = [T.from_numpy(np.frombuffer(s, dtype=np.uint8).copy()).to(self._dev)
                      for s in _FLAG_SEQS]
        self._pat = {k: T.tensor([i % k for i in range(self.n)],
                                 dtype=T.int64, device=self._dev)
                     for k in range(1, self.max_len + 1)}
        self.wave = self._pick_wave()

    def _pick_wave(self):
        T = self._T
        try:
            free, _total = T.cuda.mem_get_info()
        except Exception:
            free = 1 << 30
        cells = max(1, self.vec * self.n)                  # 单批元素数
        # 每元素峰值约 12 字节（int32 索引 + 查找表 + 若干 bool 临时量）
        budget_cells = max(1, int(free * 0.25 / 12))
        return max(1, min(8, budget_cells // cells))

    def _keys_gpu(self, klen, first, count):
        T = self._T
        idx = T.arange(int(first), int(first) + int(count), dtype=T.int64,
                       device=self._dev)
        return T.stack([((idx >> (8 * (klen - 1 - i))) & 0xFF).to(T.uint8)
                        for i in range(klen)], dim=1)

    def _score_gpu(self, pt):
        T = self._T
        cnt, L = int(pt.shape[0]), int(pt.shape[1])
        flat = pt.reshape(-1).to(T.int32)
        w = self._w8g.index_select(0, flat).view(cnt, L)
        sc = w.sum(dim=1, dtype=T.int32)
        low = self._lowg.index_select(0, flat).view(cnt, L)
        printable = ((pt >= 0x20) & (pt <= 0x7E)).all(dim=1)
        has_close = (pt == 0x7D).any(dim=1)
        flag = T.zeros(cnt, dtype=T.bool, device=self._dev)
        if L >= 4:
            for seq in self._seqs:
                k = int(seq.numel())
                if k > L:
                    continue
                hit = low[:, 0:L - k + 1] == seq[0]
                for j in range(1, k):
                    hit &= (low[:, j:L - k + 1 + j] == seq[j])
                flag |= hit.any(dim=1)
            wordc = ((low >= 0x61) & (low <= 0x7A)) | (low == 0x5F)
            generic = (wordc[:, :-3] & wordc[:, 1:-2] & wordc[:, 2:-1]
                       & (low[:, 3:] == 0x7B)).any(dim=1)
        else:
            generic = T.zeros(cnt, dtype=T.bool, device=self._dev)
        sc = sc + flag.to(T.int32) * 80
        sc = sc + (printable & generic & has_close).to(T.int32) * 20
        sc = sc + (printable & has_close).to(T.int32) * 5
        return sc

    def _score_wave(self, group):
        try:
            return self._score_wave_impl(group)
        except Exception as exc:
            if _xor_oom(exc):
                try:
                    self._T.cuda.empty_cache()
                except Exception:
                    pass
                if len(group) > 1:
                    mid = len(group) // 2
                    return self._score_wave(group[:mid]) + self._score_wave(group[mid:])
            raise

    def _score_wave_impl(self, group):
        T = self._T
        klen = group[0][0]
        s0 = group[0][2]
        total = group[-1][3] - s0
        kb = self._keys_gpu(klen, s0 - group[0][1], total)      # [total,klen] uint8
        ks = kb.index_select(1, self._pat[klen])                # [total,n] uint8
        pt = self._d_t.unsqueeze(0) ^ ks
        sc = self._score_gpu(pt)
        out = []
        off = 0
        dev = self._dev
        for (kl, base, s, e) in group:
            cnt = e - s
            sub = sc[off:off + cnt]
            k = min(self.top, cnt)
            ar = T.arange(cnt, dtype=T.int64, device=dev)
            tie = -(sub.to(T.int64) * _XOR_TIE_BASE) - ar
            if k < cnt:
                sel = T.topk(tie, k, largest=False, sorted=True).indices
            else:
                sel = T.argsort(tie)
            sel_np = sel.cpu().numpy()
            sc_np = sub.index_select(0, sel).cpu().numpy()
            pt_rows = pt[off:off + cnt].index_select(0, sel).cpu().numpy()
            koff = s - base
            for t, i in enumerate(sel_np):
                ki = koff + int(i)
                out.append({'event': 'hit', 'kind': 'xorbrute',
                            'score': int(sc_np[t]),
                            'key': ki.to_bytes(klen, 'big').hex(),
                            'plaintext': pt_rows[t].tobytes().hex()})
            off += cnt
        return out

    def _process(self, batches):
        out = []
        i = 0
        nb = len(batches)
        while i < nb:
            klen = batches[i][0]
            j = i + 1
            while (j < nb and batches[j][0] == klen and (j - i) < self.wave
                   and batches[j][2] == batches[j - 1][3]):
                j += 1
            out += self._score_wave(batches[i:j])
            i = j
        return out


def _xorbrute_scan(data, max_len=3, top=10, chunk=1 << 14, checkpoint='',
                   reset=False, progress=True):
    """重复 XOR 爆破（分块 + 断点续跑）；返回 (结果列表, 处理候选数, 秒数)。

    候选空间按 1..max_len 字节密钥顺序编号为一个全局序列，因此断点文件里的 next
    就是全局候选序号：中断后续跑不会重复扫描，也不会漏掉后面的密钥长度。
    有 GPU 时自动走 `_XorScannerGPU`（打分/top-k 全在显存），否则走 numpy 路径；
    两条路径共用同一批次划分与并列规则，结果逐项一致。
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
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError(f'--chunk 必须 >= 1（chunk={chunk}）')
    total = sum(256 ** k for k in range(1, max_len + 1))
    cls = _XorScannerGPU if _is_gpu() else _XorScanner
    scanner = cls(d, n, max_len, top, end=total)
    if checkpoint:
        scanner.wave = 1                 # 断点模式下每块立刻出 hits，避免跨块丢结果
    done, hits, dt = C.chunked(
        scanner.fn, total, 0, chunk=chunk, label='xorbrute',
        checkpoint=checkpoint, reset=reset, progress=progress,
        # meta 指纹：不同密文/密钥长度上限的任务不串用断点
        ckpt_meta={'cmd': 'xorbrute', 'len': n, 'max_len': max_len,
                   'data_sha': hashlib.sha256(d.tobytes()).hexdigest()[:16]})
    hits = [h for h in hits if isinstance(h, dict) and h.get('kind') == 'xorbrute']
    hits.sort(key=lambda h: -int(h['score']))
    seen, out = set(), []                            # 按明文去重（与旧行为一致）
    for h in hits:
        pt = bytes.fromhex(h['plaintext'])
        if pt in seen:
            continue
        seen.add(pt)
        out.append((int(h['score']), h['key'], pt))
    return out[:top], done, dt


def xorbrute(data, max_len=3, top=10):
    """
    重复 XOR 爆破。data: bytes。返回 [(score, key_hex, plaintext), ...]
    max_len 只支持 1~3：3 字节 = 1677 万候选（几秒），4 字节 = 43 亿，不可行。
    （分块/断点等调度参数由 _xorbrute_scan 提供，子命令走它。）
    """
    res, _, _ = _xorbrute_scan(data, max_len=max_len, top=top, progress=False)
    return res


# =====================================================================
# 6. AES-192 CTR 密钥流
# =====================================================================
def aesctr_keystream(key, nonce, nbytes=64, counter=0, nonce_mode='le'):
    """AES-128/192/256-CTR 密钥流。

    计数器块构造（按 nonce 长度自动选择）：
      nonce 8 字节  → nonce || counter(64bit)
      nonce 12 字节 → nonce || counter(32bit)      （GCM 风格）
      nonce 16 字节 → 整个块就是计数器，counter 直接加到块上
    nonce_mode='le'|'be' 控制计数器字节序（8/12 字节模式下生效）。
    """
    key, nonce = bytes(key), bytes(nonce)
    if len(key) not in (16, 24, 32):
        raise ValueError(f'AES 密钥必须 16/24/32 字节，收到 {len(key)} 字节')
    if len(nonce) not in (8, 12, 16):
        raise ValueError(f'nonce 必须 8/12/16 字节，收到 {len(nonce)} 字节')
    if nonce_mode not in ('le', 'be'):
        raise ValueError(f"nonce_mode 只能是 'le' 或 'be'，收到 {nonce_mode!r}")
    nbytes, counter = int(nbytes), int(counter)
    if nbytes < 0:
        raise ValueError(f'nbytes 不能为负: {nbytes}')
    blocks = (nbytes + 15) // 16
    order = 'little' if nonce_mode == 'le' else 'big'
    suffix = 16 - len(nonce)
    limit = 1 << (8 * suffix) if suffix else (1 << 128)
    if counter < 0 or counter + max(0, blocks - 1) >= limit:
        raise ValueError(f'AES-CTR 计数器越界（counter={counter}, blocks={blocks}）')
    if blocks == 0:
        return b''
    T = _T(); is_t = _is_torch()
    if suffix == 0:
        base = int.from_bytes(nonce, order)
        ctrs = [((base + counter + i) % limit).to_bytes(16, order) for i in range(blocks)]
    else:
        ctrs = [nonce + (counter + i).to_bytes(suffix, order) for i in range(blocks)]
    blk = np.frombuffer(b''.join(ctrs), dtype=np.uint8).reshape(blocks, 16)
    blk_t = T.tensor(blk, dtype=T.uint8, device=_dev()) if is_t else blk
    key_t = T.tensor(list(key), dtype=T.uint8, device=_dev()) if is_t else np.frombuffer(key, dtype=np.uint8)
    out = G.aes_ecb(blk_t, key_t)
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

    # --- AES-CTR nonce 模式（12B GCM 风格 / 16B 整块计数器；le 是自定义语义）---
    # 期望值由独立实现生成：pycryptodome AES-ECB 逐块加密 + 显式拼装计数器块
    k24 = bytes(range(24))
    n12, n16 = bytes(range(12)), bytes(range(16))
    ok_nm = (aesctr_keystream(k24, n12, 48, 5, 'le').hex() ==
             '1b084ee09a83ecc1ebf4b2b3076abecf469cf1b0ed586b8b55b9ddde6971823d'
             '402f743b7bb003a157da9557d2773ed2' and
             aesctr_keystream(k24, n12, 48, 5, 'be').hex() ==
             '7edada648641ceea4224d7afd8ae7607b69c09734999b440da3bb8fdf348d748'
             '54c85ba1fe4582fecf898fe450222722' and
             aesctr_keystream(k24, n16, 48, 5, 'le').hex() ==
             '76f4db719a4144bee9a95172c17ac8e7ceb6f7c10531b64c29042960e794f3a3'
             '9e83f0c83f2d50b802dbf891f2432a9b' and
             aesctr_keystream(k24, n16, 48, 5, 'be').hex() ==
             '152b8051a7ee48817db06421e9fbb99ed11579699aa331bd8ba7e2eadab94344'
             '5a308e8a142bf9168728bb76d87a085a')
    check('[AES-CTR nonce 模式]', ok_nm)

    # --- 输入容错：带 UTF-8 BOM 的 keys-file（Windows 记事本导出）---
    import contextlib
    import io
    import json as _json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        kp = os.path.join(td, 'keys_bom.txt')
        with open(kp, 'wb') as fh:
            fh.write(b'\xef\xbb\xbf' + bytes(range(32)).hex().encode() + b'\r\n')
        try:
            kb = _load_keys_file(kp, 32)
            ok_bom = kb.shape == (1, 32) and kb[0].tobytes() == bytes(range(32))
        except ValueError:
            ok_bom = False
    check('[keys-file BOM 容错]', ok_bom)

    # --- --json 契约：stdout 只有单行 JSON 事件（用 crc32 冒烟）---
    saved_out = (C.json_mode(), C.quiet())
    ok_json = False
    buf = io.StringIO()
    try:
        C.set_output(json_mode=True, quiet=True)
        with contextlib.redirect_stdout(buf):
            cmd_crc32(argparse.Namespace(data_hex='313233343536373839', target_hex=None))
        ev = _json.loads(buf.getvalue().strip())
        ok_json = (ev.get('event') == 'result' and ev.get('kind') == 'crc32'
                   and ev.get('crc') == 'cbf43926')
    except Exception:
        ok_json = False
    finally:
        C.set_output(json_mode=saved_out[0], quiet=saved_out[1])
    check('[JSON 输出契约]', ok_json)

    # --- 断点续跑（--checkpoint）：全局候选序号，中断后不重扫不漏扫 ---
    ok_ck = True
    with tempfile.TemporaryDirectory() as td:
        try:
            errbuf = io.StringIO()
            with contextlib.redirect_stderr(errbuf):      # 续跑提示走 stderr，别刷屏
                ck = os.path.join(td, 'ck_chacha.json')
                far = bytes([0xFF] * 8)      # 4 个随机密钥几乎不可能命中 8B 目标
                d1, _, _ = chacha_brute(bytes(8), 0, far, 4, False, chunk=2,
                                        checkpoint=ck, progress=False)
                with open(ck, encoding='utf-8') as fh:
                    nxt = int(_json.load(fh).get('next', -1))
                d2, _, _ = chacha_brute(bytes(8), 0, far, 4, False, chunk=2,
                                        checkpoint=ck, progress=False)
                with open(ck, 'w', encoding='utf-8') as fh:   # 模拟中途中断在 2/4
                    _json.dump({'next': 2, 'hits': []}, fh)
                d3, _, _ = chacha_brute(bytes(8), 0, far, 4, False, chunk=2,
                                        checkpoint=ck, progress=False)
                # 同一断点文件换目标：meta 指纹不一致 -> 告警并从头重扫（不能直接续到 end）
                d4, _, _ = chacha_brute(bytes(8), 0, bytes([0xFE] * 8), 4, False, chunk=2,
                                        checkpoint=ck, progress=False)
                ckx = os.path.join(td, 'ck_xor.json')
                r1, x1, _ = _xorbrute_scan(b'abc', 1, 3, chunk=64, checkpoint=ckx,
                                           progress=False)
                with open(ckx, 'w', encoding='utf-8') as fh:
                    _json.dump({'next': 128, 'hits': []}, fh)
                r2, x2, _ = _xorbrute_scan(b'abc', 1, 3, chunk=64, checkpoint=ckx,
                                           progress=False)
                _, x3, _ = _xorbrute_scan(b'abd', 1, 3, chunk=64, checkpoint=ckx,
                                          progress=False)     # 换密文 -> 指纹不一致
            ok_ck = (nxt == 4 and d1 == 4 and d2 == 0 and d3 == 2 and d4 == 4
                     and x1 == 256 and x2 == 128 and x3 == 256 and r2 == r1)
            # 断点若落在批次中间，该批上次未完成 → 从头重扫整批：r2 必须与完整扫描 r1 逐项一致
        except Exception:
            ok_ck = False
    check('[断点续跑]', ok_ck)

    # --- ChaCha20 多块并行：一次算完 [B*blocks,16]；独立期望值来自 pycryptodome ---
    ks_mb_rfc = bytes(chacha20_keystream(
        bytes(range(32)), bytes.fromhex('000000090000004a00000000'), 1, 192)[0])
    ks_mb_orig = bytes(chacha20_keystream(
        bytes(range(32)), bytes(8), 0x12345678, 192, False)[0])
    ks_mb_edge = bytes(chacha20_keystream(
        bytes(range(32)), bytes.fromhex('000000090000004a00000000'), 0xFFFFFFF0, 200)[0])
    ok_mb = (
        ks_mb_rfc.hex() ==
        '10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e'
        'd2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e'
        '0a88837739d7bf4ef8ccacb0ea2bb9d69d56c394aa351dfda5bf459f0a2e9fe8'
        'e721f89255f9c486bf21679c683d4f9c5cf2fa27865526005b06ca374c86af3bd'
        'cbfbdcb83be65862ed5c20eae5a43241d6a92da6dca9a156be25297f51c27188'
        'a861e93cc3aeb129a76598baccd27453ac6941b4b4e1e5153a9fee95d1ba00e' and
        ks_mb_orig.hex() ==
        '75453f6602f673f4ff0449d7afd8928b59d0dd3ced361409e11f16dd82ae81d2'
        '8c9839483395fcaa3ffab58e26a1c1a44d20d5465794ab6abb7ba6c44a387ab3'
        '6647b996f50d5576a16f05578789b7bb474a8378683be8341981d0e142f1fc20'
        '91e78120b490ff281dbec457d84317c7b345f8776a0a9e4b00924c009f80f84c'
        '6c11a9f2e49b4525aa47aa960d9b3c006bf6745df81c50eb1de93fc6335ae9e8'
        '6869d95b4613dd7d090e60974b92308028167ddff76118dd48da707d432c353a' and
        ks_mb_edge.hex() ==
        '38fa98b60c9da1221eaf7b270e2e49ee1f0f354c7b23e825cd433bbc8913ee56'
        '51fead1f95f2cd054c9cd3d6b5a958847322b6d286f7d32e7f90c84e8f5a2d07'
        'aa355697297cd8fd5f72ba27d1d314cfb5fbc23689527be5afe46cc2f54eb265'
        '8972003ca549685923c263f61149a2f580622dae6ba617e7c23018bb420085e1'
        '8778122a75160d743d320ebe7fe966e9a82eef3d5bcb59e79a0cd9236b53e3aa'
        '5f5575892d8050d2df7692d672441988b511db4f5ec345535d86b2ec7224cbfa'
        '9c4f11fead68ddcc')
    check('[ChaCha20 多块并行]', ok_mb)

    # --- GPU 内核 vs numpy 内核逐字节对拍（无 GPU 时跳过）---
    ok_kg = True
    kg_note = ' (无 GPU，跳过)'
    if _is_gpu():
        rnd2 = np.random.default_rng(20261110)
        keys_np = rnd2.integers(0, 256, (64, 32), dtype=np.uint8)
        for nl_, rfc_, nb_ in ((12, True, 64), (12, True, 200),
                               (8, False, 64), (8, False, 200)):
            nce = bytes(rnd2.integers(0, 256, nl_, dtype=np.uint8))
            ref = _chacha20_ks_numpy(keys_np, nce, 7, nb_, rfc_)
            keys_t = G.torch.from_numpy(keys_np.copy()).to(_dev())
            got = _chacha20_ks_torch(keys_t, nce, 7, nb_, rfc_).cpu().numpy()
            ok_kg = ok_kg and np.array_equal(ref, got)
        kg_note = ''
    check('[ChaCha20 GPU 内核]', ok_kg, kg_note)

    # --- chacha --brute GPU 命中路径：GPU 生成 keys → GPU keystream → GPU 比较 ---
    ok_bh = True
    bh_note = ' (无 GPU，跳过)'
    if _is_gpu():
        nonce_b = bytes.fromhex('000000090000004a00000000')
        k_np = np.frombuffer(bytes(range(32)), dtype=np.uint8).reshape(1, 32)
        tgt = bytes(chacha20_keystream(bytes(range(32)), nonce_b, 3, 64, True)[0])
        k_t = G.torch.from_numpy(k_np.copy()).to(_dev())
        got_hits = _chacha_brute_batch(k_t, nonce_b, 3, tgt, True)
        miss = _chacha_brute_batch(k_t, nonce_b, 4, tgt, True)
        ok_bh = (len(got_hits) == 1 and got_hits[0]['index'] == 0
                 and got_hits[0]['key'] == bytes(range(32)).hex() and not miss)
        bh_note = ''
    check('[chacha brute GPU 命中]', ok_bh, bh_note)

    # --- xorbrute 并列次序（确定性规则）+ GPU/CPU 结果逐项一致 ---
    ok_tie = _xor_select(np.array([5, 5, 3, 5, 1, 5], dtype=np.int64), 3).tolist() == [5, 3, 1]
    check('[xorbrute 并列次序]', ok_tie)
    ok_par = True
    par_note = ' (无 GPU，跳过)'
    if _is_gpu():
        orig_backend = G.BACKEND
        rnd3 = np.random.default_rng(7)
        samples = [bytes(rnd3.integers(0, 256, 16, dtype=np.uint8)),
                   bytes(rnd3.integers(0, 256, 7, dtype=np.uint8)),
                   b'flag{xor_gpu_selftest}']
        try:
            G.init_backend('numpy')
            cpu_res = [xorbrute(c, max_len=2, top=8) for c in samples]
            G.init_backend('torch')
            gpu_res = [xorbrute(c, max_len=2, top=8) for c in samples]
            ok_par = cpu_res == gpu_res
        finally:
            G.init_backend('torch' if orig_backend.startswith('torch') else 'numpy')
        par_note = ''
    check('[xorbrute GPU==CPU]', ok_par, par_note)

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
    variant = 'rfc7539 12B nonce' if rfc7539 else '原版 8B nonce + 64bit 计数器'
    if a.keys_file:
        keys = _load_keys_file(a.keys_file, 32)
        if not C.json_mode():                     # 文本模式保留原起始行（结果流 stdout）
            print(f'[scan] {len(keys)} 个候选密钥 × {len(target)}B 目标 （{variant}）')
        t0 = time.time()
        out = chacha20_keystream(keys, nonce, a.counter, len(target), rfc7539)
        eq = (out == np.frombuffer(target, dtype=np.uint8)).all(1)
        dt = time.time() - t0
        hits = [int(i) for i in np.nonzero(eq)[0]]
        best = None
        if not hits:                              # 未命中：给出最接近的候选（比特一致度）
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
        rate = len(keys) / dt / 1e3 if dt else 0.0
        if C.json_mode():
            for i in hits:
                _json_hit('chacha', index=i, key=keys[i].tobytes().hex())
            payload = dict(mode='keys-file', candidates=len(keys), hit_count=len(hits),
                           elapsed=round(dt, 4), rate_kps=round(rate, 2))
            if best is not None:
                payload['best'] = {'bits': best[0], 'total_bits': len(target) * 8,
                                   'index': best[1]}
            _json_result('chacha', **payload)
        else:
            print(f'[done] {rate:.1f} K/s')
            for i in hits:
                print(f'  ★ 命中 #{i} key={keys[i].tobytes().hex()}')
            if best is not None:
                print(f'  未命中；最佳 {best[0]}/{len(target)*8} bit (候选 #{best[1]})')
    elif a.brute:
        if not C.json_mode():
            print('[brute] 随机密钥搜索（仅为演示吞吐；真实场景用 --keys-file 或直接生成候选）')
        chunk = a.chunk
        if chunk is None:                     # 默认：GPU 用大块（≥262144）吃满显存带宽
            chunk = 262144 if _is_gpu() else (1 << 14)
        done, hits, dt = chacha_brute(nonce, a.counter, target, a.tries, rfc7539,
                                      chunk=chunk, checkpoint=a.checkpoint,
                                      reset=a.reset_checkpoint,
                                      progress=not a.no_progress)
        found = bool(hits)
        rate = done / dt / 1e3 if dt else 0.0
        if C.json_mode():
            for h in hits:
                _json_hit('chacha', index=h.get('index'), key=h.get('key'))
            _json_result('chacha', mode='brute', tried=done, found=found,
                         hit_count=len(hits), elapsed=round(dt, 4),
                         rate_kps=round(rate, 2))
        else:
            for h in hits:
                print(f'  ★ 命中 #{h.get("index")} key={h.get("key")}')
            print(f'[done] {done:,} 键 / {dt:.2f}s = {rate:.1f} K/s  '
                  f'{"命中" if found else "未命中"}')
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
    otk_derived = False
    if a.nonce_hex is not None:
        if a.mode != 'aead':
            raise ValueError('--nonce-hex 只在 --mode aead 下使用')
        nonce = _parse_hex(a.nonce_hex, '--nonce-hex')
        if len(nonce) != 12:
            raise ValueError(f'AEAD nonce 需要 12 字节，收到 {len(nonce)} 字节')
        # 方便用法：给 ChaCha20 主密钥 + nonce，自动派生 Poly1305 子密钥（计数器 0 的前 32B）
        otk = bytes(chacha20_keystream(key, nonce, counter=0, nbytes=32)[0])
        otk_derived = True
        if not C.json_mode():
            print(f'[poly1305] 派生 otk = {otk.hex()}')
    if a.mode == 'aead':
        # aead 模式下 msg 按「密文」处理：tag 覆盖 pad(aad)||pad(ct)||len(aad)||len(ct)
        tag = poly1305_tag(otk, aad, msg)
    else:
        tag = poly1305(otk, msg)
    fields = dict(mode=a.mode, tag=tag.hex())
    if otk_derived:
        fields['otk'] = otk.hex()
    if a.expect_hex:
        want = _parse_hex(a.expect_hex, '--expect-hex')
        if len(want) != 16:
            raise ValueError(f'--expect-hex 应为 16 字节 tag，收到 {len(want)} 字节')
        ok = tag == want
        fields.update(expect=want.hex(), match=ok)
        if C.json_mode():
            _json_result('poly1305', **fields)
        else:
            print(f'[poly1305/{a.mode}] tag = {tag.hex()}')
            print(f'  期望 {want.hex()}  →  {"匹配 ✅" if ok else "不匹配 ❌"}')
        return 0 if ok else 1          # 校验命令：不匹配时退出码 1，脚本可感知
    if C.json_mode():
        _json_result('poly1305', **fields)
    else:
        print(f'[poly1305/{a.mode}] tag = {tag.hex()}')
    return tag


def cmd_aesctr(a):
    """AES-128/192/256-CTR 密钥流 / 加解密。"""
    key = _parse_hex(a.key_hex, '--key-hex')
    nonce = _parse_hex(a.nonce_hex, '--nonce-hex')
    if len(key) not in (16, 24, 32):
        raise ValueError(f'AES 密钥必须 16/24/32 字节，收到 {len(key)} 字节')
    if len(nonce) not in (8, 12, 16):
        raise ValueError(f'nonce 必须 8/12/16 字节，收到 {len(nonce)} 字节')
    if a.mode == 'ks':
        if a.length < 0:
            raise ValueError(f'--length 不能为负: {a.length}')
        ks = aesctr_keystream(key, nonce, a.length, a.counter, a.nonce_mode)
        if C.json_mode():
            _json_result('aesctr', mode='ks', length=a.length, keystream=ks.hex())
        else:
            print(f'[aesctr] keystream({a.length}B) = {ks.hex()}')
        return ks
    data = _parse_hex(a.data_hex, '--data-hex') if a.data_hex else b''
    ks = aesctr_keystream(key, nonce, len(data), a.counter, a.nonce_mode)
    out = bytes(x ^ y for x, y in zip(data, ks))
    if C.json_mode():
        fields = dict(mode=a.mode, length=len(out), data=out.hex())
        if a.out_text:
            fields['text'] = repr(out)
        _json_result('aesctr', **fields)
    else:
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
        if C.json_mode():
            _json_result('rs63', length=len(code), valid=True, order=hit[0],
                         fcr=hit[1], nsym=a.nsym)
        else:
            print(f'[RS(63,31)] 长度 {len(code)}  合法=True  '
                  f'(order={hit[0]} fcr={hit[1]} nsym={a.nsym})')
        return
    o0, f0 = orders[0], fcrs[0]
    _, sh = rs63_check(code, nsym=a.nsym, fcr=f0, order=o0, auto=False)
    if C.json_mode():
        _json_result('rs63', length=len(code), valid=False, syndrome=sh,
                     order=o0, fcr=f0, nsym=a.nsym)
    else:
        print(f'[RS(63,31)] 长度 {len(code)}  合法=False')
        print(f'  syndrome = {sh}  (order={o0} fcr={f0} nsym={a.nsym})')
    if not a.locate:
        return
    if not C.json_mode():
        print('  尝试单字节定位…')
    for o in orders:
        for f in fcrs:
            r = rs63_locate(code, nsym=a.nsym, fcr=f, order=o)
            if r:
                pos, delta = r
                if C.json_mode():
                    _json_locate('rs63', found=True, pos=pos, old=code[pos],
                                 new=code[pos] ^ delta, order=o, fcr=f)
                else:
                    print(f'  ★ 修正（order={o} fcr={f}）：pos={pos} '
                          f'{code[pos]:#04x} → {code[pos] ^ delta:#04x}')
                return
    if C.json_mode():
        _json_locate('rs63', found=False)
    else:
        print('  单字节无法修正（可能是合法码字或错更多字节）')


def cmd_crc32(a):
    data = _parse_hex(a.data_hex, '--data-hex')
    if a.target_hex is None:
        c = int(crc32_batch(np.frombuffer(data, dtype=np.uint8).reshape(1, -1))[0])
        if C.json_mode():
            _json_result('crc32', crc=f'{c:08x}')
        else:
            print(f'[crc32] {c:08x}')
        return
    target = _parse_hex(a.target_hex, '--target-hex')
    if not 1 <= len(target) <= 4:
        raise ValueError(f'--target-hex 需 1~4 字节的 CRC 值，收到 {len(target)} 字节')
    want = int.from_bytes(target, 'big')
    if not C.json_mode():
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
        tail = f'{int(i):02x}'
        if C.json_mode():
            _json_hit('crc32', data=data.hex(), tail=tail)
        else:
            print(f'  ★ 命中：{data.hex()}{tail}')
        nhit += 1
    st2 = tab[(st1[:, None] ^ b[None, :]) & 0xFF] ^ (st1[:, None] >> 8)
    crc2 = (st2 ^ 0xFFFFFFFF) & 0xFFFFFFFF
    for i, j in np.argwhere(crc2 == want):
        tail = f'{int(i):02x}{int(j):02x}'
        if C.json_mode():
            _json_hit('crc32', data=data.hex(), tail=tail)
        else:
            print(f'  ★ 命中：{data.hex()}{tail}')
        nhit += 1
    if C.json_mode():
        _json_result('crc32', mode='tail-brute', target=f'{want:08x}',
                     variants=256 + 65536, hit_count=nhit, found=nhit > 0)
    else:
        if not nhit:
            print('  未命中')
        print(f'  扫描 {256 + 65536:,} 变体完成')


def cmd_xorbrute(a):
    data = _parse_hex(a.data_hex, '--data-hex')
    if not data:
        raise ValueError('--data-hex 不能为空')
    t0 = time.time()
    chunk = a.chunk if a.chunk is not None else (1 << 14)
    res, done, dt = _xorbrute_scan(data, max_len=a.max_len, top=a.top,
                                   chunk=chunk, checkpoint=a.checkpoint,
                                   reset=a.reset_checkpoint,
                                   progress=not a.no_progress)
    if C.json_mode():
        for rank, (sc, kh, pt) in enumerate(res):
            _json_hit('xorbrute', rank=rank, score=sc, key=kh, plaintext=pt.hex())
        _json_result('xorbrute', length=len(data), max_len=a.max_len,
                     scanned=done, elapsed=round(dt, 4), hit_count=len(res))
    else:
        print(f'[xorbrute] {len(data)}B, key 长度 ≤{a.max_len}, 用时 {time.time()-t0:.2f}s')
        for sc, kh, pt in res:
            print(f'  score={sc:5d} key={kh:<12} → {pt[:60]!r}')


def main():
    ap = argparse.ArgumentParser(
        description='ctf_tools — CTF 密码学内核（完整说明见 CLI_REFERENCE.md）')
    ap.add_argument('--backend', default='auto', choices=['auto', 'torch', 'numpy'],
                    help='计算后端：auto=torch-cuda→torch-cpu→numpy（默认 auto）')
    ap.add_argument('--chunk', type=int, default=None,
                    help='每批候选数 / 断点粒度（默认：chacha --brute 在 GPU 上 262144、'
                         '其它 16384；xorbrute 默认 16384）')
    C.add_output_args(ap)
    C.add_sched_args(ap)
    sub = ap.add_subparsers(dest='cmd', required=True,
                            metavar='{selftest,chacha,poly1305,rs63,crc32,xorbrute,aesctr}')
    sub.add_parser('selftest', help='已知向量自测（18 项；全 PASS 退出码 0）')

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

    p = sub.add_parser('aesctr', help='AES-128/192/256-CTR 密钥流 / 加解密')
    p.add_argument('--key-hex', required=True, help='16/24/32 字节 AES 密钥（128/192/256）')
    p.add_argument('--nonce-hex', required=True,
                   help='8 字节（nonce||counter64）、12 字节（GCM 风格 nonce||counter32）或 16 字节整块计数器')
    p.add_argument('--nonce-mode', default='le', choices=['le', 'be'],
                   help='计数器字节序：le=小端（默认）/ be=大端')
    p.add_argument('--mode', default='dec', choices=['ks', 'enc', 'dec'],
                   help='ks=输出密钥流；enc/dec=对 --data-hex 异或（默认 dec）')
    p.add_argument('--data-hex', default='', help='enc/dec 模式的输入数据 hex')
    p.add_argument('--length', type=int, default=64, help='ks 模式密钥流字节数，默认 64')
    p.add_argument('--counter', type=int, default=0, help='起始计数器，默认 0')
    p.add_argument('--out-text', action='store_true', help='额外打印 text = repr(out)')

    a = ap.parse_args()
    C.apply_output_args(a)
    name, dev = G.init_backend(a.backend)
    C.info(f'[backend] {name} device={dev}\n')
    handler = {'selftest': cmd_selftest, 'chacha': cmd_chacha, 'rs63': cmd_rs63,
               'crc32': cmd_crc32, 'xorbrute': cmd_xorbrute,
               'poly1305': cmd_poly1305, 'aesctr': cmd_aesctr}[a.cmd]
    try:
        rc = handler(a)
    except ValueError as e:                     # 参数/输入错误：干净报错，不再抛 traceback
        return C.fail(str(e), 2)
    return 0 if not isinstance(rc, int) else rc


if __name__ == '__main__':
    sys.exit(main())
