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
import ctypes
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


# ---------------------------------------------------------------- 原生 CUDA 内核（NVRTC JIT）
# 目标：把"每块上千次小 kernel launch"变成"每块 1 次融合 kernel"，从根上解决 launch-bound。
# 实现：ctypes 直接调 torch 自带的 NVRTC（编译 CUDA C） + CUDA Driver API（加载/发射/拷贝）。
#       原生层自身只依赖 numpy/ctypes（不 import torch）；`--backend`/`init_backend`/`xp()`/`DEV`
#       语义完全不变 —— 它只是 cmd_mtscan/cmd_lfsrscan 在 torch-cuda 后端上的内部加速器。
# 失败/缺失时 _NV 置 False，命令自动回退到 torch/numpy 路径，功能与退出码不变。
_NV = None                 # None=尚未尝试；False=不可用；_NvCuda=可用
_NV_TRIED = False
_NV_ERR = ''               # 最近一次失败原因（报告/排障用）
_NV_DISABLE = bool(os.environ.get('GPU_CRACK_NO_NATIVE', '') or os.environ.get('CTF_NO_NATIVE', ''))
_NV_MAX_HITS = 1 << 20     # 单块命中缓冲上限（短目标如 1 字节时命中很多）
_P = ctypes.c_void_p


class _NvFail(RuntimeError):
    """原生内核运行期失败：命令层捕获后静默回退 torch/numpy 路径。"""

_NV_SRC = rb'''
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 mt_temper(u32 y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= y >> 18;
    return y;
}

/* modded mt19937: init_genrand then temper the first L states (no twist). */
extern "C" __global__ void mt_scan_t0(u64 seed0, u32 count, int L,
                                      const unsigned char* __restrict__ tgt,
                                      u32* __restrict__ hit_count,
                                      u64* __restrict__ hit_seeds,
                                      u32 max_hits)
{
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    u32 st[24];
    u32 v = (u32)(seed0 + (u64)tid);
    st[0] = v;
    for (int i = 1; i < L; ++i) {
        v = 1812433253u * (v ^ (v >> 30)) + (u32)i;
        st[i] = v;
    }
    unsigned int ok = 1u;
    for (int i = 0; i < L; ++i) {
        u32 y = mt_temper(st[i]);
        if ((unsigned char)(y & 0xFFu) != tgt[i]) { ok = 0u; break; }
    }
    if (ok) {
        u32 idx = atomicAdd(hit_count, 1u);
        if (idx < max_hits) hit_seeds[idx] = seed0 + (u64)tid;
    }
}

/* standard std::mt19937: only the first L twisted states are needed.
 * twist step i uses old_st[i], old_st[i+1], old_st[i+397]; for i < L the indices
 * (i+1, i+397) are both > i, i.e. never overwritten yet by the in-place twist.
 * So init up to index 396+L and save {0..L} U {397..396+L} (2L+1 words) and the
 * first L twisted outputs can be computed directly - 624 sequential steps avoided. */
extern "C" __global__ void mt_scan_t1(u64 seed0, u32 count, int L,
                                      const unsigned char* __restrict__ tgt,
                                      u32* __restrict__ hit_count,
                                      u64* __restrict__ hit_seeds,
                                      u32 max_hits)
{
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    u32 saved[49];
    int ns = 0;
    u32 v = (u32)(seed0 + (u64)tid);
    int maxidx = 396 + L;
    saved[ns++] = v;                          /* old st[0] */
    for (int i = 1; i <= maxidx; ++i) {
        v = 1812433253u * (v ^ (v >> 30)) + (u32)i;
        if (i <= L) saved[ns++] = v;         /* old st[1..L] */
        else if (i >= 397) saved[ns++] = v;  /* old st[397..396+L] */
    }
    unsigned int ok = 1u;
    for (int i = 0; i < L; ++i) {
        u32 a = saved[i], b = saved[i + 1], c = saved[L + 1 + i];
        u32 yv = (a & 0x80000000u) | (b & 0x7FFFFFFFu);
        u32 key = c ^ (yv >> 1) ^ ((yv & 1u) * 0x9908B0DFu);
        key = mt_temper(key);
        if ((unsigned char)(key & 0xFFu) != tgt[i]) { ok = 0u; break; }
    }
    if (ok) {
        u32 idx = atomicAdd(hit_count, 1u);
        if (idx < max_hits) hit_seeds[idx] = seed0 + (u64)tid;
    }
}

/* bit-packed Fibonacci LFSR: one u64 per trial (bit j = state bit j).
 * fb = parity(state & tapmask); state <<= 1 | fb (shifted-out high bits never
 * feed back, so no mask is needed as long as n <= 63). */
extern "C" __global__ void lfsr_scan(const u64* __restrict__ init, u64 tapmask, u32 count,
                                     int Tlen, const unsigned char* __restrict__ tgt,
                                     u64* __restrict__ best)
{
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    u64 st = init[tid];
    u64 sc = 0;
    for (int k = 0; k < Tlen; ++k) {
        u64 fb = __popcll(st & tapmask) & 1ull;
        sc += (fb == (u64)tgt[k]);
        st = (st << 1) | fb;
    }
    atomicMax(best, sc);
}
'''

_NV_KERNELS = ('mt_scan_t0', 'mt_scan_t1', 'lfsr_scan')


class _NvCuda(object):
    """NVRTC + CUDA Driver API 的最小封装（ctypes；不依赖 torch 导入）。"""

    def __init__(self):
        import glob
        import importlib.util
        spec = importlib.util.find_spec('torch')
        if spec is None or not spec.origin:
            raise RuntimeError('未找到 torch 安装，无法定位 NVRTC')
        libdir = os.path.join(os.path.dirname(spec.origin), 'lib')
        win = sys.platform.startswith('win')
        if win:
            load = ctypes.WinDLL
            driver_name = 'nvcuda.dll'
            nvrtc_pat, builtin_pat = 'nvrtc64_*.dll', 'nvrtc-builtins64_*.dll'
        else:
            load = ctypes.CDLL
            driver_name = 'libcuda.so.1'
            nvrtc_pat, builtin_pat = 'libnvrtc.so*', 'libnvrtc-builtins.so*'

        def pick(pattern, exclude=()):
            cands = [p for p in glob.glob(os.path.join(libdir, pattern))
                     if not any(x in os.path.basename(p).lower() for x in exclude)]
            if not cands:
                raise RuntimeError('缺少 %s（torch/lib 下未找到）' % pattern)
            return sorted(cands)[-1]

        nvrtc_path = pick(nvrtc_pat, exclude=('.alt',))
        builtin_path = pick(builtin_pat)
        try:                                   # nvrtc 运行期会按名字加载 builtins，先显式载入
            self._builtin = load(builtin_path)
        except OSError:
            self._builtin = None
        self._nvrtc = load(nvrtc_path)
        self._cuda = load(driver_name)
        self._init_api()
        self._nvrtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(_P), ctypes.c_char_p,
                                                   ctypes.c_char_p, ctypes.c_int, _P, _P]
        self._nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
        self._nvrtc.nvrtcCompileProgram.argtypes = [_P, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
        for fn, args in (('nvrtcGetPTXSize', [ _P, ctypes.POINTER(ctypes.c_size_t)]),
                         ('nvrtcGetPTX', [_P, ctypes.c_char_p]),
                         ('nvrtcGetProgramLogSize', [_P, ctypes.POINTER(ctypes.c_size_t)]),
                         ('nvrtcGetProgramLog', [_P, ctypes.c_char_p])):
            getattr(self._nvrtc, fn).argtypes = args
            getattr(self._nvrtc, fn).restype = ctypes.c_int

        self._chk(self._cuda.cuInit(0), 'cuInit')
        dev = ctypes.c_int()
        self._chk(self._cuda.cuDeviceGet(ctypes.byref(dev), 0), 'cuDeviceGet')
        maj, mnr = ctypes.c_int(), ctypes.c_int()
        self._cuda.cuDeviceGetAttribute(ctypes.byref(maj), 75, dev)
        self._cuda.cuDeviceGetAttribute(ctypes.byref(mnr), 76, dev)
        self.arch = 'compute_%d%d' % (maj.value, mnr.value)
        # 用 primary context（与 torch/numpy 共用同一 CUDA 上下文，避免双 context 冲突）
        self._ctx = _P()
        self._chk(self._cuda.cuDevicePrimaryCtxRetain(ctypes.byref(self._ctx), dev),
                  'cuDevicePrimaryCtxRetain')
        self._chk(self._cuda.cuCtxSetCurrent(self._ctx), 'cuCtxSetCurrent')
        ptx = self._compile(_NV_SRC)
        self._module = _P()
        self._chk(self._cuda.cuModuleLoadData(ctypes.byref(self._module),
                                              ctypes.cast(ptx, _P)), 'cuModuleLoadData')
        self._funcs = {}
        for name in _NV_KERNELS:
            f = _P()
            self._chk(self._cuda.cuModuleGetFunction(ctypes.byref(f), self._module, name.encode()),
                      'cuModuleGetFunction(%s)' % name)
            self._funcs[name] = f
        self._d = {}
        self._alloc('tgt', 64)
        self._alloc('cnt', 4)
        self._alloc('best', 8)
        self._alloc('hits', _NV_MAX_HITS * 8)
        self._init_cap = 0
        self._tbits_cap = 0

    # ---- ctypes / driver 基础
    def _init_api(self):
        c = self._cuda
        def proto(name, argtypes, restype=ctypes.c_int):
            fn = getattr(c, name)
            fn.argtypes = argtypes
            fn.restype = restype
        proto('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
        proto('cuDeviceGetAttribute', [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int])
        proto('cuDevicePrimaryCtxRetain', [ctypes.POINTER(_P), ctypes.c_int])
        proto('cuCtxSetCurrent', [_P])
        proto('cuCtxSynchronize', [])
        proto('cuModuleLoadData', [ctypes.POINTER(_P), _P])
        proto('cuModuleGetFunction', [ctypes.POINTER(_P), _P, ctypes.c_char_p])
        alloc = getattr(c, 'cuMemAlloc_v2', c.cuMemAlloc)
        alloc.argtypes = [ctypes.POINTER(_P), ctypes.c_size_t]
        alloc.restype = ctypes.c_int
        self._alloc_fn = alloc
        free = getattr(c, 'cuMemFree_v2', None)
        if free is not None:
            free.argtypes = [_P]
        h2d = getattr(c, 'cuMemcpyHtoD_v2', c.cuMemcpyHtoD)
        h2d.argtypes = [_P, _P, ctypes.c_size_t]
        h2d.restype = ctypes.c_int
        self._h2d_fn = h2d
        d2h = getattr(c, 'cuMemcpyDtoH_v2', c.cuMemcpyDtoH)
        d2h.argtypes = [_P, _P, ctypes.c_size_t]
        d2h.restype = ctypes.c_int
        self._d2h_fn = d2h
        memset = getattr(c, 'cuMemsetD8_v2', c.cuMemsetD8)
        memset.argtypes = [_P, ctypes.c_ubyte, ctypes.c_size_t]
        memset.restype = ctypes.c_int
        self._memset_fn = memset
        launch = c.cuLaunchKernel
        launch.argtypes = [_P, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                           ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                           _P, ctypes.POINTER(_P), ctypes.POINTER(_P)]
        launch.restype = ctypes.c_int
        self._launch_fn = launch

    @staticmethod
    def _chk(rc, what):
        if rc != 0:
            raise RuntimeError('CUDA driver 调用失败 %s: rc=%d' % (what, rc))

    def _compile(self, src):
        prog = _P()
        self._chk(self._nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b'gpu_crack_kernels.cu',
                                                 0, None, None), 'nvrtcCreateProgram')
        opts = ['--gpu-architecture=' + self.arch, '-default-device']
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        rc = self._nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
        sz = ctypes.c_size_t()
        self._nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
        log = ctypes.create_string_buffer(sz.value or 1)
        self._nvrtc.nvrtcGetProgramLog(prog, log)
        if rc != 0:
            raise RuntimeError('NVRTC 编译失败: ' + log.value.decode(errors='replace')[:1500])
        sz = ctypes.c_size_t()
        self._chk(self._nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz)), 'nvrtcGetPTXSize')
        ptx = ctypes.create_string_buffer(sz.value)
        self._chk(self._nvrtc.nvrtcGetPTX(prog, ptx), 'nvrtcGetPTX')
        return ptx

    def _alloc(self, key, nbytes):
        p = _P()
        self._chk(self._alloc_fn(ctypes.byref(p), max(1, int(nbytes))), 'cuMemAlloc(%s)' % key)
        self._d[key] = p
        return p

    def _upload(self, arr):
        arr = np.ascontiguousarray(arr)
        return arr, ctypes.c_void_p(arr.ctypes.data), int(arr.nbytes)

    def _h2d(self, dev, arr):
        arr, ptr, n = self._upload(arr)
        if n:
            self._chk(self._h2d_fn(dev, ptr, n), 'cuMemcpyHtoD')
        return arr

    def _d2h(self, host_arr, dev, nbytes):
        host_arr = np.ascontiguousarray(host_arr)
        self._chk(self._d2h_fn(ctypes.c_void_p(host_arr.ctypes.data), dev, int(nbytes)),
                  'cuMemcpyDtoH')
        return host_arr

    def _launch(self, name, count, block, arg_objs):
        args = (_P * len(arg_objs))()
        for i, obj in enumerate(arg_objs):
            args[i] = ctypes.cast(obj, _P)
        grid = (int(count) + block - 1) // block
        self._chk(self._launch_fn(self._funcs[name], grid, 1, 1, block, 1, 1, 0, None, args, None),
                  'cuLaunchKernel(%s)' % name)

    def sync(self):
        self._chk(self._cuda.cuCtxSynchronize(), 'cuCtxSynchronize')

    # ---- 业务入口
    def mt_scan(self, start, count, L, twist, tgt):
        """扫描 [start, start+count) 种子，返回 (升序命中种子列表, 内核报告命中数)。

        命中数超过 _NV_MAX_HITS 时列表会被截断（reported 仍为真实数，调用方负责告警）。
        """
        count = int(count)
        if count <= 0:
            return [], 0
        self._memset_fn(self._d['cnt'], 0, 4)
        self._h2d(self._d['tgt'], np.frombuffer(bytes(tgt), dtype=np.uint8).copy())
        seed0 = ctypes.c_uint64(int(start) & 0xFFFFFFFFFFFFFFFF)
        cnt = ctypes.c_uint32(count)
        Lc = ctypes.c_int(int(L))
        maxh = ctypes.c_uint32(_NV_MAX_HITS)
        self._launch('mt_scan_t1' if twist else 'mt_scan_t0', count, 256,
                     [ctypes.byref(seed0), ctypes.byref(cnt), ctypes.byref(Lc),
                      ctypes.byref(self._d['tgt']), ctypes.byref(self._d['cnt']),
                      ctypes.byref(self._d['hits']), ctypes.byref(maxh)])
        self.sync()
        cnt_host = np.zeros(1, dtype=np.uint32)
        self._d2h(cnt_host, self._d['cnt'], 4)
        nhit = int(cnt_host[0])
        k = min(nhit, _NV_MAX_HITS)
        out = np.zeros(k, dtype=np.uint64)
        if k:
            self._d2h(out, self._d['hits'], k * 8)
        hits = []
        for v in out.tolist():
            hits.append(int(v) if v < (1 << 63) else int(v) - (1 << 64))
        hits.sort()
        return hits, nhit

    def lfsr_scan(self, states, n, taps, tgt_bits):
        """states: u64 数组（每 trial 一个打包状态）；返回最佳一致位数。"""
        states = np.ascontiguousarray(states, dtype=np.uint64)
        count = int(states.size)
        if count == 0:
            return 0
        tapmask = 0
        for t in taps:
            tapmask |= 1 << (int(t) - 1)
        if count * 8 > self._init_cap:
            cap = max(count * 8, self._init_cap * 2, 1 << 20)
            self._alloc('init', cap)
            self._init_cap = cap
        self._h2d(self._d['init'], states)
        tgt = np.ascontiguousarray(np.asarray(tgt_bits), dtype=np.uint8)
        if tgt.size > self._tbits_cap:
            cap = max(int(tgt.size), self._tbits_cap * 2, 4096)
            self._alloc('tbits', cap)
            self._tbits_cap = cap
        self._h2d(self._d['tbits'], tgt)
        self._memset_fn(self._d['best'], 0, 8)
        tapmask_c = ctypes.c_uint64(tapmask)
        cnt = ctypes.c_uint32(count)
        tlen = ctypes.c_int(int(tgt.size))
        self._launch('lfsr_scan', count, 256,
                     [ctypes.byref(self._d['init']), ctypes.byref(tapmask_c), ctypes.byref(cnt),
                      ctypes.byref(tlen), ctypes.byref(self._d['tbits']),
                      ctypes.byref(self._d['best'])])
        self.sync()
        best_host = np.zeros(1, dtype=np.uint64)
        self._d2h(best_host, self._d['best'], 8)
        return int(best_host[0])


def _nv_available():
    """懒初始化原生 CUDA 层；不可用（无 GPU/无 NVRTC/禁用）返回 False，绝不抛异常。"""
    global _NV, _NV_TRIED, _NV_ERR
    if _NV_TRIED:
        return _NV is not None and _NV is not False
    _NV_TRIED = True
    if _NV_DISABLE:
        _NV, _NV_ERR = False, 'GPU_CRACK_NO_NATIVE 已设置'
        return False
    try:
        _NV = _NvCuda()
        return True
    except Exception as exc:                 # noqa: BLE001 - 任何失败都只回退，不影响功能
        _NV = False
        _NV_ERR = '%s: %s' % (type(exc).__name__, exc)
        return False


def _native_enabled():
    """原生内核只在 torch-cuda 后端上作为加速器启用（不改 BACKEND/xp()/DEV 语义）。"""
    return BACKEND == "torch-cuda" and _nv_available()


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


class _MuteStderrFd(object):
    """临时把 C 层 stderr(fd=2) 指到 devnull —— libkineto 在 Windows 上会朝 stderr
    打 "USDT: ... profiler_start/stop" 日志，--profile 时不应污染用户终端。"""

    def __init__(self, fd=2):
        self._fd = fd
        self._saved = None
        self._null = None

    def __enter__(self):
        try:
            sys.stderr.flush()
            self._null = os.open(os.devnull, os.O_WRONLY)
            self._saved = os.dup(self._fd)
            os.dup2(self._null, self._fd)
        except OSError:                        # 非文件描述符环境（如 IDE 捕获）：放弃抑制
            self._saved = None
        return self

    def __exit__(self, *exc_info):
        if self._saved is not None:
            try:
                os.dup2(self._saved, self._fd)
            finally:
                for f in (self._saved, self._null):
                    try:
                        os.close(f)
                    except OSError:
                        pass
                self._saved = self._null = None
        return False


def _impl_profile(fn, iters, count):
    """对单个内核做一次 CUDA 计时：返回 {gpu_s, launches, wall_s}（每次调用的累计量）。"""
    for _ in range(2):                       # 预热：JIT/缓存分配/首次 kernel 加载
        fn()
    _cuda_sync()
    iters = max(1, int(iters))
    prof = None
    if BACKEND == "torch-cuda":
        try:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA])
        except Exception:                    # noqa: BLE001 - 无 kineto 时退回事件计时
            prof = None
    if prof is not None:
        try:
            t0 = time.perf_counter()
            with _MuteStderrFd():
                with prof:
                    for _ in range(iters):
                        fn()
                    _cuda_sync()
            wall = time.perf_counter() - t0
            gpu_s = 0.0
            launches = 0
            for ev in prof.key_averages():
                dt = float(getattr(ev, 'self_device_time_total', 0.0) or 0.0)   # 微秒
                if dt > 0:
                    gpu_s += dt * 1e-6
                    launches += int(ev.count)
            return {'gpu_s': gpu_s / iters, 'launches': launches / iters,
                    'wall_s': wall / iters, 'count': count}
        except Exception:                    # noqa: BLE001
            pass
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return {'gpu_s': start.elapsed_time(end) / 1e3 / iters, 'launches': float(iters),
            'wall_s': wall / iters, 'count': count}


def _bench_profile_objs(specs):
    """--profile：对每个内核返回一条带 GPU 时间/launch 数/占比的事件对象。

    wall 用"不挂 profiler 的预热后多次取最小"单独测量（profiler 自身开销很大，
    若拿带 profiler 的 wall 当分母，GPU 占比会被严重低估）。
    """
    out = []
    for label, fn, iters, count in specs:
        for _ in range(2):
            fn()
        _cuda_sync()
        best = float('inf')
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            _cuda_sync()
            best = min(best, time.perf_counter() - t0)
        r = _impl_profile(fn, iters, count)
        ratio = (r['gpu_s'] / best) if best > 0 else 0.0
        out.append({'event': 'profile', 'kernel': label,
                    'launches_per_call': round(r['launches'], 3),
                    'gpu_ms_per_call': round(r['gpu_s'] * 1e3, 4),
                    'wall_ms_per_call': round(best * 1e3, 4),
                    'gpu_ratio': round(min(ratio, 1.0), 4), 'count': count})
    return out


def cmd_bench(a):
    T = xp()
    md = bool(getattr(a, 'markdown', False))
    js = C.json_mode()
    profiling = bool(getattr(a, 'profile', False))
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
        """预热后多次取最小 —— 避免"短 kernel 被首次启动/缓存抖动放大"的统计口径问题。"""
        warm(fn, *args)
        best = float('inf')
        for _ in range(5):
            t0 = time.perf_counter()
            fn(*args)
            _cuda_sync()
            best = min(best, time.perf_counter() - t0)
        return best

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

    if not profiling:
        return
    if BACKEND != "torch-cuda":
        C.info(f"[profile] 当前后端 {BACKEND} 不支持 profiler（需 torch-cuda）")
        return
    # --profile：GPU 时间 / launch 次数 / 占 wall 比例，便于用户自查"GPU 利用率"由什么构成
    specs = []
    if BACKEND.startswith("torch"):
        specs.append(('popcount', lambda: popcount_matrix(x), 10, f'{n} 个 32bit'))
        specs.append(('mt19937', lambda: mt19937_keys(s, False), 20, f'{m} 个种子'))
        specs.append(('aes192', lambda: aes_ecb(blk, k, None, ai, ai), 10,
                      f'{blk.shape[0]} 块'))
    if _native_enabled():
        far = bytes(range(24))
        specs.append(('native_mt_t0', lambda: _NV.mt_scan(0, 50_000_000, 24, False, far),
                      10, '50M 个种子'))
        specs.append(('native_mt_t1', lambda: _NV.mt_scan(0, 10_000_000, 24, True, far),
                      5, '10M 个种子'))
        st_lfsr = _lfsr_gen_packed(np.random.default_rng(1), 1 << 16, 25)
        tbits = np.zeros(512, dtype=np.int64)
        specs.append(('native_lfsr', lambda: _NV.lfsr_scan(st_lfsr, 25, [10, 11, 24], tbits),
                      20, '65536 trials x512 拍'))
    if not specs:
        C.info(f"[profile] 当前后端 {BACKEND} 不支持 profiler（需 torch-cuda）")
        return
    try:
        objs = _bench_profile_objs(specs)
    except Exception as exc:                 # noqa: BLE001 - profiler 不可用不影响 bench 主输出
        C.info(f"[profile] profiler 不可用：{exc}")
        return
    if js:
        for obj in objs:
            C.emit(obj)
    else:
        for o in objs:
            print(f"[profile] {o['kernel']:<12} {o['launches_per_call']:>6.2f} 次 launch/次  "
                  f"GPU {o['gpu_ms_per_call']:>8.3f} ms/次  "
                  f"wall {o['wall_ms_per_call']:>8.3f} ms/次  "
                  f"GPU 占比 {o['gpu_ratio']*100:5.1f}%  ({o['count']})")


def _chunk_explicit(a):
    """用户是否显式给了 --chunk（用于 auto-chunk：默认值 65536 表示"自动档"）。

    兼容进程内调用方（直接构造 Namespace/对象并设置 .chunk）：只要 .chunk 存在且不等于
    默认值 65536，就视为用户显式指定，完全尊重。
    """
    if getattr(a, '_chunk_explicit', False):
        return True
    c = getattr(a, 'chunk', None)
    try:
        return c is not None and int(c) != (1 << 16)
    except (TypeError, ValueError):
        return False


def _native_mt_chunk(total):
    """原生融合内核的单块上限（命中数远小于 2^20，故可一次扫很多种子）。"""
    return max(1, min(int(total), 1 << 26))


def _torch_mt_chunk(total, twist):
    """torch 回退路径的 auto-chunk：按实测最优档位（twist=1 的 [N,624] 内存大，取小档）。"""
    base = (1 << 16) if twist else (1 << 18)
    return max(1, min(int(total), base))


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
    # 原生融合内核（默认加速器）：整块种子扫描 = 1 次 kernel launch，彻底去掉 launch-bound。
    # 失败/不可用/显存不足一律回退到 torch/numpy 路径，结果语义不变。
    native = _native_enabled()
    if native:
        try:                                   # 预检：极小规模试跑，失败立即回退
            _NV.mt_scan(a.seed_start, min(2, a.seed_count), L, twist, target)
        except Exception as exc:               # noqa: BLE001
            C.info(f"[warn] 原生 CUDA 内核不可用，回退 torch/numpy 路径：{exc}")
            native = False
    tgt_t = (T.tensor(list(target), dtype=T.uint8, device=DEV) if BACKEND.startswith("torch")
             else np.frombuffer(target, dtype=np.uint8))

    def _key_of(seed):
        """按种子重算完整密钥（命中复核 / 过滤断点里的陈旧命中）。"""
        seeds = (T.tensor([seed], dtype=T.int64, device=DEV) if BACKEND.startswith("torch")
                 else np.array([seed], dtype=np.int64))
        return np.asarray(from_dev(mt19937_keys(seeds, twist=twist, nbytes=L)[0]),
                          dtype=np.uint8)

    def _torch_step(lo, hi):
        cnt = hi - lo
        seeds = (T.arange(lo, hi, dtype=T.int64, device=DEV) if BACKEND.startswith("torch")
                 else np.arange(lo, hi, dtype=np.int64))
        keys = mt19937_keys(seeds, twist=twist, nbytes=L)
        eq = (keys == tgt_t).all(axis=1)
        out = []
        if bool(eq.any()):
            out = [int(lo) + int(i) for i in hit_indices(eq)]
        return cnt, out

    def _auto_chunk():
        if _chunk_explicit(a):
            # 安全上限：单块命中缓冲 2^20 条；L=1 时命中率 ~1/256，2^27 种子仍安全
            return max(1, min(int(a.chunk), 1 << 27))
        if native:
            return _native_mt_chunk(a.seed_count)
        if BACKEND == "torch-cuda":
            return _torch_mt_chunk(a.seed_count, twist)
        return 1 << 16

    truncated = [False]

    def _native_step(lo, hi):
        try:
            hits, reported = _NV.mt_scan(lo, hi - lo, L, twist, target)
        except Exception as exc:               # noqa: BLE001 - 运行期失败 -> 回退
            raise _NvFail(str(exc))
        if reported > len(hits) and not truncated[0]:
            truncated[0] = True
            C.info(f"[warn] 单块命中 {reported} 条超过回传上限 {len(hits)}，仅保留前 "
                   f"{len(hits)} 条（可加长 --key-hex 或减小 --chunk）")
        return hi - lo, hits

    while True:
        _step = _native_step if native else _torch_step
        try:
            total, found, dt = C.chunked(_step, a.seed_count, start=a.seed_start,
                                         chunk=_auto_chunk(), label='mtscan',
                                         checkpoint=getattr(a, 'checkpoint', ''),
                                         reset=getattr(a, 'reset_checkpoint', False),
                                         progress=not getattr(a, 'no_progress', False),
                                         ckpt_meta={'kind': 'mtscan', 'key_hex': target.hex(),
                                                    'twist': int(a.twist)},
                                         on_hit=lambda h: True)
            break
        except _NvFail as exc:                  # 原生内核运行期异常：静默回退后重跑
            C.info(f"[warn] 原生 CUDA 内核运行失败，回退 torch/numpy 路径：{exc}")
            native = False
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


def _lfsr_gen_packed(rng, count, n):
    """单次整数抽样生成均匀随机的 n 位初始状态（n<=63；跨后端逐位一致，可复现）。"""
    return np.ascontiguousarray(rng.integers(0, 1 << int(n), size=int(count), dtype=np.uint64))


def _lfsr_packed_best(states, taps, tbits):
    """位打包回退实现（torch 或 numpy）：每 trial 一个整数，逐拍 parity+移位，返回最佳一致位数。"""
    shifts = [int(t) - 1 for t in taps]
    total = int(states.size)
    if total == 0:
        return 0
    tb = np.asarray(tbits, dtype=np.int64)
    Tlen = int(tb.size)
    if BACKEND.startswith("torch"):
        st = torch.as_tensor(states, dtype=torch.int64, device=DEV)
        score = torch.zeros(total, dtype=torch.int64, device=DEV)
        tb_t = torch.as_tensor(tb, device=DEV)
        for k in range(Tlen):
            fb = st >> shifts[0]
            for s in shifts[1:]:
                fb = fb ^ (st >> s)
            fb = fb & 1
            score.add_(fb ^ tb_t[k])           # 先累加"不一致位数"，最后换成一致位数
            st.mul_(2)                         # 等价于 <<1（int64 回绕语义一致）
            st.bitwise_or_(fb)
        mism = int(score.min().item())
        return Tlen - mism
    st = np.asarray(states, dtype=np.uint64).copy()
    score = np.zeros(total, dtype=np.int64)
    for k in range(Tlen):
        fb = st >> np.uint64(shifts[0])
        for s in shifts[1:]:
            fb ^= st >> np.uint64(s)
        fb &= np.uint64(1)
        score += (fb != np.uint64(int(tb[k]))).astype(np.int64)
        st = (st << np.uint64(1)) | fb
    return Tlen - int(score.min())


def _lfsr_bits_best(rng, trials, n, taps, tbits):
    """n>63 的兼容路径（逐 bit 状态矩阵，保留旧实现语义）。"""
    state = rng.integers(0, 2, size=(trials, n), dtype=np.int64)
    taps_i = [t - 1 for t in taps]
    Tlen = int(np.size(tbits))
    if BACKEND.startswith("torch"):
        st = torch.tensor(state, device=DEV)
        score = torch.zeros(trials, dtype=torch.int64, device=DEV)
        for k in range(Tlen):
            fb = st[:, taps_i[0]].clone()
            for t in taps_i[1:]:
                fb ^= st[:, t]
            score += (fb == int(tbits[k])).to(torch.int64)
            st = torch.cat([fb.unsqueeze(1), st[:, :-1]], dim=1)
        return int(score.max().item())
    st = state.copy()
    score = np.zeros(trials, dtype=np.int64)
    for k in range(Tlen):
        fb = st[:, taps_i[0]].copy()
        for t in taps_i[1:]:
            fb ^= st[:, t]
        score += (fb == int(tbits[k]))
        st = np.concatenate([fb[:, None], st[:, :-1]], axis=1)
    return int(score.max())


def cmd_lfsrscan(a):
    """
    并行跑多条 Fibonacci LFSR，匹配目标比特流。
    --taps  "25:10,11,24" 或 "25:10,11,24;28:12,22,25;27:1,4,19,21,23"（多组）
    """
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
    # 原生融合内核：n<=63 时可把所有 trial 压到 u64 位打包，整组 1 次 kernel launch。
    native = _native_enabled() and all(n <= 63 for n, _t in groups)
    if native:
        try:
            n0, taps0 = groups[0]
            _NV.lfsr_scan(np.zeros(4, dtype=np.uint64), n0, taps0, target_bits)
        except Exception as exc:               # noqa: BLE001
            C.info(f"[warn] 原生 CUDA 内核不可用，回退 torch/numpy 路径：{exc}")
            native = False
    chunk = getattr(a, 'chunk', 1 << 16)
    if not _chunk_explicit(a):
        chunk = min(trials, (1 << 22) if native else (1 << 20))
    else:
        chunk = max(1, min(int(chunk), 1 << 24))   # 安全上限：1M 状态/8MB 设备缓冲 ×16
    chunk = max(1, int(chunk))

    for g in range(len(groups)):
        n, taps = groups[g]
        m = -1
        try:
            if n > 63:                          # 兼容路径（逐 bit 状态矩阵）
                m = _lfsr_bits_best(rng, trials, n, taps, target_bits)
            else:
                for lo in range(0, trials, chunk):
                    cnt = min(chunk, trials - lo)
                    states = _lfsr_gen_packed(rng, cnt, n)
                    if native:
                        try:
                            cur = _NV.lfsr_scan(states, n, taps, target_bits)
                        except Exception as exc:   # noqa: BLE001 - 运行期失败回退
                            C.info(f"[warn] 原生 CUDA 内核运行失败，回退 torch/numpy 路径：{exc}")
                            native = False
                            cur = _lfsr_packed_best(states, taps, target_bits)
                    else:
                        cur = _lfsr_packed_best(states, taps, target_bits)
                    if cur > m:
                        m = cur
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
class _ChunkAction(argparse.Action):
    """记录 --chunk 是否被显式给出（默认值 = 自动档，显式给出则完全尊重用户值）。"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, '_chunk_explicit', True)


_GLOBAL_SPECS = (
    (('--backend',), dict(default='auto', choices=['auto', 'torch', 'numpy'],
                          help='计算后端：auto=torch-cuda→torch-cpu→numpy（默认 auto；子命令前后均可）')),
    (('--chunk',), dict(type=int, default=1 << 16, action=_ChunkAction,
                        help='每批候选数（默认 65536=自动档：mtscan 按后端/内核自动选 256K~64M；'
                             '显式给出时完全尊重该值；显存不足时自动减半；子命令前后均可）')),
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
    p.add_argument("--profile", action="store_true",
                   help="额外逐内核输出 GPU 时间 / launch 次数 / 占 wall 比例（诊断 GPU 利用率；"
                        "stdout 追加 [profile] 行，--json 时追加 event=profile 对象）")

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
