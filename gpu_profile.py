#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_profile.py — GPU 利用率测量/对比工具（v3.5，可 CI/用户直接使用）。

两种模式
--------
1) CLI 子进程模式（默认）：
   对每条代表性命令后台起 `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw
   --format=csv,noheader,nounits -lms 100` 采样；统计 wall、GPU busy%
   （Σ util×间隔 / wall）、util 均值/峰值、显存峰值、功率均值；`--subtract-startup`
   额外用同解释器跑 `-c pass` 与 `import <模块>` 估启动成本（给出 wall_net）。

2) `--inproc` 进程内模式：
   import 被测模块 → 先 warmup（并做一次 profiler attach 丢弃首次 ~1s 开销）→ 再用
   torch.profiler 统计 **device_type==CUDA** 的事件（避免 CPU/CUDA 双计数）：
   GPU kernel 时间、launch 次数、CPU 侧时间（wall−kernel 间隙）；输出 kernel_time/wall。

用例（CASES，10 条）：bench / mtscan twist0 50M / mtscan twist1 10M / sboxscan /
ksscore 5M / lfsrscan 65536×512 / chacha --brute 2M / xorbrute 3B 16B / eccscan 1M /
md5（库级批量）。`--quick` 用小规模变体（CI 友好）。

用法
----
    python -X utf8 gpu_profile.py --list
    python -X utf8 gpu_profile.py                          # CLI 模式，全部用例
    python -X utf8 gpu_profile.py --quick --subtract-startup
    python -X utf8 gpu_profile.py --only mtscan_twist0_50m,chacha_brute_2m --repeat 3
    python -X utf8 gpu_profile.py --json --save baseline.json
    python -X utf8 gpu_profile.py --compare baseline.json
    python -X utf8 gpu_profile.py --inproc --only bench,mtscan_twist0_50m
    python -X utf8 gpu_profile.py --selfcheck

退出码：0 = 全部用例成功；1 = 有用例失败/采样不可用；2 = 参数/环境错误。
脚本目录即被测仓库根目录（所有命令 cwd=脚本目录）——把本文件拷进 `git archive` 导出的
基线副本即可测量历史版本。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback

SCHEMA_VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
SMI = shutil.which("nvidia-smi") or "nvidia-smi"
SMI_QUERY = ["--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"]
SMI_INFO = ["--query-gpu=name,memory.total", "--format=csv,noheader"]
DEFAULT_TIMEOUT = 300.0
STARTUP_IMPORT_MODULES = ("gpu_crack", "ctf_tools", "ctf_ext", "ctf_misc")

# ---------------------------------------------------------------- fixtures
LFSR_BITS_512 = ("01101001110101011010110101101" * 18)[:512]
LFSR_BITS_128 = LFSR_BITS_512[:128]
XOR_DATA_16 = "1b37373331363f78151b7f2b78343133"
KS_TARGET_32 = "00112233445566778899aabbccddeeff" * 2
MTS_KEY_TW0 = "ac2f75c043fbc36709d315f2"      # 12B，50M 种子内不命中（用于测满扫描吞吐）
MTS_KEY_TW1 = "0123456789abcdef01234567"      # 12B，twist=1 下同样无命中
SBOX_PLAIN = "00" * 16
SBOX_CIPHER = "11" * 16
SBOX_KEY = "00" * 16
# k=2000137 的 secp256k1 公钥（eccscan 在 0..1M 内不会命中，测量整段搜索）
ECC_PX = ("50784693194400064320386007011096335489173144059601428367603945265906539926358")
ECC_PY = ("48687551354910519962870191552886541707169251120027075262506023744225214818254")
MD5_CODE = ("import time, ctf_ext\n"
            "n = 3000000\n"
            "msgs = [i.to_bytes(6, 'little') for i in range(n)]\n"
            "t = time.time()\n"
            "ctf_ext.md5_batch(msgs)\n"
            "print('MD5BATCH %.3fs' % (time.time() - t))\n")
MD5_CODE_QUICK = MD5_CODE.replace("n = 3000000", "n = 800000")

# ---------------------------------------------------------------- CASES
# kind: module -> argv 为相对脚本目录的命令（argv[0]=脚本名）
#       pyexpr -> code 为 -c 代码（用于库级 md5 批量）
CASES = [
    dict(name="bench", desc="bench 吞吐（popcount/mt19937/AES）", kind="module",
         module="gpu_crack", args=["gpu_crack.py", "bench"],
         work=None, unit=None,
         inproc=dict(kind="gpu", args=["--backend", "torch", "--quiet", "bench"])),
    dict(name="mtscan_twist0_50m", desc="mtscan twist0 50M 种子（v3.5 native 融合内核自动启用；v3.4 torch 回退）",
         kind="module", module="gpu_crack",
         args=["gpu_crack.py", "--backend", "torch", "--no-progress", "mtscan",
               "--key-hex", MTS_KEY_TW0, "--seed-count", "50000000"],
         quick_args=["gpu_crack.py", "--backend", "torch", "--no-progress", "mtscan",
                     "--key-hex", MTS_KEY_TW0, "--seed-count", "5000000"],
         work=50_000_000, unit="seeds",
         inproc=dict(kind="gpu", args=["--backend", "torch", "--quiet", "--no-progress",
                                       "mtscan", "--key-hex", MTS_KEY_TW0,
                                       "--seed-count", "1000000"],
                     quick_args=["--backend", "torch", "--quiet", "--no-progress",
                                 "mtscan", "--key-hex", MTS_KEY_TW0,
                                 "--seed-count", "300000"])),
    dict(name="mtscan_twist1_10m", desc="mtscan twist1 10M 种子（标准 std::mt19937；v3.5 native 自动启用）",
         kind="module", module="gpu_crack",
         args=["gpu_crack.py", "--backend", "torch", "--no-progress", "mtscan",
               "--key-hex", MTS_KEY_TW1, "--seed-count", "10000000", "--twist", "1"],
         quick_args=["gpu_crack.py", "--backend", "torch", "--no-progress", "mtscan",
                     "--key-hex", MTS_KEY_TW1, "--seed-count", "1000000", "--twist", "1"],
         work=10_000_000, unit="seeds",
         inproc=dict(kind="gpu", args=["--backend", "torch", "--quiet", "--no-progress",
                                       "mtscan", "--key-hex", MTS_KEY_TW1,
                                       "--seed-count", "200000", "--twist", "1"],
                     quick_args=["--backend", "torch", "--quiet", "--no-progress",
                                 "mtscan", "--key-hex", MTS_KEY_TW1,
                                 "--seed-count", "50000", "--twist", "1"])),
    dict(name="sboxscan", desc="sboxscan 65280 个 (a,b) 组合（合法向量，未命中）",
         kind="module", module="gpu_crack",
         args=["gpu_crack.py", "--backend", "torch", "--no-progress", "sboxscan",
               "--plain-hex", SBOX_PLAIN, "--cipher-hex", SBOX_CIPHER,
               "--key-hex", SBOX_KEY],
         work=65280, unit="(a,b) pairs",
         inproc=dict(kind="gpu", args=["--backend", "torch", "--quiet", "--no-progress",
                                       "sboxscan", "--plain-hex", SBOX_PLAIN,
                                       "--cipher-hex", SBOX_CIPHER, "--key-hex", SBOX_KEY])),
    dict(name="ksscore_5m", desc="ksscore 5M 候选比特一致度打分",
         kind="module", module="gpu_crack",
         args=["gpu_crack.py", "--backend", "torch", "--no-progress", "ksscore",
               "--target-hex", KS_TARGET_32, "--candidates", "5000000"],
         quick_args=["gpu_crack.py", "--backend", "torch", "--no-progress", "ksscore",
                     "--target-hex", KS_TARGET_32, "--candidates", "1000000"],
         work=5_000_000, unit="candidates",
         inproc=dict(kind="gpu", args=["--backend", "torch", "--quiet", "--no-progress",
                                       "ksscore", "--target-hex", KS_TARGET_32,
                                       "--candidates", "1000000"],
                     quick_args=["--backend", "torch", "--quiet", "--no-progress",
                                 "ksscore", "--target-hex", KS_TARGET_32,
                                 "--candidates", "200000"])),
    dict(name="lfsrscan_65536x512", desc="lfsrscan 65536 trials × 512 拍（--rng-seed 固定；v3.5 n<=63 走 native 融合内核）",
         kind="module", module="gpu_crack",
         args=["gpu_crack.py", "--backend", "torch", "--no-progress", "lfsrscan",
               "--taps", "25:10,11,24", "--bits", LFSR_BITS_512,
               "--trials", "65536", "--rng-seed", "11"],
         quick_args=["gpu_crack.py", "--backend", "torch", "--no-progress", "lfsrscan",
                     "--taps", "25:10,11,24", "--bits", LFSR_BITS_128,
                     "--trials", "8192", "--rng-seed", "11"],
         work=65536, unit="trials",
         inproc=dict(kind="gpu", args=["--backend", "torch", "--quiet", "--no-progress",
                                       "lfsrscan", "--taps", "25:10,11,24",
                                       "--bits", LFSR_BITS_128, "--trials", "8192",
                                       "--rng-seed", "11"],
                     quick_args=["--backend", "torch", "--quiet", "--no-progress",
                                 "lfsrscan", "--taps", "25:10,11,24",
                                 "--bits", LFSR_BITS_128, "--trials", "2048",
                                 "--rng-seed", "11"])),
    dict(name="chacha_brute_2m", desc="chacha --brute 2M 候选（无命中；默认 chunk：v3.4=16K / v3.5 GPU=256K）",
         kind="module", module="ctf_tools",
         args=["ctf_tools.py", "--backend", "torch", "--quiet",
               "chacha", "--target-hex", "ffffffff", "--brute", "--tries", "2000000"],
         quick_args=["ctf_tools.py", "--backend", "torch", "--quiet",
                     "chacha", "--target-hex", "ffffffff", "--brute", "--tries", "200000"],
         work=2_000_000, unit="keys",
         inproc=dict(kind="tools", args=["--backend", "torch", "--quiet", "--chunk",
                                         "262144", "chacha", "--target-hex", "ffffffff",
                                         "--brute", "--tries", "1000000"],
                     quick_args=["--backend", "torch", "--quiet", "--chunk", "131072",
                                 "chacha", "--target-hex", "ffffffff", "--brute",
                                 "--tries", "200000"])),
    dict(name="xorbrute_3b_16b", desc="xorbrute 3 字节密钥 × 16B 密文（v3.5 GPU 打分/top-k；v3.4 CPU numpy）",
         kind="module", module="ctf_tools",
         args=["ctf_tools.py", "--backend", "torch", "--quiet", "xorbrute",
               "--data-hex", XOR_DATA_16, "--max-len", "3", "--top", "3"],
         quick_args=["ctf_tools.py", "--backend", "torch", "--quiet", "xorbrute",
                     "--data-hex", XOR_DATA_16, "--max-len", "2", "--top", "3"],
         work=16_843_008, unit="candidates",
         inproc=dict(kind="tools", args=["--backend", "torch", "--quiet", "xorbrute",
                                         "--data-hex", XOR_DATA_16, "--max-len", "2",
                                         "--top", "3"],
                     quick_args=["--backend", "torch", "--quiet", "xorbrute",
                                 "--data-hex", XOR_DATA_16, "--max-len", "1",
                                 "--top", "3"])),
    dict(name="eccscan_1m", desc="eccscan secp256k1 0..1M 私钥区间（v3.5 仍单进程 CPU 椭圆曲线，jobs=1）",
         kind="module", module="ctf_ext",
         args=["ctf_ext.py", "--backend", "torch", "--quiet", "eccscan",
               "--curve", "secp256k1", "--px", ECC_PX, "--py", ECC_PY,
               "--start", "0", "--count", "1000000"],
         quick_args=["ctf_ext.py", "--backend", "torch", "--quiet", "eccscan",
                     "--curve", "secp256k1", "--px", ECC_PX, "--py", ECC_PY,
                     "--start", "0", "--count", "200000"],
         work=1_000_000, unit="keys",
         inproc=dict(kind="ext", args=["--backend", "torch", "--quiet", "eccscan",
                                       "--curve", "secp256k1", "--px", ECC_PX,
                                       "--py", ECC_PY, "--start", "0",
                                       "--count", "200000"],
                     quick_args=["--backend", "torch", "--quiet", "eccscan",
                                 "--curve", "secp256k1", "--px", ECC_PX,
                                 "--py", ECC_PY, "--start", "0", "--count", "50000"])),
    dict(name="md5_crack_3m",
         desc="md5 口令爆破 12^6≈3.26M 候选（v3.5 GPU 批处理；v3.4 纯 CPU hashlib）",
         kind="module", module="ctf_ext",
         args=["ctf_ext.py", "--backend", "torch", "--quiet", "md5",
               "--target", "4c467321dd2d571a787706eb4466a09d",
               "--charset", "abcdefghijkl", "--max-len", "6"],
         quick_args=["ctf_ext.py", "--backend", "torch", "--quiet", "md5",
                     "--target", "4c467321dd2d571a787706eb4466a09d",
                     "--charset", "abcdefghijkl", "--max-len", "5"],
         work=3_257_437, unit="candidates",
         inproc=dict(kind="ext", args=["--backend", "torch", "--quiet", "md5",
                                       "--target", "4c467321dd2d571a787706eb4466a09d",
                                       "--charset", "abcdefghijkl", "--max-len", "6"],
                     quick_args=["--backend", "torch", "--quiet", "md5",
                                 "--target", "4c467321dd2d571a787706eb4466a09d",
                                 "--charset", "abcdefghijkl", "--max-len", "5"])),
]
CASES_BY_NAME = {c["name"]: c for c in CASES}


# ---------------------------------------------------------------- 基础工具
def _info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _num(text: str):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
    return float(m.group()) if m else None


def parse_gpu_line(line: str):
    """解析一行 nvidia-smi CSV（无单位输出 '98, 7123, 39.77'；兼容带单位/N/A）。

    返回 (util_pct, mem_mb, power_w) 或 None；mem/power 可为 None（N/A）。
    """
    if not line or not line.strip():
        return None
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 3:
        return None
    util = _num(parts[0])
    if util is None:
        return None
    mem = _num(parts[1])
    power = _num(parts[2])
    return util, mem, power


def parse_smi_info(line: str):
    """解析 'NVIDIA GeForce RTX 5070 Laptop GPU, 8192' -> (name, total_mb)。"""
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if not parts or not parts[0]:
        return None
    total = _num(parts[1]) if len(parts) > 1 else None
    return parts[0], total


def aggregate_samples(samples, t0: float, t1: float) -> dict:
    """把 [(t, util, mem, power)] 聚合成 busy%/均值/峰值（busy=Σutil×dt/wall）。"""
    span = max(t1 - t0, 1e-9)
    sel = [s for s in samples if t0 - 1.0 <= s[0] <= t1 + 1.0]
    if not sel:
        return {"available": False, "samples": 0, "busy_pct": None, "util_mean_pct": None,
                "util_peak_pct": None, "mem_peak_mb": None, "power_mean_w": None,
                "sample_span_s": 0.0}
    busy = 0.0
    for i, (t, util, _mem, _pw) in enumerate(sel):
        nxt = sel[i + 1][0] if i + 1 < len(sel) else t1
        a, b = max(t, t0), min(nxt, t1)
        if b > a:
            busy += (util / 100.0) * (b - a)
    busy = max(0.0, min(busy, span))
    utils = [s[1] for s in sel]
    mems = [s[2] for s in sel if s[2] is not None]
    pows = [s[3] for s in sel if s[3] is not None]
    return {
        "available": True,
        "samples": len(sel),
        "sample_span_s": round(sel[-1][0] - sel[0][0], 3),
        "util_mean_pct": round(statistics.fmean(utils), 2),
        "util_peak_pct": round(max(utils), 2),
        "busy_s": round(busy, 4),
        "busy_pct": round(busy / span * 100.0, 2),
        "mem_peak_mb": round(max(mems), 1) if mems else None,
        "power_mean_w": round(statistics.fmean(pows), 2) if pows else None,
    }


def gpu_info() -> dict:
    info = {"smi": SMI, "available": False, "name": None, "mem_total_mb": None,
            "util_now_pct": None, "mem_used_now_mb": None, "power_now_w": None}
    try:
        r = subprocess.run([SMI, *SMI_INFO], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            parsed = parse_smi_info(r.stdout.strip().splitlines()[0])
            if parsed:
                info["available"] = True
                info["name"], info["mem_total_mb"] = parsed
        r2 = subprocess.run([SMI, *SMI_QUERY], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=20)
        if r2.returncode == 0 and r2.stdout.strip():
            cur = parse_gpu_line(r2.stdout.strip().splitlines()[0])
            if cur:
                info["util_now_pct"], info["mem_used_now_mb"], info["power_now_w"] = cur
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = str(exc)
    return info


def idle_check(max_wait: float = 60.0, threshold: float = 15.0) -> dict:
    """测量前查 GPU 是否低负载；忙则等待重试，超时仍忙则标注 contended。"""
    t0 = time.monotonic()
    probes = []
    while True:
        vals = []
        for _ in range(4):
            try:
                r = subprocess.run([SMI, *SMI_QUERY], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=15)
                if r.returncode == 0 and r.stdout.strip():
                    parsed = parse_gpu_line(r.stdout.strip().splitlines()[0])
                    if parsed:
                        vals.append(parsed[0])
            except (OSError, subprocess.SubprocessError):
                pass
            time.sleep(0.1)
        waited = round(time.monotonic() - t0, 1)
        if not vals:
            return {"available": False, "contended": False, "waited_s": waited,
                    "idle_util_pct": None, "probes": probes, "threshold_pct": threshold}
        mean_util = round(statistics.fmean(vals), 1)
        probes.append(mean_util)
        if mean_util <= threshold:
            return {"available": True, "contended": False, "waited_s": waited,
                    "idle_util_pct": mean_util, "probes": probes,
                    "threshold_pct": threshold}
        if time.monotonic() - t0 >= max_wait:
            _info(f"[gpu_profile] GPU 仍繁忙（util={mean_util}% > {threshold}%），"
                  f"已等待 {waited}s，继续测量并标注 contended=True")
            return {"available": True, "contended": True, "waited_s": waited,
                    "idle_util_pct": mean_util, "probes": probes,
                    "threshold_pct": threshold}
        _info(f"[gpu_profile] GPU 繁忙 util={mean_util}%（等待中，已等 {waited}s）")
        time.sleep(min(5.0, max(0.5, max_wait - (time.monotonic() - t0))))


class GPUSampler(threading.Thread):
    """后台线程：`nvidia-smi ... -lms 100` 流式采样，记录 (t, util, mem, power)。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.error = None
        self._stop = threading.Event()
        self.proc = None

    def run(self):
        try:
            self.proc = subprocess.Popen(
                [SMI, *SMI_QUERY, "-lms", "100"], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1, env=ENV)
            for line in self.proc.stdout:
                parsed = parse_gpu_line(line)
                if parsed:
                    self.samples.append((time.monotonic(), *parsed))
                if self._stop.is_set():
                    break
        except (OSError, subprocess.SubprocessError) as exc:
            self.error = str(exc)

    def stop(self, tail: float = 0.25):
        if tail:
            time.sleep(tail)
        self._stop.set()
        proc = self.proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                except OSError:
                    pass
        self.join(timeout=5)


# ---------------------------------------------------------------- 背景负载
def sample_background(duration: float = 3.0) -> dict:
    """采样一段「不动 GPU」的空闲窗口，量化并行 owner/其它进程的背景占用。

    用途：共享 GPU 上 busy% 会混入他人负载；报告同时给出 busy_pct_adj =
    max(0, busy% - 背景 busy%)，并保留原始 busy% 以便追溯。
    """
    sampler = GPUSampler()
    sampler.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    time.sleep(max(0.5, duration))
    t1 = time.monotonic()
    sampler.stop()
    agg = aggregate_samples(sampler.samples, t0, t1)
    agg["duration_s"] = round(t1 - t0, 3)
    if sampler.error:
        agg["error"] = sampler.error
    return agg


# ---------------------------------------------------------------- 启动成本
_STARTUP_CACHE = {}


def _timed_cmd(cmd, timeout=180.0, repeats=2):
    best = None
    for _ in range(max(1, repeats)):
        t = time.monotonic()
        try:
            subprocess.run(cmd, cwd=HERE, env=ENV, capture_output=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
        dt = time.monotonic() - t
        best = dt if best is None else min(best, dt)
    return round(best, 3)


def measure_startup(want_torch: bool = False, modules=(), bootstraps=()) -> dict:
    out = {"pass_s": _timed_cmd([PY, "-X", "utf8", "-c", "pass"]),
           "import_torch_s": None, "import_s": {}, "bootstrap_s": {}}
    if want_torch:
        out["import_torch_s"] = _timed_cmd([PY, "-X", "utf8", "-c", "import torch"],
                                           timeout=300.0, repeats=1)
    for mod in modules:
        if mod not in out["import_s"]:
            key = ("import", mod)
            if key not in _STARTUP_CACHE:
                _STARTUP_CACHE[key] = _timed_cmd(
                    [PY, "-X", "utf8", "-c", f"import {mod}"], timeout=300.0, repeats=2)
            out["import_s"][mod] = _STARTUP_CACHE[key]
    for name in bootstraps:
        code = BOOTSTRAPS.get(name)
        if not code:
            continue
        key = ("bootstrap", name)
        if key not in _STARTUP_CACHE:
            _STARTUP_CACHE[key] = _timed_cmd([PY, "-X", "utf8", "-c", code],
                                             timeout=300.0, repeats=2)
        out["bootstrap_s"][name] = _STARTUP_CACHE[key]
    return out


# ---------------------------------------------------------------- CLI 模式
BOOTSTRAPS = {
    "gpu_backend": "import gpu_crack as G; G.init_backend('torch')",
    "tools_backend": "import ctf_tools as T, gpu_crack as G; G.init_backend('torch')",
    "ext_backend": "import ctf_ext as E, gpu_crack as G; G.init_backend('torch')",
    "ctf_ext_import": "import ctf_ext",
}


def case_bootstrap_name(case: dict) -> str:
    """该用例最贴近的真实启动成本：import 模块 + init_backend('torch')（含 CUDA 初始化）。

    pyexpr（如库级 md5 批量）不初始化 torch 后端，只测模块 import，避免过度扣减。
    """
    if case.get("kind") == "pyexpr":
        return "ctf_ext_import"
    if case.get("module") == "gpu_crack":
        return "gpu_backend"
    if case.get("module") == "ctf_tools":
        return "tools_backend"
    if case.get("module") == "ctf_ext":
        return "ext_backend"
    return "ctf_ext_import"


def build_cmd(case: dict, quick: bool) -> list:
    if case["kind"] == "pyexpr":
        code = case.get("quick_code") if quick and case.get("quick_code") else case["code"]
        return [PY, "-X", "utf8", "-c", code]
    args = case.get("quick_args") if quick and case.get("quick_args") else case["args"]
    return [PY, "-X", "utf8", *args]


def _tail(text: str, n: int = 6) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def run_cli_case(case: dict, quick: bool, repeat: int, timeout: float,
                 subtract_startup: bool, startup: dict, repeat_cutoff: float = 15.0) -> dict:
    cmd = build_cmd(case, quick)
    boot_name = case_bootstrap_name(case)
    import_s = None
    if subtract_startup:
        import_s = startup.get("bootstrap_s", {}).get(boot_name)
    runs = []
    for i in range(repeat):
        sampler = GPUSampler()
        sampler.start()
        time.sleep(0.3)                              # 先拿到 2~3 个采样点再开跑
        t0 = time.monotonic()
        rc = -1
        stdout = stderr = ""
        try:
            r = subprocess.run(cmd, cwd=HERE, env=ENV, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
            rc, stdout, stderr = r.returncode, r.stdout or "", r.stderr or ""
        except subprocess.TimeoutExpired:
            rc = -9
            stderr = f"[timeout] 超过 {timeout:.0f}s"
        except (OSError, subprocess.SubprocessError) as exc:
            rc = -1
            stderr = f"[无法启动] {exc}"
        t1 = time.monotonic()
        sampler.stop()
        wall = round(t1 - t0, 3)
        wall_net = round(wall - import_s, 3) if import_s and wall > import_s else None
        wall_net_pass = (round(wall - startup["pass_s"], 3)
                         if startup.get("pass_s") and wall > startup["pass_s"] else None)
        gpu = aggregate_samples(sampler.samples, t0, t1)
        if sampler.error:
            gpu["error"] = sampler.error
        if gpu.get("busy_s") is not None and wall_net and wall_net > 0:
            gpu["busy_net_pct"] = round(min(gpu["busy_s"] / wall_net, 1.0) * 100.0, 2)
        throughput = (round(case["work"] / wall, 3) if case.get("work") and wall > 0
                      else None)
        throughput_net = (round(case["work"] / wall_net, 3)
                          if case.get("work") and wall_net and wall_net > 0 else None)
        runs.append({
            "run": i + 1, "wall_s": wall, "wall_net_s": wall_net,
            "bootstrap": boot_name,
            "wall_net_pass_s": wall_net_pass, "rc": rc, "gpu": gpu,
            "throughput": throughput, "throughput_net": throughput_net,
            "stdout_tail": _tail(stdout), "stderr_tail": _tail(stderr),
        })
        _info(f"[gpu_profile]   {case['name']} run{i + 1}/{repeat}: wall={wall:.2f}s "
              f"busy={gpu.get('busy_pct')}% util_peak={gpu.get('util_peak_pct')}% rc={rc}")
        if rc == -9:
            break
        if wall > repeat_cutoff and i == 0:
            _info(f"[gpu_profile]   {case['name']} 单次 wall={wall:.2f}s > "
                  f"{repeat_cutoff:.0f}s，长用例只跑 1 次（--repeat-cutoff 可调）")
            break
    ok = [r for r in runs if r["rc"] == 0]
    best = min(ok or runs, key=lambda r: r["wall_s"])
    busy_vals = [r["gpu"].get("busy_pct") for r in runs
                 if r["gpu"].get("available") and r["gpu"].get("busy_pct") is not None]
    gpu_best = dict(best["gpu"])
    gpu_best["busy_pct_mean_runs"] = (round(statistics.fmean(busy_vals), 2)
                                      if busy_vals else None)
    return {
        "name": case["name"],
        "desc": case["desc"],
        "cmd": " ".join(cmd[2:]) if case["kind"] == "module" else f"{PY} -c <md5 batch code>",
        "mode": "cli",
        "quick": quick,
        "runs": len(runs),
        "rc": 0 if len(ok) == len(runs) else best["rc"],
        "wall_s": best["wall_s"],
        "wall_net_s": best["wall_net_s"],
        "wall_net_pass_s": best["wall_net_pass_s"],
        "startup_import_s": import_s,
        "throughput": best["throughput"],
        "throughput_net": best["throughput_net"],
        "throughput_unit": case.get("unit"),
        "gpu": gpu_best,
        "run_details": runs,
    }


# ---------------------------------------------------------------- inproc 模式
_INPROC_MODS = {}


def _load_module(name: str):
    if name not in _INPROC_MODS:
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        _INPROC_MODS[name] = __import__(name)
    return _INPROC_MODS[name]


class _SilentIO:
    """吞掉被测代码的 stdout/stderr（inproc 下不污染本工具的 JSON 输出）。"""

    def __enter__(self):
        import io
        self._out, self._err = io.StringIO(), io.StringIO()
        self._old = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self._out, self._err
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._old
        return False


def _inproc_args(case: dict, quick: bool) -> list:
    spec = case["inproc"]
    if quick and spec.get("quick_args") is not None:
        return list(spec["quick_args"])
    return list(spec["args"])


def build_inproc_callable(case: dict, quick: bool):
    spec = case["inproc"]
    kind = spec["kind"]
    if kind == "code":
        code = spec.get("quick_code") if quick and spec.get("quick_code") else spec["code"]

        def _run():
            with _SilentIO():
                exec(compile(code, "<md5_batch_code>", "exec"), {"__name__": "__main__"})
        return _run
    args = _inproc_args(case, quick)
    if kind == "gpu":
        mod = _load_module("gpu_crack")

        def _run():
            with _SilentIO():
                rc = mod.main(list(args))
                if isinstance(rc, int) and rc:
                    raise RuntimeError(f"gpu_crack.main returned {rc}")
        return _run
    modname = {"tools": "ctf_tools", "ext": "ctf_ext"}[kind]
    mod = _load_module(modname)
    script = modname + ".py"

    def _run():
        old_argv = sys.argv
        try:
            sys.argv = [script, *args]
            with _SilentIO():
                try:
                    mod.main()
                except SystemExit as exc:              # ctf_ext.main 用 sys.exit 结束
                    if exc.code not in (0, None):
                        raise RuntimeError(f"{modname}.main SystemExit({exc.code})")
        finally:
            sys.argv = old_argv
    return _run


def _profile_run(fn) -> dict:
    import torch
    acts = [torch.profiler.ProfilerActivity.CPU]
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    # 丢弃首次 profiler attach 的 ~1s 开销（单独一次 enter/exit）
    with torch.profiler.profile(activities=acts):
        if has_cuda:
            torch.cuda.synchronize()
    if has_cuda:
        torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.profiler.profile(activities=acts) as prof:
        fn()
        if has_cuda:
            torch.cuda.synchronize()
    t1 = time.monotonic()
    cuda_dt = getattr(torch.autograd, "DeviceType", None)
    cuda_dt = cuda_dt.CUDA if cuda_dt is not None else None
    cpu_dt = cuda_dt and cuda_dt.CPU
    kernel_us = 0.0
    launches = 0
    by_kernel = {}
    cpu_event_us = 0.0
    try:
        events = list(prof.events())
    except (AttributeError, RuntimeError):
        events = []
    for evt in events:
        dt = getattr(evt, "device_type", None)
        dur = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if cuda_dt is not None and dt == cuda_dt:
            kernel_us += dur
            launches += 1
            if dur:
                by_kernel[evt.key] = by_kernel.get(evt.key, 0.0) + dur
        elif cpu_dt is not None and dt == cpu_dt:
            cpu_event_us += float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
    wall = max(t1 - t0, 1e-9)
    kernel_ms = kernel_us / 1000.0
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1])[:5]
    return {
        "wall_s": round(wall, 4),
        "kernel_ms": round(kernel_ms, 3),
        "kernel_pct": round(kernel_ms / 1000.0 / wall * 100.0, 2),
        "launches": launches,
        "cpu_gap_ms": round(max(0.0, wall * 1000.0 - kernel_ms), 3),
        "cpu_event_ms": round(cpu_event_us / 1000.0, 3),
        "top_kernels": [{"kernel": k, "ms": round(v / 1000.0, 3)} for k, v in top],
        "cuda_available": has_cuda,
    }


def run_inproc_case(case: dict, quick: bool, repeat: int) -> dict:
    if "inproc" not in case:
        return {"name": case["name"], "desc": case["desc"], "mode": "inproc",
                "skipped": True, "reason": "该用例无 inproc 定义", "runs": 0}
    fn = build_inproc_callable(case, quick)
    fn()                                              # warmup（含模块 import/CUDA 初始化）
    runs = []
    for i in range(max(1, repeat)):
        try:
            res = _profile_run(fn)
            runs.append(res)
            _info(f"[gpu_profile]   {case['name']} prof{i + 1}/{repeat}: "
                  f"wall={res['wall_s']:.2f}s kernel={res['kernel_ms']:.1f}ms "
                  f"({res['kernel_pct']:.1f}%) launches={res['launches']}")
        except Exception as exc:                      # profiler/被测代码异常不拖垮整体
            runs.append({"wall_s": None, "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-800:]})
            _info(f"[gpu_profile]   {case['name']} prof{i + 1}: ERROR {exc}")
            break
    ok = [r for r in runs if r.get("wall_s")]
    best = min(ok, key=lambda r: r["wall_s"]) if ok else (runs[0] if runs else {})
    return {
        "name": case["name"], "desc": case["desc"], "mode": "inproc",
        "quick": quick, "runs": len(runs),
        "rc": 0 if ok else 1,
        "wall_s": best.get("wall_s"),
        "kernel_ms": best.get("kernel_ms"),
        "kernel_pct": best.get("kernel_pct"),
        "launches": best.get("launches"),
        "cpu_gap_ms": best.get("cpu_gap_ms"),
        "cpu_event_ms": best.get("cpu_event_ms"),
        "top_kernels": best.get("top_kernels"),
        "cuda_available": best.get("cuda_available"),
        "error": best.get("error"),
        "run_details": runs,
    }


# ---------------------------------------------------------------- 对比/渲染
def compare_results(before: dict, after: dict) -> list:
    rows = []
    after_cases = {c["name"]: c for c in after.get("cases", []) if not c.get("skipped")}
    for b in before.get("cases", []):
        if b.get("skipped") or b["name"] not in after_cases:
            continue
        a = after_cases[b["name"]]
        wb, wa = b.get("wall_s"), a.get("wall_s")
        if before.get("mode") == "cli":
            gb, ga = b.get("gpu") or {}, a.get("gpu") or {}
            bb, ba = gb.get("busy_pct"), ga.get("busy_pct")
            bba, baa = gb.get("busy_pct_adj"), ga.get("busy_pct_adj")
        else:
            bb = ba = bba = baa = None
        rows.append({
            "name": b["name"],
            "wall_before_s": wb, "wall_after_s": wa,
            "speedup": round(wb / wa, 2) if wb and wa else None,
            "wall_delta_pct": round((wa - wb) / wb * 100.0, 1) if wb and wa else None,
            "busy_before_pct": bb, "busy_after_pct": ba,
            "busy_delta": round(ba - bb, 2) if bb is not None and ba is not None else None,
            "busy_adj_before_pct": bba, "busy_adj_after_pct": baa,
            "throughput_before": b.get("throughput"),
            "throughput_after": a.get("throughput"),
            "throughput_unit": b.get("throughput_unit") or a.get("throughput_unit"),
        })
    return rows


def _fmt_num(value, fmt: str = ".2f", none: str = "-") -> str:
    """表格格式化：None/NaN 显示为 '-' 而不是 'nan'。"""
    if value is None:
        return none
    try:
        f = float(value)
    except (TypeError, ValueError):
        return none
    if f != f:  # NaN
        return none
    return format(f, fmt)


def render_results(doc: dict) -> str:
    lines = []
    mode = doc["mode"]
    lines.append(f"gpu_profile  模式={mode}  label={doc['label']}  仓库={doc['repo']}")
    g = doc.get("gpu") or {}
    if g.get("available"):
        lines.append(f"GPU: {g.get('name')}  {g.get('mem_total_mb')} MB"
                     f"（测量前 util={g.get('util_now_pct')}%）")
    else:
        lines.append(f"GPU: 不可用（{g.get('error', 'nvidia-smi 未找到')}）")
    idle = doc.get("idle") or {}
    if idle.get("available"):
        lines.append(f"负载检查: util={idle['idle_util_pct']}%  等待={idle['waited_s']}s  "
                     f"contended={idle['contended']}（阈值 {idle['threshold_pct']}%）")
    bg = doc.get("background") or {}
    if bg.get("mean_busy_pct") is not None:
        lines.append(f"背景负载: busy={bg['mean_busy_pct']}%（before/after 各 "
                     f"{bg.get('before', {}).get('busy_pct')}% / "
                     f"{bg.get('after', {}).get('busy_pct')}%）")
    st = doc.get("startup") or {}
    if st.get("pass_s") is not None:
        lines.append(f"启动成本: -c pass={st['pass_s']:.3f}s"
                     + (f"  import torch={st['import_torch_s']:.3f}s"
                        if st.get("import_torch_s") else "")
                     + (f"  bootstrap={st['bootstrap_s']}" if st.get("bootstrap_s") else ""))
    lines.append("")
    if mode == "cli":
        hdr = (f"{'case':<24}{'wall(s)':>9}{'net(s)':>9}{'busy%':>7}{'netbusy%':>10}"
               f"{'peak%':>7}{'memMB':>8}{'powerW':>8}  {'throughput':<22}rc")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for c in doc["cases"]:
            gg = c.get("gpu") or {}
            thr = (f"{c['throughput']:,.0f} {c['throughput_unit']}/s"
                   if c.get("throughput") and c.get("throughput_unit") else "-")
            lines.append(
                f"{c['name']:<24}{_fmt_num(c.get('wall_s')):>9}"
                f"{_fmt_num(c.get('wall_net_s')):>9}"
                f"{_fmt_num(gg.get('busy_pct'), '.1f'):>7}"
                f"{_fmt_num(gg.get('busy_net_pct'), '.1f'):>10}"
                f"{_fmt_num(gg.get('util_peak_pct'), '.1f'):>7}"
                f"{_fmt_num(gg.get('mem_peak_mb'), '.0f'):>8}"
                f"{_fmt_num(gg.get('power_mean_w'), '.1f'):>8}"
                f"  {thr:<22}{c.get('rc')}")
    else:
        hdr = (f"{'case':<24}{'wall(s)':>9}{'kernel(ms)':>12}{'kernel%':>9}"
               f"{'launches':>10}{'cpu-gap(ms)':>13}  {'top kernel':<34}")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for c in doc["cases"]:
            if c.get("skipped"):
                lines.append(f"{c['name']:<24}  SKIP（{c.get('reason')}）")
                continue
            top = (c.get("top_kernels") or [{}])[0]
            top_s = (f"{top.get('kernel', '')[:30]} {top.get('ms', 0):.1f}ms"
                     if top else "-")
            lines.append(
                f"{c['name']:<24}{_fmt_num(c.get('wall_s')):>9}"
                f"{_fmt_num(c.get('kernel_ms'), '.1f'):>12}"
                f"{_fmt_num(c.get('kernel_pct'), '.1f'):>9}"
                f"{_fmt_num(c.get('launches'), '.0f'):>10}"
                f"{_fmt_num(c.get('cpu_gap_ms'), '.1f'):>13}"
                f"  {top_s:<34}")
    comp = doc.get("compare") or []
    if comp:
        lines.append("")
        lines.append("before → after 对比：")
        hdr2 = (f"{'case':<24}{'wall_b(s)':>10}{'wall_a(s)':>10}{'speedup':>9}"
                f"{'Δwall%':>9}{'busy_b%':>9}{'busy_a%':>9}{'Δtput':>10}")
        lines.append(hdr2)
        lines.append("-" * len(hdr2))
        for r in comp:
            tput = "-"
            if r.get("throughput_before") and r.get("throughput_after"):
                try:
                    tput = f"{r['throughput_after'] / r['throughput_before']:.2f}x"
                except ZeroDivisionError:
                    tput = "-"
            lines.append(
                f"{r['name']:<24}{_fmt_num(r.get('wall_before_s')):>10}"
                f"{_fmt_num(r.get('wall_after_s')):>10}"
                f"{_fmt_num(r.get('speedup')):>9}"
                f"{_fmt_num(r.get('wall_delta_pct'), '.1f'):>9}"
                f"{_fmt_num(r.get('busy_before_pct'), '.1f'):>9}"
                f"{_fmt_num(r.get('busy_after_pct'), '.1f'):>9}"
                f"{tput:>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------- selfcheck
def selfcheck() -> int:
    fails = []
    checks = [0]

    def check(name, cond, detail=""):
        checks[0] += 1
        print(("PASS " if cond else "FAIL ") + name + (f"  |  {detail}" if not cond else ""))
        if not cond:
            fails.append(name)

    # 1) nvidia-smi 行解析
    check("parse '98, 7123, 39.77'", parse_gpu_line("98, 7123, 39.77") == (98.0, 7123.0, 39.77))
    check("parse 带单位 '12 %, 1862 MiB, 18.42 W'",
          parse_gpu_line("12 %, 1862 MiB, 18.42 W") == (12.0, 1862.0, 18.42))
    check("parse N/A power", parse_gpu_line("0, 100, N/A") == (0.0, 100.0, None))
    check("parse 空行/坏行返回 None",
          parse_gpu_line("") is None and parse_gpu_line("bad") is None)
    # 2) busy% 聚合（合成样本）
    samples = [(0.0, 50.0, 100.0, 20.0), (0.1, 100.0, 200.0, 40.0),
               (0.2, 0.0, 300.0, 10.0)]
    agg = aggregate_samples(samples, 0.0, 0.25)
    check("aggregate busy% = 60%（Σutil×dt/wall）", abs(agg["busy_pct"] - 60.0) < 0.01, agg)
    check("aggregate util mean/peak = 50/100",
          abs(agg["util_mean_pct"] - 50.0) < 0.01 and abs(agg["util_peak_pct"] - 100.0) < 0.01,
          agg)
    check("aggregate mem/power 峰值/均值", agg["mem_peak_mb"] == 300.0 and
          abs(agg["power_mean_w"] - (70.0 / 3)) < 0.01, agg)
    empty = aggregate_samples([], 0.0, 1.0)
    check("aggregate 空样本 available=False", empty["available"] is False and
          empty["busy_pct"] is None, empty)
    # 3) CASES 完整性
    names = [c["name"] for c in CASES]
    check("CASES >= 10 条", len(CASES) >= 10, len(CASES))
    check("CASES 名称唯一", len(names) == len(set(names)), names)
    check("CASES 字段完整",
          all(c.get("kind") in ("module", "pyexpr") and c.get("desc") and
              (c.get("args") or c.get("code")) and
              (c.get("kind") != "pyexpr" or c.get("code")) for c in CASES))
    check("CASES 均带 inproc 定义", all("inproc" in c for c in CASES))
    q = [n for n in names if CASES_BY_NAME[n].get("quick_args") or
         CASES_BY_NAME[n].get("quick_code")]
    check("quick 变体覆盖长用例（>=5）", len(q) >= 5, q)
    # 4) 命令构造
    cmd = build_cmd(CASES_BY_NAME["bench"], quick=False)
    check("build_cmd(module) 以解释器开头且含脚本名",
          cmd[0] == PY and "-X" in cmd and "gpu_crack.py" in cmd, cmd)
    cmdq = build_cmd(CASES_BY_NAME["mtscan_twist0_50m"], quick=True)
    check("build_cmd(quick) 使用 quick_args", "5000000" in " ".join(cmdq), cmdq)
    cmdpy = build_cmd({"kind": "pyexpr", "code": "print(1)"}, quick=False)
    check("build_cmd(pyexpr) 使用 -c", cmdpy[:4] == [PY, "-X", "utf8", "-c"], cmdpy)
    # 5) 对比表
    before = {"mode": "cli", "cases": [
        {"name": "a", "wall_s": 10.0, "throughput": 100.0, "throughput_unit": "seeds",
         "gpu": {"busy_pct": 20.0}},
        {"name": "b", "wall_s": 5.0, "gpu": {"busy_pct": 50.0}}]}
    after = {"mode": "cli", "cases": [
        {"name": "a", "wall_s": 5.0, "throughput": 200.0, "throughput_unit": "seeds",
         "gpu": {"busy_pct": 40.0}},
        {"name": "b", "wall_s": 5.0, "gpu": {"busy_pct": 50.0}}]}
    rows = compare_results(before, after)
    check("compare 行数与 speedup", len(rows) == 2 and rows[0]["speedup"] == 2.0 and
          rows[0]["busy_delta"] == 20.0, rows)
    # 6) 渲染不崩
    doc = {"mode": "cli", "label": "selfcheck", "repo": HERE, "gpu": {"available": False},
           "idle": {}, "startup": {}, "cases": [
               {"name": "a", "desc": "x", "wall_s": 1.0, "wall_net_s": 0.5, "rc": 0,
                "throughput": 10.0, "throughput_unit": "seeds",
                "gpu": {"busy_pct": 10.0, "util_mean_pct": 8.0, "util_peak_pct": 20.0,
                        "mem_peak_mb": 100.0, "power_mean_w": 20.0}}],
           "compare": rows}
    text = render_results(doc)
    check("render_results(cli) 输出表格", "case" in text and "busy%" in text and "netbusy%" in text and "a" in text)
    doc2 = dict(doc, mode="inproc", cases=[{"name": "a", "desc": "x", "wall_s": 1.0,
                                            "kernel_ms": 100.0, "kernel_pct": 10.0,
                                            "launches": 3, "cpu_gap_ms": 900.0,
                                            "top_kernels": [{"kernel": "k", "ms": 100.0}]}])
    text2 = render_results(doc2)
    check("render_results(inproc) 输出 kernel 列", "kernel(ms)" in text2 and "launches" in text2)
    # 6b) 数字渲染与 compare raw busy
    check("_fmt_num(None/NaN/正常值)", _fmt_num(None) == "-" and _fmt_num(float("nan")) == "-"
          and _fmt_num(1.234) == "1.23")
    check("compare 使用 raw busy", rows[0]["busy_before_pct"] == 20.0 and
          rows[0]["busy_after_pct"] == 40.0 and rows[0]["busy_delta"] == 20.0, rows[0])
    # 7) --list 文本
    lst = render_case_list()
    check("--list 覆盖全部用例", all(n in lst for n in names))
    # 8) 环境信息（不要求 GPU）
    info = gpu_info()
    check("gpu_info() 可调用且字段齐全",
          isinstance(info, dict) and "available" in info and "smi" in info)
    print()
    print(f"GPU-PROFILE: {'ALL PASS' if not fails else str(len(fails)) + ' FAIL'}"
          f"（{checks[0] - len(fails)}/{checks[0]} 项）")
    return 0 if not fails else 1


def render_case_list() -> str:
    lines = [f"{'name':<24}{'kind':<8}{'quick':<7}{'work':<22}desc"]
    lines.append("-" * 96)
    for c in CASES:
        quick = "yes" if (c.get("quick_args") or c.get("quick_code")) else "-"
        work = f"{c['work']:,} {c.get('unit') or ''}".strip() if c.get("work") else "-"
        lines.append(f"{c['name']:<24}{c['kind']:<8}{quick:<7}{work:<22}{c['desc']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def select_cases(args) -> list:
    if not args.only:
        return list(CASES)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    missing = [w for w in wanted if w not in CASES_BY_NAME]
    if missing:
        raise SystemExit(f"[错误] --only 未知用例：{', '.join(missing)}；"
                         f"可用：{', '.join(CASES_BY_NAME)}")
    return [CASES_BY_NAME[w] for w in wanted]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="GPU 利用率测量/对比工具（nvidia-smi 采样 + torch.profiler）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--list", action="store_true", help="列出全部用例")
    ap.add_argument("--only", default="", help="只跑指定用例（逗号分隔，见 --list）")
    ap.add_argument("--json", action="store_true", help="stdout 输出 JSON 结果")
    ap.add_argument("--save", default="", help="把 JSON 结果写入文件")
    ap.add_argument("--compare", default="", help="与之前 --save 的 JSON 对比（before → after）")
    ap.add_argument("--repeat", type=int, default=0,
                    help="用例重复次数（默认 CLI 模式 2、--quick 1；长于 --repeat-cutoff 的用例只跑 1 次）")
    ap.add_argument("--repeat-cutoff", type=float, default=15.0,
                    help="单次 wall 超过该值不再重复（默认 15s）")
    ap.add_argument("--quick", action="store_true", help="快速/小规模用例（CI 友好）")
    ap.add_argument("--inproc", action="store_true", help="进程内 torch.profiler 模式")
    ap.add_argument("--subtract-startup", action="store_true",
                    help="测 -c pass / import 模块启动成本，输出 wall_net")
    ap.add_argument("--label", default="", help="结果标签（默认仓库目录名）")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"单条命令超时秒数（默认 {DEFAULT_TIMEOUT:.0f}）")
    ap.add_argument("--max-wait", type=float, default=60.0,
                    help="测量前等待 GPU 空闲的最长秒数（默认 60；见 --no-idle-check）")
    ap.add_argument("--no-idle-check", action="store_true", help="跳过测量前 GPU 负载检查")
    ap.add_argument("--idle-per-case", action="store_true",
                    help="每个用例开始前都等待 GPU 低负载（最终安静窗口测量用）")
    ap.add_argument("--idle-threshold", type=float, default=15.0,
                    help="低负载阈值 util%%（默认 15）")
    ap.add_argument("--background-s", type=float, default=3.0,
                    help="测量前后各采样一段空闲窗口量化背景 GPU 负载（默认 3s；"
                         "--no-background 关闭）")
    ap.add_argument("--no-background", action="store_true", help="不做背景负载采样")
    ap.add_argument("--selfcheck", action="store_true", help="运行 selfcheck（无需 GPU）")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    if args.selfcheck:
        return selfcheck()
    try:
        cases = select_cases(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.list:
        print(render_case_list())
        return 0
    if not cases:
        print("[错误] 没有可执行的用例", file=sys.stderr)
        return 2

    repeat = args.repeat if args.repeat > 0 else (1 if args.quick else 2)
    label = args.label or os.path.basename(HERE.rstrip("\\/")) or "repo"
    _info(f"[gpu_profile] 模式={'inproc' if args.inproc else 'cli'} quick={args.quick} "
          f"repeat={repeat} cases={len(cases)} label={label}")
    _info(f"[gpu_profile] 仓库={HERE} 解释器={PY}")

    idle = {"available": False, "contended": False, "waited_s": 0.0}
    if not args.no_idle_check and not args.inproc:
        idle = idle_check(args.max_wait, args.idle_threshold)
        _info(f"[gpu_profile] 负载检查: {idle}")
    startup = measure_startup(want_torch=(args.subtract_startup and not args.inproc),
                              modules=(tuple(sorted({c.get("module") for c in cases
                                                     if c.get("module")}))
                                       if args.subtract_startup else ()),
                              bootstraps=(tuple(sorted({case_bootstrap_name(c) for c in cases}))
                                          if args.subtract_startup and not args.inproc else ()))
    doc = {
        "schema": SCHEMA_VERSION,
        "tool": "gpu_profile.py",
        "version": "1.0",
        "label": label,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "inproc" if args.inproc else "cli",
        "quick": args.quick,
        "repo": HERE,
        "python": sys.version.split()[0],
        "executable": PY,
        "gpu": gpu_info(),
        "idle": idle,
        "startup": startup if args.subtract_startup else {},
        "cases": [],
    }

    bg_before = bg_after = None
    if not args.no_background and not args.inproc:
        _info("[gpu_profile] 采样背景负载（3s，不跑命令）...")
        bg_before = sample_background(args.background_s)
        _info(f"[gpu_profile] 背景负载 before: busy={bg_before.get('busy_pct')}% "
              f"util_mean={bg_before.get('util_mean_pct')}%")

    results = []
    for idx, case in enumerate(cases, 1):
        _info(f"[gpu_profile] ({idx}/{len(cases)}) {case['name']} — {case['desc']}")
        case_idle = None
        if args.idle_per_case and not args.inproc and not args.no_idle_check:
            case_idle = idle_check(args.max_wait, args.idle_threshold)
            if case_idle.get("contended"):
                _info(f"[warn] {case['name']} 测量前 GPU 仍忙（util={case_idle.get('idle_util_pct')}%），"
                      f"结果将标注 contended")
        if args.inproc:
            res = run_inproc_case(case, args.quick, repeat)
        else:
            res = run_cli_case(case, args.quick, repeat, args.timeout,
                               args.subtract_startup, startup, args.repeat_cutoff)
        if case_idle is not None:
            res["idle_at_start"] = case_idle
        results.append(res)
    if not args.no_background and not args.inproc:
        bg_after = sample_background(args.background_s)
        _info(f"[gpu_profile] 背景负载 after: busy={bg_after.get('busy_pct')}%")
        bg_vals = [b.get("busy_pct") for b in (bg_before, bg_after)
                   if b and b.get("busy_pct") is not None]
        bg_mean = round(statistics.fmean(bg_vals), 2) if bg_vals else None
        for c in results:
            g = c.get("gpu") or {}
            if g.get("busy_pct") is not None and bg_mean is not None:
                g["busy_pct_adj"] = round(max(0.0, g["busy_pct"] - bg_mean), 2)
        doc["background"] = {"before": bg_before, "after": bg_after,
                             "mean_busy_pct": bg_mean}
        if bg_mean is not None and bg_mean > 15.0:
            _info(f"[warn] 测量期间背景 GPU 占用较高（mean busy={bg_mean}%），"
                  f"busy% 已给出 busy_pct_adj 校正值；wall 仍受共享 GPU 竞争影响")
    doc["cases"] = results

    failed = [c for c in results if c.get("rc") not in (0, None) and not c.get("skipped")]
    if any((c.get("mode") == "cli" and (c.get("gpu") or {}).get("available") is False)
           for c in results) and not any((c.get("gpu") or {}).get("available")
                                         for c in results if c.get("mode") == "cli"):
        _info("[warn] nvidia-smi 采样不可用：busy%/util 字段为 None（wall/rc 仍有效）")

    if args.compare:
        try:
            with open(args.compare, encoding="utf-8") as fh:
                before = json.load(fh)
            doc["compare"] = compare_results(before, doc)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[错误] 无法读取 --compare 文件 {args.compare}: {exc}", file=sys.stderr)
            return 2

    if args.save:
        save_path = os.path.abspath(args.save)
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        _info(f"[gpu_profile] JSON 已保存：{save_path}")

    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=2))
    else:
        print(render_results(doc))
        print()
        if failed:
            print(f"GPU-PROFILE: {len(failed)} FAIL / {len(results)} 用例")
            for c in failed:
                print(f"  FAIL {c['name']} rc={c.get('rc')}  {c.get('stderr_tail', '')[:200]}")
        else:
            print(f"GPU-PROFILE: ALL PASS（{len(results)}/{len(results)} 用例）")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
