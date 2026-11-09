#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_crack.py 主框架自测：已知向量 + 双后端一致性 + 植入-找回 + CLI 契约。

跑法：python -X utf8 selftest.py
输出：每项 `[名称] PASS/FAIL`；末尾汇总。全部 PASS 退出码 0，否则 1。
整套约 20s（含原生 NVRTC 加速层对拍与 3 次 CLI 子进程 smoke；无 GPU 时更快）。
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np                                          # noqa: E402
import gpu_crack as G                                       # noqa: E402

RESULTS = []


def check(name, cond, detail=''):
    ok = bool(cond)
    RESULTS.append((name, ok))
    print(f'[{name}] {"PASS" if ok else "FAIL"}' + (f'  {detail}' if detail and not ok else ''))
    return ok


def run_main(argv):
    """in-process 调 CLI：返回 (rc, stdout, stderr)，并复位 --json/--quiet 状态。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = G.main(argv)
        except SystemExit as exc:                          # argparse 的退出码
            rc = exc.code
    G.C.set_output(False, False)
    return rc, out.getvalue(), err.getvalue()


def first_event(out):
    """取 stdout 里第一条 JSON 事件（--json 模式）；没有/非法返回 {}。"""
    for line in out.splitlines():
        if line.strip():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return {}
    return {}


def to_np(x):
    return np.asarray(G.from_dev(x))


def tensor(x, dtype=np.uint8):
    if G.BACKEND.startswith('torch'):
        torch = G.torch
        if not isinstance(dtype, torch.dtype):
            dtype = torch.uint8 if dtype == np.uint8 else torch.int64
        return torch.tensor(x, dtype=dtype, device=G.DEV)
    return np.asarray(x, dtype=dtype)


def use(be):
    """切后端；无 torch 时 'torch' 退回 numpy。返回实际后端名。"""
    name, _ = G.init_backend(be)
    return name


def has_torch():
    return G.torch is not None


class RefMT:
    """纯 Python std::mt19937 参考（twist=False 直接 temper 初始状态）。"""

    def __init__(self, seed):
        self.mt = [seed & 0xFFFFFFFF] + [0] * 623
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF

    @staticmethod
    def temper(y):
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def keys(self, n=24):
        return bytes(self.temper(self.mt[k]) & 0xFF for k in range(n))

    def keys_twist(self, n=24):
        mt = list(self.mt)
        for k in range(624):
            y = (mt[k] & 0x80000000) + (mt[(k + 1) % 624] & 0x7FFFFFFF)
            mt[k] = (mt[(k + 397) % 624] ^ (y >> 1)) ^ (0x9908B0DF if y & 1 else 0)
        return bytes(self.temper(mt[k]) & 0xFF for k in range(n))


# ================================================================ 1) AES 已知向量
NIST_PT = bytes.fromhex('00112233445566778899aabbccddeeff')
NIST_KEYS = {
    16: ('000102030405060708090a0b0c0d0e0f', '69c4e0d86a7b0430d8cdb78070b4c55a'),
    24: ('000102030405060708090a0b0c0d0e0f1011121314151617',
         'dda97ca4864cdfe06eaf70a0ec0d7191'),
    32: ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
         '8ea2b7ca516745bfeafc49904b496089'),
}
G.init_backend('auto')
backends = ['numpy', 'torch'] if has_torch() else ['numpy']
for be in backends:
    name = use(be)
    for kl, (khex, exp) in NIST_KEYS.items():
        blk = tensor(list(NIST_PT)).reshape(1, 16)
        kt = tensor(list(bytes.fromhex(khex)))
        got = bytes(to_np(G.aes_ecb(blk, kt))[0])
        check(f'AES-{kl*8} NIST 向量 ({name})', got.hex() == exp, got.hex())
        # 通用路径（逐候选密钥）：同一密钥铺成 2D，结果必须与 fast path 一致
        blk2 = tensor(list(NIST_PT)).reshape(1, 16)
        kt2 = tensor([list(bytes.fromhex(khex))])
        got2 = bytes(to_np(G.aes_ecb(blk2, kt2))[0])
        check(f'AES-{kl*8} 通用路径 == NIST 向量', got2.hex() == exp, got2.hex())

# S-box 破坏 (a,b)：双后端一致 + 允许 numpy 表传给 torch 后端
A_, B_ = 0x42, 0x99
bv_ = int(G.INV_SBOX[B_])
key192 = bytes.fromhex(NIST_KEYS[24][0])
blk = bytes(16)
sbox_out = {}
for be in backends:
    use(be)
    o = to_np(G.aes_ecb(tensor(list(blk)).reshape(1, 16), tensor(list(key192)), None,
                        tensor([[A_]]), tensor([[bv_]])))[0]
    sbox_out[be] = bytes(o)
if has_torch():
    check('AES-192 S-box 破坏(a,b) torch==numpy', sbox_out.get('numpy') == sbox_out.get('torch'))
    use('torch')
    o_tab = to_np(G.aes_ecb(tensor(list(blk)).reshape(1, 16), tensor(list(key192)), G.SBOX,
                            tensor([[A_]]), tensor([[bv_]])))[0]
    check('aes_ecb numpy sbox_table 在 torch 后端可用', bytes(o_tab) == sbox_out.get('torch'))

# ================================================================ 2) mt19937
seeds = [0, 1, 5489, 123456, 0xDEADBEEF, 0xFFFFFFFF, 2 ** 32, -1]
seeds_np = np.array(seeds, dtype=np.int64)
mt_ref = {}
for tw in (False, True):
    pyref = [bytes(RefMT(s).keys_twist() if tw else RefMT(s).keys()) for s in seeds]
    if has_torch():
        use('torch')
        got_t = to_np(G.mt19937_keys(G.torch.tensor(seeds, dtype=G.torch.int64, device=G.DEV),
                                     twist=tw))
    use('numpy')
    got_n = np.asarray(G.mt19937_keys(seeds_np, twist=tw))
    check(f'MT19937 twist={int(tw)} vs 纯 Python 参考',
          all(bytes(got_n[i]) == pyref[i] for i in range(len(seeds))))
    if has_torch():
        check(f'MT19937 twist={int(tw)} torch==numpy（含负种子/2^32 边界）',
              np.array_equal(got_t, got_n))

# nbytes 按需截断 == 24B 前缀；非法 nbytes 抛 ValueError
use('numpy')
full24 = np.asarray(G.mt19937_keys(seeds_np, twist=False))
ok_nb = all(
    np.array_equal(np.asarray(G.mt19937_keys(seeds_np, twist=False, nbytes=nb)),
                   full24[:, :nb]) for nb in (1, 2, 3, 4, 5, 8, 12, 23, 24))
try:
    G.mt19937_keys(seeds_np, nbytes=0)
    ok_nb = False
except ValueError:
    pass
check('MT19937 nbytes 截断 == 24B 前缀 / 越界报错', ok_nb)

# ================================================================ 3) popcount / bit_agreement
pv = np.array([0, 0xFFFFFFFF, 0xDEADBEEF, -1, -2], dtype=np.int64)
expect_pc = np.array([bin(int(v) & 0xFFFFFFFFFFFFFFFF).count('1') for v in pv], dtype=np.uint8)
res_pc = {}
for be in backends:
    use(be)
    p = to_np(G.popcount_matrix(tensor(pv, dtype=(G.torch.int64 if G.BACKEND.startswith('torch')
                                                 else np.int64)) if G.BACKEND.startswith('torch')
                                else pv))
    res_pc[be] = (p, p.dtype)
check('popcount_matrix == Python 计数且双后端 dtype 一致(uint8)',
      np.array_equal(res_pc['numpy'][0], expect_pc) and res_pc['numpy'][1] == np.uint8 and
      (not has_torch() or (np.array_equal(res_pc['torch'][0], expect_pc) and
                           res_pc['torch'][1] == np.uint8)))

rng = np.random.default_rng(20261109)
cand = rng.integers(0, 256, (37, 12), dtype=np.uint8)
tgt1 = rng.integers(0, 256, (12,), dtype=np.uint8)
tgt2 = rng.integers(0, 256, (5, 12), dtype=np.uint8)
exp1 = 8 * 12 - np.unpackbits(cand ^ tgt1[None, :], axis=1).sum(1)
exp2 = 8 * 12 - np.unpackbits(cand[:, None, :] ^ tgt2[None, :, :], axis=2).sum(2)
use('numpy')
ag_n1 = to_np(G.bit_agreement(cand, tgt1))
ag_n2 = to_np(G.bit_agreement(cand, tgt2))
empty_n = to_np(G.bit_agreement(np.zeros((0, 12), dtype=np.uint8), tgt2))
ok_ag = np.array_equal(ag_n1, exp1) and np.array_equal(ag_n2, exp2) and empty_n.shape == (0, 5)
if has_torch():
    use('torch')
    ag_t1 = to_np(G.bit_agreement(tensor(cand), tensor(tgt1)))
    ag_t2 = to_np(G.bit_agreement(tensor(cand), tensor(tgt2)))
    empty_t = to_np(G.bit_agreement(tensor(np.zeros((0, 12), dtype=np.uint8)), tensor(tgt2)))
    ok_ag &= (np.array_equal(ag_t1, exp1) and np.array_equal(ag_t2, exp2) and
              empty_t.shape == (0, 5))
check('bit_agreement 1D/2D/空候选 双后端 == 逐 bit 参考', ok_ag)


# ================================================================ 4) mtscan 植入-找回
class A:
    pass


def mtscan_args(key_hex, start, count, chunk=1024, twist=0):
    a = A()
    a.key_hex = key_hex
    a.seed_start = start
    a.seed_count = count
    a.chunk = chunk
    a.twist = twist
    a.checkpoint = ''
    a.reset_checkpoint = False
    a.no_progress = True
    return a


PLANT = 987654
use('torch')
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    G.cmd_mtscan(mtscan_args(RefMT(PLANT).keys().hex(), PLANT - 1000, 2000))
out = buf.getvalue()
check('mtscan 植入-找回 (twist=0)', '★ 命中' in out and f'seed={PLANT}' in out)
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    G.cmd_mtscan(mtscan_args(RefMT(PLANT).keys_twist().hex(), PLANT - 1000, 2000, twist=1))
out = buf.getvalue()
check('mtscan 植入-找回 (twist=1 标准 std::mt19937)', '★ 命中' in out and f'seed={PLANT}' in out)

# 1 字节目标：每 256 个种子 1 个命中，torch 曾因 nonzero() 形状只报第一个命中
hit_counts = {}
for be in backends:
    rc, out, err = run_main(['--backend', be, '--json', '--quiet', '--chunk', '65536',
                             'mtscan', '--key-hex', '00', '--seed-start', '0',
                             '--seed-count', '65536'])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    try:
        events = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError:
        events = []
    hit_counts[be] = sum(1 for e in events if e.get('event') == 'hit')
check('mtscan 批量命中枚举 双后端一致（1B 目标应 256 个/65536）',
      hit_counts.get('numpy') == 256 and (not has_torch() or hit_counts.get('torch') == 256),
      str(hit_counts))

# ================================================================ 5) sboxscan 植入-找回
keyT = bytes.fromhex(NIST_KEYS[24][0])
plain = b'gylike{12345678}'
use('torch')
ai = tensor([[A_]])
bv = tensor([[bv_]])
plant_ct = bytes(to_np(G.aes_ecb(tensor(list(plain)).reshape(1, 16), tensor(list(keyT)), None,
                                 ai, bv))[0])


def sbox_args():
    a = A()
    a.plain_hex = plain.hex()
    a.cipher_hex = plant_ct.hex()
    a.key_hex = keyT.hex()
    a.chunk = 1 << 16
    a.checkpoint = ''
    a.reset_checkpoint = False
    a.no_progress = True
    return a


for be in backends:
    name = use(be)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = G.cmd_sboxscan(sbox_args())
    out = buf.getvalue()
    check(f'sboxscan 植入-找回 ({name})',
          rc in (0, None) and f'({A_},{B_})' in out and '全块验证=通过' in out, out[:120])

# CLI JSON 契约：sboxscan stdout 只有 JSON 行（不能混入 [scan] 提示）
rc, out, err = run_main(['--backend', 'numpy', '--json', '--quiet', 'sboxscan',
                         '--plain-hex', plain.hex(), '--cipher-hex', '11' * 16,
                         '--key-hex', keyT.hex()])
lines = [ln for ln in out.splitlines() if ln.strip()]
ok_json = bool(lines)
for ln in lines:
    try:
        json.loads(ln)
    except json.JSONDecodeError:
        ok_json = False
check('sboxscan --json stdout 只有合法 JSON 行', rc == 0 and ok_json and '[scan]' not in out,
      out[:120])

# ================================================================ 6) lfsrscan 双后端一致
lfsr_outs = {}
for be in backends:
    use(be)
    a = A()
    a.taps = '25:10,11,24;28:12,22,25'
    a.bits = '1011010011'
    a.trials = 32
    a.rng_seed = 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        G.cmd_lfsrscan(a)
    lfsr_outs[be] = buf.getvalue()
check('lfsrscan 同 --rng-seed torch==numpy 输出一致',
      not has_torch() or lfsr_outs['torch'] == lfsr_outs['numpy'])

# ================================================================ 7) CLI 输入校验 / 全局参数位置
bad_cases = [
    ('mtscan 非法 hex', ['--backend', 'numpy', 'mtscan', '--key-hex', 'zz', '--seed-count', '1']),
    ('mtscan key 25B', ['--backend', 'numpy', 'mtscan', '--key-hex', '00' * 25, '--seed-count', '1']),
    ('mtscan seed-count=0', ['--backend', 'numpy', 'mtscan', '--key-hex', '00', '--seed-count', '0']),
    ('sboxscan 非法 hex', ['--backend', 'numpy', 'sboxscan', '--plain-hex', 'zz',
                           '--cipher-hex', '00', '--key-hex', '00']),
    ('sboxscan 明文长度非法', ['--backend', 'numpy', 'sboxscan', '--plain-hex', '00' * 15,
                              '--cipher-hex', '00' * 15, '--key-hex', '00' * 24]),
    ('ksscore 非法 hex', ['--backend', 'numpy', 'ksscore', '--target-hex', 'zz']),
    ('lfsr taps 非整数', ['--backend', 'numpy', 'lfsrscan', '--taps', '25:abc',
                          '--bits', '01', '--trials', '2']),
    ('lfsr rng-seed 负', ['--backend', 'numpy', 'lfsrscan', '--taps', '5:1,2',
                          '--bits', '01', '--trials', '2', '--rng-seed', '-1']),
    ('--chunk 0', ['--backend', 'numpy', '--chunk', '0', 'mtscan', '--key-hex', '00',
                   '--seed-count', '1']),
]
bad_ok = True
for name, argv in bad_cases:
    rc, out, err = run_main(argv)
    if not (rc == 2 and 'Traceback' not in out + err):
        bad_ok = False
        print(f'  (!) {name}: rc={rc} out={out[:80]!r} err={err[:120]!r}')
check('非法输入统一退出码 2 且不抛 traceback', bad_ok)

rc, out, err = run_main(['mtscan', '--key-hex', '00', '--seed-count', '1', '--backend',
                         'numpy', '--json', '--quiet'])
lines = [ln for ln in out.splitlines() if ln.strip()]
try:
    ev = json.loads(lines[0]) if len(lines) == 1 else {}
except json.JSONDecodeError:
    ev = {}
check('全局参数可写在子命令之后（--backend/--json）', rc == 0 and ev.get('event') == 'hit',
      f'rc={rc} out={out[:120]!r}')

rc, out, err = run_main(['--backend', 'numpy', '--json', 'mtscan', '--key-hex', 'zz',
                         '--seed-count', '1'])
lines = [ln for ln in out.splitlines() if ln.strip()]
try:
    ev = json.loads(lines[0]) if len(lines) == 1 else {}
except json.JSONDecodeError:
    ev = {}
check('--json 错误路径输出 JSON error 事件', rc == 2 and ev.get('event') == 'error',
      f'rc={rc} out={out[:120]!r}')

# ================================================================ 8) bench 输出契约（纯函数渲染，不跑基准）
rows = [('popcount', '900.0', 'M 个/s', '4194304 个 32bit', 0.004),
        ('mt19937', '30000.0', 'K keys/s', '65536 个种子', 0.002),
        ('aes128', '1500.0', 'K blocks/s', '16384 块', 0.011),
        ('aes192', '1400.0', 'K blocks/s', '16384 块', 0.012),
        ('aes256', '1300.0', 'K blocks/s', '16384 块', 0.013)]
md_lines = G._bench_text_lines(rows, markdown=True)
txt_lines = G._bench_text_lines(rows, markdown=False)
json_objs = G._bench_json_objs(rows)
check('bench --markdown 渲染为纯 Markdown 表格',
      len(md_lines) == 7 and all(ln.startswith('|') for ln in md_lines) and
      '| 内核 | 吞吐 | 批量 | 耗时 |' in md_lines[0])
check('bench 文本渲染保持原有 [kernel] 格式',
      len(txt_lines) == 5 and all(ln.startswith('[') for ln in txt_lines) and
      txt_lines[1].startswith('[mt19937 ]'))
check('bench --json 渲染每内核一条 JSON 对象',
      len(json_objs) == 5 and json_objs[0]['event'] == 'bench' and
      json_objs[0]['kernel'] == 'popcount' and isinstance(json_objs[0]['rate'], float))

# torch-cpu 后端不得调用 torch.cuda.*（曾导致 bench 崩溃）
if has_torch():
    old_backend = G.BACKEND
    G.BACKEND = 'torch-cpu'
    try:
        G._cuda_sync()
        sync_ok = True
    except Exception:
        sync_ok = False
    finally:
        G.BACKEND = old_backend
    check('_cuda_sync 在 torch-cpu 后端为 no-op', sync_ok)

# ================================================================ 9) 断点续跑 / 任务指纹隔离
use('numpy')
tmpdir = tempfile.mkdtemp(prefix='gpuct-')
try:
    ck = os.path.join(tmpdir, 'ck.json')
    keyA = RefMT(PLANT).keys().hex()
    keyB = RefMT(12345).keys().hex()

    def ck_argv(key, count=2000):
        return ['--backend', 'numpy', '--json', '--quiet', '--chunk', '512',
                '--checkpoint', ck, 'mtscan', '--key-hex', key,
                '--seed-start', str(PLANT - 1000), '--seed-count', str(count)]

    rc, out, err = run_main(ck_argv(keyA))
    ev = first_event(out)
    check('断点：首轮命中并写入断点',
          rc == 0 and ev.get('event') == 'hit' and ev.get('seed') == PLANT, out[:120])
    ck_data = json.load(open(ck, encoding='utf-8'))
    check('断点：文件为 v2 且带任务指纹 meta',
          ck_data.get('version') == 2 and ck_data.get('meta', {}).get('key_hex') == keyA,
          str(ck_data)[:160])
    # 同目标续跑：不再重扫，命中从断点恢复
    rc, out, err = run_main(ck_argv(keyA))
    check('断点：同目标续跑（命中从断点恢复）',
          rc == 0 and f'"seed":{PLANT}' in out, out[:160])
    # 换目标（同范围同断点）：meta 不一致 → 从头重扫，不报陈旧假命中
    rc, out, err = run_main(ck_argv(keyB))
    ev = first_event(out)
    check('断点：换目标自动隔离并从 0 重扫（不误报陈旧命中）',
          rc == 0 and ev.get('event') == 'result' and ev.get('scanned') == 2000 and
          '★' not in out, out[:160])
    # 损坏断点：告警并从头开始，不中断任务
    open(ck, 'w', encoding='utf-8').write('{not json')
    rc, out, err = run_main(ck_argv(keyB, 800))
    ev = first_event(out)
    check('断点：损坏文件告警后从头开始', rc == 0 and ev.get('scanned') == 800, out[:120])
    # 跨命令格式断点（hits 里是别的形状）不得崩溃
    open(ck, 'w', encoding='utf-8').write(json.dumps({'next': 0, 'hits': [[1, 2]]}))
    rc, out, err = run_main(['--backend', 'numpy', '--quiet', '--checkpoint', ck,
                             'mtscan', '--key-hex', keyB, '--seed-count', '4'])
    check('断点：跨命令格式断点不崩溃', rc == 0 and 'Traceback' not in out + err,
          f'rc={rc} err={err[:120]!r}')
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ================================================================ 10) 性能快速检查
use('torch')
if G.BACKEND == 'torch-cuda':
    n = 1 << 16
    s = G.torch.arange(n, dtype=G.torch.int64, device=G.DEV)
    t0 = time.time()
    G.mt19937_keys(s, twist=False)
    G.torch.cuda.synchronize()
    dt24 = time.time() - t0
    t0 = time.time()
    G.mt19937_keys(s, twist=False, nbytes=3)
    G.torch.cuda.synchronize()
    dt3 = time.time() - t0
    print(f'  [perf] mt19937 {n} seeds: 24B={n/dt24/1e6:6.1f} M/s  '
          f'3B={n/dt3/1e6:6.1f} M/s')
    check('性能：mt19937 nbytes 截断不慢于全 24B', dt3 <= dt24)
    b = G.torch.zeros((1 << 14, 16), dtype=G.torch.uint8, device=G.DEV)
    ai = G.torch.full((1 << 14, 1), A_, dtype=G.torch.uint8, device=G.DEV)
    bv = G.torch.full((1 << 14, 1), bv_, dtype=G.torch.uint8, device=G.DEV)
    G.aes_ecb(b, tensor(list(key192)), None, ai, bv)
    G.torch.cuda.synchronize()
    t0 = time.time()
    G.aes_ecb(b, tensor(list(key192)), None, ai, bv)
    G.torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'  [perf] AES-192 {b.shape[0]} blocks: {b.shape[0]/dt/1e6:6.2f} M blocks/s')
else:
    # torch-cpu / numpy 后端：不调用 torch.cuda.*（无 CUDA 时 synchronize 会抛
    # "No CUDA GPUs are available"，CI 的 CPU-only 环境必须能过），性能项跳过。
    print(f'  [perf] 跳过（{G.BACKEND}：无 CUDA；性能项只在本机 GPU 上跑）')

G.init_backend('auto')

# ================================================================ 11) 原生 NVRTC 融合内核 + 回退
# 原生层是内部加速器：cmd_mtscan/cmd_lfsrscan 在 torch-cuda 后端上自动使用（BACKEND/xp()/DEV
# 语义不变）；不可用或运行期失败时静默回退 torch/numpy 路径。这里对拍 + 回退 smoke。
_native_ok = False
try:
    _native_ok = G._nv_available()
except Exception:                                       # noqa: BLE001
    _native_ok = False
print(f'  [native] available={_native_ok}' + ('' if _native_ok else f'  ({G._NV_ERR})'))


def _lfsr_ref_best(state_bits, taps, tbits):
    """纯 Python Fibonacci 参考：返回一条初始状态的完整输出序列。"""
    st = list(state_bits)
    out = []
    for _ in range(len(tbits)):
        fb = 0
        for t in taps:
            fb ^= st[t - 1]
        out.append(fb)
        st = [fb] + st[:-1]
    return out


if _native_ok:
    use('numpy')
    rngn = np.random.default_rng(20261110)
    ok_native = True
    detail = ''
    for tw in (False, True):
        for L in (1, 2, 3, 5, 8, 16, 23, 24):
            start = int(rngn.integers(-(2 ** 62), 2 ** 62))
            count = 2048
            seeds = start + np.arange(count, dtype=np.int64)
            keys = np.asarray(G.mt19937_keys(seeds, twist=tw, nbytes=L))
            pick = int(rngn.integers(0, count))
            tgt = bytes(keys[pick].tolist())
            hits, reported = G._NV.mt_scan(start, count, L, tw, tgt)
            expected = sorted(int(seeds[i]) for i in range(count)
                              if tuple(keys[i]) == tuple(tgt))
            if hits != expected or reported != len(expected):
                ok_native = False
                detail = f'twist={int(tw)} L={L} hits={hits[:3]} expected={expected[:3]}'
                break
        if not ok_native:
            break
    check('原生融合内核 mt19937 == numpy 逐字节对拍（twist 0/1、L=1..24、负种子/边界）',
          ok_native, detail)

    # 1 字节目标：多命中必须全部返回且升序（atomicAdd 顺序需要归一化）
    keys1 = np.asarray(G.mt19937_keys(np.arange(1 << 16, dtype=np.int64),
                                      twist=False, nbytes=1))
    hits, _n = G._NV.mt_scan(0, 1 << 16, 1, False, bytes(keys1[0].tolist()))
    expected = [int(i) for i in range(1 << 16) if bytes(keys1[i]) == bytes(keys1[0])]
    check('原生融合内核 1B 目标多命中全量返回且升序',
          hits == expected and len(hits) > 64, f'{len(hits)} vs {len(expected)}')

    # LFSR：原生内核 == numpy 位打包回退 >= 纯 Python 参考
    rngl = np.random.default_rng(7)
    ok_lfsr_native = True
    detail = ''
    for n_, taps_ in ((25, [10, 11, 24]), (28, [12, 22, 25]), (5, [1, 3, 5]),
                      (63, [1, 32, 63])):
        Tlen_ = 64
        tbits = rngl.integers(0, 2, size=Tlen_, dtype=np.int64)
        states = G._lfsr_gen_packed(rngl, 256, n_)
        m_native = G._NV.lfsr_scan(states, n_, taps_, tbits)
        m_np = G._lfsr_packed_best(states, taps_, tbits)
        want = max(sum(a == b for a, b in zip(
            _lfsr_ref_best([(int(v) >> j) & 1 for j in range(n_)], taps_, tbits.tolist()),
            tbits.tolist())) for v in states[:128])
        if not (m_native == m_np and m_native >= want):
            ok_lfsr_native = False
            detail = f'n={n_} taps={taps_} native={m_native} np={m_np} ref128={want}'
            break
    check('原生 LFSR 内核 == numpy 位打包回退 >= 纯 Python 参考（n≤63）',
          ok_lfsr_native, detail)

    # 原生调用后 torch 仍然可用（primary context 混用，不串流/不破坏 torch 状态）
    use('torch')
    tv = G.torch.ones(64, device=G.DEV) + 1.0
    check('原生内核与 torch 同进程共存（primary context 混用）',
          float(tv.sum().item()) == 128.0)

# n>63 的兼容路径（逐 bit 状态矩阵，原生内核不支持，必须仍可用）
use('numpy')
a_big = A()
a_big.taps = '70:1,2,3,69'
a_big.bits = '0110100101'
a_big.trials = 64
a_big.rng_seed = 3
out_big = io.StringIO()
with contextlib.redirect_stdout(out_big), contextlib.redirect_stderr(io.StringIO()):
    rc_big = G.cmd_lfsrscan(a_big)
m_big = None
for line in out_big.getvalue().splitlines():
    if '最佳' in line and '/' in line:
        try:
            m_big = int(line.split('最佳')[1].split('/')[0].strip())
        except ValueError:
            m_big = None
check('lfsrscan n>63 兼容路径可运行且输出最佳位数',
      rc_big in (0, None) and m_big is not None and 0 <= m_big <= 10,
      out_big.getvalue()[:120])

# 强制关闭原生层：两条命令都必须回退 torch/numpy 且结果一致
_old_nv, _old_tried = G._NV, G._NV_TRIED
G._NV, G._NV_TRIED = False, True
try:
    use('torch')
    buf_fb = io.StringIO()
    with contextlib.redirect_stdout(buf_fb), contextlib.redirect_stderr(io.StringIO()):
        G.cmd_mtscan(mtscan_args(RefMT(PLANT).keys().hex(), PLANT - 1000, 2000))
    check('原生不可用时 mtscan 回退 torch 路径（植入-找回）', '★ 命中' in buf_fb.getvalue())

    lfsr_fb = {}
    for be in backends:
        name = use(be)
        a_l = A()
        a_l.taps = '25:10,11,24'
        a_l.bits = '1011010011'
        a_l.trials = 32
        a_l.rng_seed = 7
        b2 = io.StringIO()
        with contextlib.redirect_stdout(b2), contextlib.redirect_stderr(io.StringIO()):
            G.cmd_lfsrscan(a_l)
        lfsr_fb[be] = b2.getvalue()
    check('原生关闭时 lfsrscan torch 位打包回退 == numpy 输出',
          not has_torch() or lfsr_fb.get('torch') == lfsr_fb.get('numpy'))
finally:
    G._NV, G._NV_TRIED = _old_nv, _old_tried

check('auto-chunk 档位：原生 64M / torch twist0 256K / twist1 64K',
      G._native_mt_chunk(1 << 30) == (1 << 26)
      and G._torch_mt_chunk(1 << 30, False) == (1 << 18)
      and G._torch_mt_chunk(1 << 30, True) == (1 << 16))

# bench --profile 输出契约（torch-cuda 上真实跑一次；JSON 模式追加 event=profile 对象）
use('torch')
if G.BACKEND == 'torch-cuda':
    class _BA:
        markdown = False
        profile = True
    _buf = io.StringIO()
    G.C.set_output(True, True)
    try:
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(io.StringIO()):
            G.cmd_bench(_BA())
    finally:
        G.C.set_output(False, False)
    _events = []
    for _ln in _buf.getvalue().splitlines():
        if _ln.strip():
            try:
                _events.append(json.loads(_ln))
            except json.JSONDecodeError:
                _events.append({})
    _profs = [e for e in _events if e.get('event') == 'profile']
    _need = {'kernel', 'launches_per_call', 'gpu_ms_per_call', 'wall_ms_per_call', 'gpu_ratio'}
    check('bench --profile 追加 event=profile（launch 数/GPU ms/占比 字段齐全）',
          len(_profs) >= 3 and all(_need <= set(e) for e in _profs),
          f'{len(_profs)} profile events')
else:
    print('  [bench] 非 torch-cuda 后端，跳过 --profile 契约（回退路径仍可运行）')

# ---- CLI 子进程 smoke：默认 auto 路径（本机走原生融合内核；禁用时回退 torch）
import subprocess                                           # noqa: E402


def _cli(args, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    try:
        r = subprocess.run([sys.executable, '-X', 'utf8',
                            os.path.join(ROOT, 'gpu_crack.py')] + args,
                           cwd=ROOT, capture_output=True, text=True, env=env, timeout=180)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.SubprocessError) as exc:     # noqa: BLE001
        return -1, '', str(exc)


_cli_key = RefMT(PLANT).keys().hex()
_cli_base = ['--json', '--quiet', 'mtscan', '--key-hex', _cli_key,
             '--seed-start', str(PLANT - 1000), '--seed-count', '2000']
rc, out, err = _cli(_cli_base)
check('CLI auto 路径（子进程）：mtscan 植入-找回',
      rc == 0 and f'"seed":{PLANT}' in out and 'Traceback' not in err,
      f'rc={rc} out={out[:100]!r} err={err[-160:]!r}')
rc, out_off, err = _cli(_cli_base, {'GPU_CRACK_NO_NATIVE': '1'})
check('CLI auto 路径（子进程）：GPU_CRACK_NO_NATIVE=1 回退 torch 仍命中',
      rc == 0 and f'"seed":{PLANT}' in out_off, f'rc={rc} out={out_off[:100]!r}')

# lfsrscan：CLI auto（原生）与 numpy 回退在同一 --rng-seed 下结果必须一致
_cli_bits = '1011010011'
use('numpy')
_exp_best = G._lfsr_packed_best(
    G._lfsr_gen_packed(np.random.default_rng(7), 32, 25), [10, 11, 24],
    np.array([int(c) for c in _cli_bits], dtype=np.int64))
rc, out_l, err = _cli(['--json', '--quiet', 'lfsrscan', '--taps', '25:10,11,24',
                       '--bits', _cli_bits, '--trials', '32', '--rng-seed', '7'])
check('CLI auto 路径（子进程）：lfsrscan 最佳位数与 numpy 位打包一致',
      rc == 0 and f'"best":{_exp_best}' in out_l,
      f'rc={rc} expect={_exp_best} out={out_l[:120]!r}')
G.init_backend('auto')

# ================================================================ 汇总
nfail = sum(1 for _, ok in RESULTS if not ok)
print()
if nfail:
    print(f'[selftest] {nfail} 项 FAIL / 共 {len(RESULTS)} 项')
    for name, ok in RESULTS:
        if not ok:
            print(f'  - {name}')
else:
    print(f'[selftest] 全部 PASS（{len(RESULTS)} 项）')
sys.exit(0 if nfail == 0 else 1)
