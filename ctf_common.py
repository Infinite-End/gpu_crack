#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ctf_common.py — gpu_crack / ctf_tools / ctf_ext 共用的小工具。

提供三件事：
  1. chunked()   统一分块调度：CUDA OOM 自动减半重试 + 进度/ETA + 断点续跑；
  2. emit()      统一输出：人读文本或单行 JSON（--json）；
  3. add_output_args() / add_sched_args()  统一参数定义，避免各脚本各写一套。

设计约束：
  - 结果只走 stdout，进度/警告只走 stderr（管道里可分离）；
  - OOM 判定不依赖 torch：按异常类型名与消息文本识别，torch 不存在时也不会报错；
  - 断点文件是 JSON：{"version": 2, "next": int, "hits": [...], "meta": {...}}，每块落盘一次；
    meta 记录任务指纹（label/start/total + 调用方补充项），共用同一断点文件的不同任务不会
    互相串用进度（meta 不一致自动从头开始）。

自检（不需要 GPU）：
  python ctf_common.py        # 分块边界/OOM 降档/断点/输出契约的单元自检
"""
import argparse
import json
import os
import sys
import tempfile
import time

# ---------------------------------------------------------------- 输出

_JSON = False
_QUIET = False


def set_output(json_mode=False, quiet=False):
    global _JSON, _QUIET
    _JSON = bool(json_mode)
    _QUIET = bool(quiet)


def json_mode():
    return _JSON


def quiet():
    return _QUIET


def emit(obj):
    """输出一条结果。--json 时写单行 JSON 到 stdout，否则写人读文本。

    obj 形如 {"event": "hit", "seed": 123, "key": "ab.."}；
    文本模式只打印 obj["text"]（若给），否则按 key=value 拼接。
    """
    if _JSON:
        try:
            line = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            # 内核偶尔带不可序列化对象（numpy 标量等）：退化为字符串形式，不因输出挂掉
            line = json.dumps(obj, ensure_ascii=True, separators=(',', ':'), default=str)
        try:
            sys.stdout.write(line + '\n')
        except UnicodeEncodeError:                     # 控制台编码装不下中文/符号时退化为 ASCII
            sys.stdout.write(json.dumps(obj, ensure_ascii=True, separators=(',', ':'),
                                        default=str) + '\n')
        sys.stdout.flush()
        return
    text = obj.get('text')
    if text is None:
        text = '  '.join(f'{k}={v}' for k, v in obj.items() if k not in ('event', 'text'))
    print(text, flush=True)


def info(msg):
    """非结果信息（横幅/进度/统计）→ stderr，--quiet 时静默。"""
    if not _QUIET:
        print(msg, file=sys.stderr, flush=True)


def fail(msg, code=2):
    """错误 → stderr（--json 时同时给一条 JSON 错误事件），返回 code。"""
    print(f'[错误] {msg}', file=sys.stderr, flush=True)
    if _JSON:
        emit({'event': 'error', 'error': msg})
    return code


# ---------------------------------------------------------------- 参数

def add_output_args(p):
    """输出类通用参数：--json / --quiet。"""
    p.add_argument('--json', action='store_true',
                   help='输出单行 JSON（机器可读，便于脚本/agent 解析）')
    p.add_argument('--quiet', action='store_true',
                   help='抑制横幅与进度，只保留结果行')
    return p


def add_sched_args(p):
    """调度类通用参数：--checkpoint / --reset-checkpoint / --no-progress。"""
    p.add_argument('--checkpoint', default='', metavar='FILE',
                   help='断点文件；每块落盘一次，中断后从上次位置续跑')
    p.add_argument('--reset-checkpoint', action='store_true',
                   help='忽略已有断点，从头开始（并覆盖断点文件）')
    p.add_argument('--no-progress', action='store_true',
                   help='不打印进度/ETA')
    return p


def apply_output_args(a):
    """在 main() 里调用：把解析结果落到模块状态。"""
    set_output(json_mode=getattr(a, 'json', False), quiet=getattr(a, 'quiet', False))


# ---------------------------------------------------------------- 断点

CKPT_VERSION = 2
# 断点写入失败只告警一次（避免每块刷屏）；进程内按路径去重
_CKPT_WARNED = set()


def _json_safe(obj):
    """把任意对象转成可 JSON 序列化且可稳定比较的结构（用于 meta 指纹）。"""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return json.loads(json.dumps(str(obj), ensure_ascii=False))


def _build_meta(begin, total, label, ckpt_meta):
    meta = {'label': str(label or ''), 'start': int(begin), 'total': int(total)}
    if isinstance(ckpt_meta, dict):
        meta.update({str(k): v for k, v in ckpt_meta.items()})
    return _json_safe(meta)


def _load_ckpt(path, default_next, meta=None, begin=None, end=None):
    """读断点，返回 (next, hits)；meta 不一致 / 越界 / 损坏时报警并从默认位置开始。"""
    if not path or not os.path.exists(path):
        return default_next, []
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception as exc:      # 断点损坏不应阻断任务
        info(f'[warn] 断点文件无法读取（{exc}），从头开始')
        return default_next, []
    if not isinstance(data, dict):
        info('[warn] 断点文件格式不对（不是 JSON 对象），从头开始')
        return default_next, []
    try:
        nxt = int(data.get('next', default_next))
    except (TypeError, ValueError):
        info('[warn] 断点文件 next 字段非法，从头开始')
        return default_next, []

    stored_meta = data.get('meta')
    if meta is not None and stored_meta is not None:
        if _json_safe(stored_meta) != meta:
            info('[warn] 断点属于其它任务（meta 不一致），新建断点从头开始')
            return default_next, []
    elif meta is not None:
        info('[warn] 断点缺少任务标识（旧格式），按位置续跑；建议加 --reset-checkpoint 重来')

    if begin is not None and end is not None and not (begin <= nxt <= end):
        info(f'[warn] 断点位置 {nxt} 不在本次扫描范围 [{begin}, {end})，从头开始')
        return default_next, []

    hits = data.get('hits')
    if not isinstance(hits, list):
        hits = []
    info(f'[resume] 从断点续跑：已完成 {nxt:,} 个候选（{path}）')
    return nxt, hits


def _save_ckpt(path, nxt, hits, meta=None):
    if not path:
        return
    tmp = ''
    try:
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        payload = {'version': CKPT_VERSION, 'next': int(nxt), 'hits': list(hits)}
        if meta is not None:
            payload['meta'] = meta
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            text = json.dumps(payload, ensure_ascii=False, default=str)
        fd, tmp = tempfile.mkstemp(dir=d or '.', prefix='.ctf_ckpt.', suffix='.tmp')
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, path)
        tmp = ''
    except Exception as exc:
        key = os.path.abspath(path)
        if key not in _CKPT_WARNED:
            _CKPT_WARNED.add(key)
            info(f'[warn] 断点写入失败（{exc}）；本次任务继续，但无法续跑')
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------- 分块调度

def _is_oom(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return ('outofmemory' in name or 'out of memory' in text
            or ('cuda error' in text and 'memory' in text))


def _free_cache():
    try:
        import torch                                  # noqa: F401
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def chunked(fn, total, start=0, chunk=None, *, label='', checkpoint='', reset=False,  # noqa: C901
            progress=True, min_chunk=256, on_hit=None, ckpt_meta=None):
    """分块执行 fn(lo, hi) -> (count, hits)；返回 (done, hits, seconds)。

    - CUDA OOM：自动 chunk //= 2 并重试同一区间（不丢进度，不报错退出）；
    - 进度：约每秒一次 `[progress] 12.3% 4.2 M/s ETA 42s` 打到 stderr；
    - 断点：每完成一块写一次；启动时若断点存在且未 --reset-checkpoint，则从 next 续跑；
      meta 指纹（label/start/total + ckpt_meta）不一致、或 next 越界时自动从头开始，
      避免不同任务共用同一个断点文件时互相串用进度；
    - on_hit(hits) 可返回 True 表示"得到结果，停止扫描"（早停）。
    """
    chunk = max(1, int(chunk or (1 << 16)))
    begin = int(start)
    end = begin + int(total)
    meta = _build_meta(begin, total, label, ckpt_meta)
    if reset:
        pos, hits = begin, []
    else:
        pos, hits = _load_ckpt(checkpoint, begin, meta=meta if checkpoint else None,
                               begin=begin, end=end)

    done = 0
    t0 = time.time()
    last = 0.0
    stop = False
    while pos < end:
        step = min(chunk, end - pos)
        try:
            cnt, got = fn(pos, pos + step)
        except Exception as exc:
            if _is_oom(exc) and step > min_chunk:
                new_chunk = max(min_chunk, step // 2)
                info(f'[oom ] 显存不足：chunk {step:,} -> {new_chunk:,} 后重试')
                chunk = new_chunk
                _free_cache()
                continue
            raise
        pos += step
        # fn 返回的 count 优先（允许内核报告"本块实际处理 0 个"）；None 才回退到步长
        done += step if cnt is None else int(cnt)
        if got:
            hits.extend(got)
            if on_hit is not None and on_hit(got):
                stop = True
        if checkpoint:
            _save_ckpt(checkpoint, pos, hits, meta)
        now = time.time()
        if progress and not quiet() and (now - last >= 1.0 or pos >= end):
            last = now
            el = max(now - t0, 1e-9)
            rate = done / el if done else 0.0
            frac = (pos - begin) / max(end - begin, 1)
            eta = (end - pos) / rate if rate > 0 else 0
            lag = '' if not label else f' {label}'
            print(f'[progress]{lag} {frac*100:5.1f}%  {rate/1e6:.2f} M/s  ETA {eta:,.0f}s',
                  file=sys.stderr, flush=True)
        if stop:
            break
    return done, hits, time.time() - t0


def add_bench_args(p):
    p.add_argument('--markdown', action='store_true',
                   help='以 Markdown 表格输出吞吐数据（可直接写入 docs/bench.md）')
    return p


def arg_parser_or_none():
    return argparse


# ---------------------------------------------------------------- 自检

def _selfcheck() -> int:
    """`python ctf_common.py`：分块/OOM/断点/输出契约自检（无 GPU、无网络）。"""
    import contextlib
    import io
    import shutil
    import tempfile

    fails = []

    def check(name, cond, detail=''):
        print(('PASS ' if cond else 'FAIL ') + name
              + (('  |  ' + str(detail)[:300]) if detail and not cond else ''))
        if not cond:
            fails.append(name)

    class OutOfMemoryError(Exception):        # 与 torch.cuda.OutOfMemoryError 同名
        pass

    class Boom(Exception):
        pass

    tmp = tempfile.mkdtemp(prefix='ctf_common_check_')
    old_json, old_quiet = _JSON, _QUIET
    # 自检不要真的 import torch（_free_cache 会触发 2~3s 的 torch 导入）；换成计数器，
    # 同时验证 OOM 重试路径每次都会尝试释放显存。
    free_calls = {'n': 0}
    old_free_cache = _free_cache
    globals()['_free_cache'] = lambda: free_calls.__setitem__('n', free_calls['n'] + 1)
    try:
        # ---- 1. 分块边界 ----
        calls = []
        done, hits, _dt = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, []))[1],
                                  10, chunk=100)
        check('chunk > total -> single block', done == 10 and calls == [(0, 10)], (done, calls))
        calls.clear()
        done, _, _ = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, []))[1], 5, chunk=1)
        check('chunk=1', done == 5 and len(calls) == 5, (done, calls))
        calls.clear()
        done, hits, _ = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, []))[1], 0)
        check('total=0 -> no calls', done == 0 and calls == [] and hits == [], (done, calls))
        calls.clear()
        done, hits, _ = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, [lo]))[1],
                                10, start=1000, chunk=4)
        check('start offset respected',
              calls == [(1000, 1004), (1004, 1008), (1008, 1010)] and done == 10
              and hits == [1000, 1004, 1008], (done, calls, hits))
        calls.clear()
        done, _, _ = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, []))[1], 7, chunk=-3)
        check('non-positive chunk coerced to 1', done == 7 and len(calls) == 7, (done, calls))
        done, _, _ = chunked(lambda lo, hi: (hi - lo, []), -5)
        check('negative total -> no-op', done == 0, done)

        # ---- 2. OOM 自动降档 ----
        oom_calls = []
        ok_ranges = []

        def fn_oom(lo, hi):
            oom_calls.append((lo, hi))
            if hi - lo > 40:
                raise OutOfMemoryError('CUDA out of memory')
            ok_ranges.append((lo, hi))
            return hi - lo, []

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            done, _, _ = chunked(fn_oom, 200, chunk=100, min_chunk=8)
        info_lines = [ln for ln in err.getvalue().splitlines() if '[oom ]' in ln]
        check('OOM halving: done all + successful blocks <= 40',
              done == 200 and ok_ranges == [(0, 25), (25, 50), (50, 75), (75, 100),
                                            (100, 125), (125, 150), (150, 175), (175, 200)],
              (done, ok_ranges))
        check('OOM halving: 2 notices (100->50->25)', len(info_lines) == 2, info_lines)
        check('OOM retry calls free_cache()', free_calls['n'] >= 2, free_calls['n'])

        def fn_msg_oom(lo, hi):
            if hi - lo > 16:
                raise RuntimeError('CUDA error: out of memory (synthetic)')
            return hi - lo, []

        done, _, _ = chunked(fn_msg_oom, 100, chunk=64, min_chunk=4)
        check('OOM detected from message text', done == 100, done)

        def fn_always_oom(lo, hi):
            raise OutOfMemoryError('always')

        raised = False
        try:
            chunked(fn_always_oom, 100, chunk=64, min_chunk=8)
        except OutOfMemoryError:
            raised = True
        check('OOM at min_chunk re-raised', raised)

        raised = None
        try:
            chunked(lambda lo, hi: (_ for _ in ()).throw(Boom('not oom')), 100, chunk=10)
        except Boom as exc:
            raised = exc
        check('non-OOM exception propagates unchanged', isinstance(raised, Boom))

        # ---- 3. on_hit 早停与计数 ----
        calls.clear()
        stop_calls = {'n': 0}

        def fn_hit(lo, hi):
            calls.append((lo, hi))
            return hi - lo, ([lo + 1] if lo == 20 else [])

        def on_hit(got):
            stop_calls['n'] += 1
            return True

        done, hits, _ = chunked(fn_hit, 100, chunk=10, on_hit=on_hit)
        check('on_hit early stop', done == 30 and hits == [21] and len(calls) == 3
              and stop_calls['n'] == 1, (done, hits, calls))
        done, hits, _ = chunked(fn_hit, 100, chunk=10, on_hit=lambda got: False)
        check('on_hit False continues', done == 100 and hits == [21], (done, hits))
        done, _, _ = chunked(lambda lo, hi: (0, []), 30, chunk=10)
        check('done honors count=0 from fn', done == 0, done)
        done, _, _ = chunked(lambda lo, hi: (None, []), 30, chunk=10)
        check('done falls back to step when count=None', done == 30, done)

        # ---- 4. 断点续跑 ----
        ck = os.path.join(tmp, 'ck.json')
        interrupted = {'armed': True}
        run_calls = []

        def fn_ck(lo, hi):
            run_calls.append((lo, hi))
            if lo == 30 and interrupted['armed']:
                interrupted['armed'] = False
                raise Boom('interrupt')
            return hi - lo, ([lo] if lo == 10 else [])

        try:
            chunked(fn_ck, 100, chunk=10, checkpoint=ck, label='taskA')
        except Boom:
            pass
        data = json.load(open(ck, encoding='utf-8'))
        check('checkpoint saved next/hits/meta',
              data.get('next') == 30 and data.get('hits') == [10]
              and data.get('meta', {}).get('label') == 'taskA', data)
        run_calls.clear()
        done, hits, _ = chunked(fn_ck, 100, chunk=10, checkpoint=ck, label='taskA')
        check('checkpoint resume', done == 70 and hits == [10] and run_calls[0] == (30, 40),
              (done, hits, run_calls[:2]))
        run_calls.clear()
        done, hits, _ = chunked(fn_ck, 100, chunk=10, checkpoint=ck, reset=True, label='taskA')
        check('reset-checkpoint starts over', done == 100 and hits == [10], (done, hits))

        # 不同任务共用同一断点文件：meta 不一致必须从头开始，不能直接续到 end
        run_calls.clear()
        done, hits, _ = chunked(lambda lo, hi: (run_calls.append((lo, hi)), (hi - lo, []))[1],
                                100, chunk=10, checkpoint=ck, label='taskB')
        check('checkpoint meta isolation (different task restarts)',
              run_calls and run_calls[0] == (0, 10) and done == 100, (run_calls[:2], done))
        data = json.load(open(ck, encoding='utf-8'))
        check('checkpoint meta rewritten for new task', data.get('meta', {}).get('label') == 'taskB',
              data.get('meta'))

        # ckpt_meta 补充项也参与隔离
        ck2 = os.path.join(tmp, 'ck2.json')
        chunked(lambda lo, hi: (hi - lo, []), 100, chunk=10, checkpoint=ck2,
                ckpt_meta={'key_hex': 'aa'})
        calls.clear()
        chunked(lambda lo, hi: (hi - lo, []), 100, chunk=10, checkpoint=ck2,
                ckpt_meta={'key_hex': 'bb'})
        data = json.load(open(ck2, encoding='utf-8'))
        check('ckpt_meta mismatch restarts', data.get('meta', {}).get('key_hex') == 'bb',
              data.get('meta'))

        # 断点位置越界 -> 从头开始
        ck3 = os.path.join(tmp, 'ck3.json')
        with open(ck3, 'w', encoding='utf-8') as fh:
            json.dump({'version': 2, 'next': 500, 'hits': []}, fh)
        calls.clear()
        done, _, _ = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, []))[1], 30,
                             chunk=10, checkpoint=ck3)
        check('out-of-range checkpoint restarts', done == 30 and calls[0] == (0, 10),
              (done, calls[:2]))

        # 旧格式（无 meta）容忍续跑
        ck4 = os.path.join(tmp, 'ck4.json')
        with open(ck4, 'w', encoding='utf-8') as fh:
            json.dump({'next': 20, 'hits': [1]}, fh)
        calls.clear()
        done, hits, _ = chunked(lambda lo, hi: (calls.append((lo, hi)), (hi - lo, []))[1], 30,
                                chunk=10, checkpoint=ck4)
        check('legacy checkpoint (no meta) resumes', done == 10 and calls == [(20, 30)]
              and hits == [1], (done, calls, hits))

        # 损坏断点
        ck5 = os.path.join(tmp, 'ck5.json')
        with open(ck5, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            done, _, _ = chunked(lambda lo, hi: (hi - lo, []), 20, chunk=10, checkpoint=ck5)
        check('corrupted checkpoint tolerated', done == 20 and '断点' in err.getvalue(),
              (done, err.getvalue()[:120]))

        # 断点目录不存在：自动创建（父目录）
        ck6 = os.path.join(tmp, 'sub', 'dir', 'ck.json')
        done, _, _ = chunked(lambda lo, hi: (hi - lo, []), 20, chunk=10, checkpoint=ck6)
        check('checkpoint parent dir auto-created', done == 20 and os.path.isfile(ck6),
              os.path.exists(ck6))

        # 断点无法写入：告警只出现一次，任务继续
        blocker = os.path.join(tmp, 'afile')
        with open(blocker, 'w', encoding='utf-8') as fh:
            fh.write('x')
        bad = os.path.join(blocker, 'ck.json')          # 父路径是文件 -> 无法写入
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            done, _, _ = chunked(lambda lo, hi: (hi - lo, []), 50, chunk=5, checkpoint=bad)
        n_warn = err.getvalue().count('断点写入失败')
        check('unwritable checkpoint warns once and continues', done == 50 and n_warn == 1,
              (done, n_warn, err.getvalue()[:160]))

        # ---- 5. 进度输出 ----
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            chunked(lambda lo, hi: (hi - lo, []), 200000, chunk=10000, label='probe')
        txt = err.getvalue()
        check('progress line format', '[progress] probe' in txt and '%' in txt and 'ETA' in txt,
              txt[:160])
        set_output(json_mode=False, quiet=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            chunked(lambda lo, hi: (hi - lo, []), 100000, chunk=10000, label='q')
        check('quiet suppresses progress', '[progress]' not in err.getvalue(), err.getvalue()[:120])
        set_output(json_mode=False, quiet=False)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            chunked(lambda lo, hi: (hi - lo, []), 100000, chunk=10000, label='q', progress=False)
        check('progress=False suppresses progress', '[progress]' not in err.getvalue())

        # ---- 6. 输出契约 ----
        set_output(json_mode=True, quiet=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            emit({'event': 'hit', 'msg': '中文', 'n': 1})
        line = buf.getvalue().strip()
        parsed = json.loads(line) if line else {}
        check('emit json: single line, no ascii escaping',
              line.count('\n') == 0 and '中文' in line and parsed.get('event') == 'hit', line)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            emit({'event': 'hit', 'bad': object()})
        parsed = json.loads(buf.getvalue().strip())
        check('emit json: unserializable value tolerated',
              isinstance(parsed, dict) and parsed.get('event') == 'hit', buf.getvalue()[:120])
        set_output(json_mode=False, quiet=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            emit({'event': 'hit', 'msg': '中文', 'text': '人类文本'})
        check('emit text: uses text field', buf.getvalue().strip() == '人类文本', buf.getvalue())
        set_output(json_mode=True, quiet=False)
        out, errs = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errs):
            rc = fail('坏参数', 2)
        lines = out.getvalue().strip().splitlines()
        check('fail(): stderr + single JSON error event, returns code',
              rc == 2 and len(lines) == 1 and json.loads(lines[0]).get('event') == 'error'
              and '坏参数' in errs.getvalue(), (rc, lines))
        set_output(json_mode=False, quiet=True)
        errs = io.StringIO()
        with contextlib.redirect_stderr(errs):
            info('should be hidden')
        check('quiet suppresses info()', errs.getvalue() == '', errs.getvalue())
    finally:
        globals()['_free_cache'] = old_free_cache
        set_output(json_mode=old_json, quiet=old_quiet)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print('CTF-COMMON:', 'ALL PASS' if not fails else f'{len(fails)} FAIL')
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(_selfcheck())
