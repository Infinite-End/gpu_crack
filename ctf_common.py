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
  - 断点文件是 JSON：{"next": int, "hits": [...], "meta": {...}}，每块落盘一次。
"""
import argparse
import json
import os
import sys
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
        sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')
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

def _load_ckpt(path, default_next):
    if not path or not os.path.exists(path):
        return default_next, []
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        nxt = int(data.get('next', default_next))
        hits = list(data.get('hits') or [])
        info(f'[resume] 从断点续跑：已完成 {nxt:,} 个候选（{path}）')
        return nxt, hits
    except Exception as exc:      # 断点损坏不应阻断任务
        info(f'[warn] 断点文件无法读取（{exc}），从头开始')
        return default_next, []


def _save_ckpt(path, nxt, hits):
    if not path:
        return
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump({'next': int(nxt), 'hits': hits}, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        info(f'[warn] 断点写入失败：{exc}')


# ---------------------------------------------------------------- 分块调度

def _is_oom(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return 'outofmemory' in name or 'out of memory' in text or 'cuda error' in text and 'memory' in text


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
    - on_hit(hits) 可返回 True 表示"得到结果，停止扫描"（早停）。
    """
    chunk = max(1, int(chunk or (1 << 16)))
    begin = int(start)
    end = begin + int(total)
    pos, hits = (begin, []) if reset else _load_ckpt(checkpoint, begin)

    if ckpt_meta is not None and checkpoint and not reset:
        pass                                           # meta 只写不读，避免覆盖用户手改

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
                chunk = max(min_chunk, step // 2)
                _free_cache()
                info(f'[oom ] 显存不足，chunk 自动降到 {chunk:,} 后重试')
                continue
            raise
        pos += step
        done += int(cnt or step)
        if got:
            hits.extend(got)
            if on_hit is not None and on_hit(got):
                stop = True
        if checkpoint:
            _save_ckpt(checkpoint, pos, hits)
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
