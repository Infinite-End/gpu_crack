#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_all.py — 一键跑完全部验证层（改代码后、CI、交付前使用）。

把 README 第 8 节「从下往上跑」的分散命令收敛成一条：

    python -X utf8 verify_all.py                  # 全部层（默认 --keep-going）
    python -X utf8 verify_all.py --quick          # 跳过慢层（示例 --check / 随机对拍）
    python -X utf8 verify_all.py --backend numpy  # 无 GPU 环境（把 numpy 传给各模块）
    python -X utf8 verify_all.py --list           # 只列出层名
    python -X utf8 verify_all.py --only doc,consistency   # 仅跑指定层

层级（数字越小越底层，失败时先修下面层）：
    1 doc         文档与真实 CLI 一致性（doc_lint.py）
    2 modules     三个模块已知向量自测（ctf_tools / ctf_ext / ctf_misc）
    3 framework   主框架自测（selftest.py：AES-192 NIST / mt19937 / 植入-找回）
    4 consistency 跨实现一致性固定向量（consistency.py）
    5 fuzz        随机差分模糊测试（fuzz_kernels.py --quick，可选层）
    6 dispatcher  统一入口四模块组合自测（skill/scripts/ctf.py selftest）
    7 examples    示例生成器（examples/make_examples.py [--check]）

退出码：0 = 全部 PASS/SKIP；1 = 有 FAIL；2 = 参数/环境错误。
任何一层失败都不影响后续层（--keep-going 默认开），结尾给出汇总表。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TOOLKIT_PY = os.path.join(ROOT, 'skill', 'scripts', 'ctf.py')
MAKE_EXAMPLES = os.path.join(ROOT, 'examples', 'make_examples.py')
FUZZ = os.path.join(ROOT, 'fuzz_kernels.py')
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
STEP_TIMEOUT = 900          # 单层上限（秒）；超出按 FAIL 处理


def _py(*args: str) -> list:
    return [PY, '-X', 'utf8', *args]


def _supports_check(path: str) -> bool:
    """示例生成器是否支持 --check（避免给旧版本传未知参数）。"""
    try:
        with open(path, encoding='utf-8') as fh:
            return '--check' in fh.read()
    except OSError:
        return False


def build_steps(args) -> list:
    """返回 [(层号, 名称, [命令], 说明)]；不存在的可选层自动跳过。"""
    backend = args.backend
    steps = [
        (1, 'doc', _py(os.path.join(ROOT, 'doc_lint.py')),
         '文档/CLI 一致性（doc_lint）'),
        (2, 'modules', _py(os.path.join(ROOT, 'ctf_tools.py'), '--backend', backend, 'selftest'),
         'ctf_tools 已知向量'),
        (2, 'modules', _py(os.path.join(ROOT, 'ctf_ext.py'), '--backend', backend, 'selftest'),
         'ctf_ext 已知向量'),
        (2, 'modules', _py(os.path.join(ROOT, 'ctf_misc.py'), 'selftest'),
         'ctf_misc 已知向量'),
        (3, 'framework', _py(os.path.join(ROOT, 'selftest.py')),
         '主框架自测（AES/mt19937/植入-找回；torch 缺失时自动回落 numpy）'),
        (4, 'consistency', _py(os.path.join(ROOT, 'consistency.py')),
         '跨实现一致性固定向量'),
    ]
    if os.path.isfile(FUZZ):
        steps.append((5, 'fuzz', _py(FUZZ, '--quick'), '随机差分模糊测试（--quick）'))
    steps.append((6, 'dispatcher', _py(TOOLKIT_PY, 'selftest'), '统一入口组合自测'))
    if os.path.isfile(MAKE_EXAMPLES):
        if _supports_check(MAKE_EXAMPLES) and not args.quick:
            cmd = _py(MAKE_EXAMPLES, '--check')
            desc = '示例生成器（--check 逐条复跑）'
        else:
            cmd = _py(MAKE_EXAMPLES)
            desc = '示例生成器（生成 + 自检）'
        steps.append((6, 'examples', cmd, desc))       # 与 dispatcher 同属第 6 层「组合自测 + 示例」
    return steps


def _run(cmd_list: list) -> tuple:
    """跑一条命令，返回 (rc, 耗时, stdout+stderr)。超时不算崩溃，按 FAIL 处理。"""
    t0 = time.time()
    try:
        r = subprocess.run(cmd_list, cwd=ROOT, env=ENV, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=STEP_TIMEOUT)
        return r.returncode, time.time() - t0, (r.stdout or '') + (r.stderr or '')
    except subprocess.TimeoutExpired as exc:
        out = b''
        if isinstance(exc.output, bytes):
            out = exc.output
        return -9, time.time() - t0, out.decode('utf-8', 'replace') + \
            f'\n[timeout] 超过 {STEP_TIMEOUT}s'
    except OSError as exc:
        return -1, time.time() - t0, f'[无法启动] {exc}'


def _summarize(output: str) -> str:
    """从输出里挑结果行（PASS/FAIL/错误/命中），没有则取末尾若干行。"""
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    keep = [ln for ln in lines
            if any(tok in ln for tok in ('PASS', 'FAIL', 'ALL PASS', '通过', '[错误]', 'Traceback',
                                         'Error', 'error: ', '未命中', '★'))]
    if not keep:
        keep = lines[-6:]
    if len(keep) > 12:
        keep = keep[:6] + ['  ...'] + keep[-5:]
    return '\n'.join('    ' + ln for ln in keep)


def main() -> int:
    ap = argparse.ArgumentParser(description='gpu_crack 全验证层一键执行器')
    ap.add_argument('--backend', default='auto', choices=['auto', 'torch', 'numpy'],
                    help='传给各模块的 --backend（无 GPU 环境用 numpy）')
    ap.add_argument('--quick', action='store_true',
                    help='快速模式：跳过示例逐条复跑等慢层')
    ap.add_argument('--list', action='store_true', help='只列出各层，不执行')
    ap.add_argument('--only', default='', help='只跑名字匹配的层，逗号分隔（如 doc,consistency）')
    ap.add_argument('--json', action='store_true', help='结束时额外输出单行 JSON 摘要')
    args = ap.parse_args()

    steps = build_steps(args)
    if args.only:
        wanted = {s.strip() for s in args.only.split(',') if s.strip()}
        steps = [s for s in steps if s[1] in wanted]
        if not steps:
            print('[错误] --only 未匹配任何层，可用：doc,modules,framework,consistency,'
                  'fuzz,dispatcher,examples', file=sys.stderr)
            return 2
    if args.list:
        for num, name, _cmd, desc in steps:
            print(f'  {num}. {name:<12} {desc}')
        return 0
    if not steps:
        print('[错误] 没有可执行的验证层', file=sys.stderr)
        return 2

    n_layers = len({num for num, _n, _c, _d in steps})
    print(f'verify_all: {len(steps)} 个执行步骤（{n_layers} 个验证层）  backend={args.backend}')
    print(f'            解释器={PY}')
    print(f'            工具箱={ROOT}\n')
    results = []
    for num, name, cmd, desc in steps:
        print(f'—— [{num}] {name} — {desc}', flush=True)
        rc, sec, out = _run(cmd)
        status = 'PASS' if rc == 0 else 'FAIL'
        results.append({'layer': num, 'name': name, 'rc': rc,
                        'seconds': round(sec, 2), 'status': status})
        print(f'   {status}  rc={rc}  {sec:.2f}s')
        if rc != 0:
            print(_summarize(out))
        print(flush=True)

    failed = [r for r in results if r['status'] == 'FAIL']
    n_layers = len({r['layer'] for r in results})
    print('=' * 62)
    for r in results:
        print(f"  {r['status']:<4} {r['name']:<12} {r['seconds']:8.2f}s  rc={r['rc']}")
    print('=' * 62)
    print(f"VERIFY-ALL: {'ALL PASS' if not failed else str(len(failed)) + ' FAIL'}"
          f"  ({len(results) - len(failed)}/{len(results)} 个步骤通过，覆盖 {n_layers} 个验证层)")
    if args.json:
        print(json.dumps({'result': 'pass' if not failed else 'fail',
                          'steps': results}, ensure_ascii=False, separators=(',', ':')))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
