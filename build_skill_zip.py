#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_skill_zip.py — 生成/校验 skill.zip（DSH 等宿主的技能 zip 安装包）。

为什么需要它
------------
部分宿主（含 DSH 的技能上传）只接受 **zip 形式的技能安装包**：zip 内必须是
**单一顶层目录 + SKILL.md**，解压后内容落到 `<dest>/SKILL.md`。而 gpu_crack 的
skill 运行时需要工具箱本体（`gpu_crack.py` 等），因此本脚本把「skill 目录」与
「运行所需工具箱 + 文档/示例」打到同一个顶层目录 `ctf-toolkit/` 里：

    ctf-toolkit/
    ├─ SKILL.md                      ← 技能入口（frontmatter 含 name/description）
    ├─ SKILL.en.md / references/ / scripts/ / agents/
    ├─ gpu_crack.py 等工具箱运行文件  ← 与 SKILL.md 同目录，scripts/ctf.py 向上即可发现
    ├─ plugins/  examples/  docs/
    └─ README.md / CLI_REFERENCE.md / requirements.txt / LICENSE

于是安装后 `python <dest>/scripts/ctf.py ...` 无需任何环境变量即可自举；
无 CUDA/torch 时工具箱自动回落 numpy 后端。

用法
----
  python -X utf8 build_skill_zip.py              # 重新生成仓库根的 skill.zip
  python -X utf8 build_skill_zip.py --check      # 校验 zip 与源文件同步（CI；rc=0/1）
  python -X utf8 build_skill_zip.py --list       # 列出 zip 条目
  python -X utf8 build_skill_zip.py --output X.zip

确定性：条目排序 + 固定时间戳 + 固定权限位 + 文本行尾统一为 LF，因此**同一解释器**下源文件
不变时 zip 字节不变；`--check` 做**内容级**校验（条目名 + 解压后 sha256），不依赖具体 zlib
压缩实现，可在 Windows/Ubuntu、Python 3.11~3.15 之间稳定判定「skill.zip 是否与源文件同步」。

退出码：0 = 成功/同步；1 = `--check` 发现不同步或 zip 损坏；2 = 参数/源文件错误。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.join(ROOT, 'skill')
DEFAULT_ZIP = os.path.join(ROOT, 'skill.zip')

#: zip 内唯一的顶层目录（安装器会剥掉它，使内容落到 <dest>/SKILL.md）
TOP = 'ctf-toolkit'

#: 固定时间戳，保证打包确定性（与 git 无关，只影响 zip 头）
FIXED_DATE = (2026, 11, 9, 0, 0, 0)

#: skill 目录中必须随包分发的文件（相对 skill/）
SKILL_FILES = (
    'SKILL.md',
    'SKILL.en.md',
    'references/kernels.md',
    'scripts/ctf.py',
    'agents/openai.yaml',
)

#: 工具箱运行 + 质量门禁所需文件（相对仓库根；打进 zip 顶层）
TOOLKIT_FILES = (
    # 运行时（统一入口会 import 这些）
    'gpu_crack.py',
    'ctf_tools.py',
    'ctf_ext.py',
    'ctf_misc.py',
    'ctf_common.py',
    'ctf_registry.py',
    # 质量门禁（SKILL.md §7 引用）
    'selftest.py',
    'consistency.py',
    'fuzz_kernels.py',
    'verify_all.py',
    'doc_lint.py',
    'gpu_profile.py',
    # 文档与依赖
    'README.md',
    'CLI_REFERENCE.md',
    'requirements.txt',
    'LICENSE',
    'docs/bench.md',
)

#: 需要整目录收录的（相对仓库根）
TOOLKIT_DIRS = ('plugins', 'examples')

EXCLUDE_DIR_NAMES = {'__pycache__', '.git', '.venv', 'venv', 'out', '.mypy_cache', '.ruff_cache'}
EXCLUDE_SUFFIXES = ('.pyc', '.pyo', '.log', '.tmp', '.zip')

#: 打包时统一规范为 LF 的文本文件类型（后缀或完整文件名）。
#: 为什么需要：同一个提交在不同平台 checkout 到的工作区行尾可能不同（Windows 上 text=auto
#: 的文件可能是 CRLF，Linux CI 上是 LF）；若原样打包，`--check` 会在 CI 与本地结果不一致。
#: 这里统一 CRLF→LF 并去掉 UTF-8 BOM，保证「同一提交 → 同一 zip 字节」跨平台成立。
TEXT_SUFFIXES = ('.py', '.md', '.txt', '.yaml', '.yml', '.json', '.mjs', '.js', '.css', '.html')
TEXT_NAMES = {'LICENSE', 'README', 'NOTICE', 'CHANGELOG', 'AUTHORS'}


def _normalize_text_bytes(arcname: str, data: bytes) -> bytes:
    """把文本类条目的换行/BOM 规范化为跨平台稳定形式；二进制原样返回。"""
    name = os.path.basename(arcname)
    if name in TEXT_NAMES or name.lower().endswith(TEXT_SUFFIXES):
        if data.startswith(b'\xef\xbb\xbf'):
            data = data[3:]
        return data.replace(b'\r\n', b'\n')
    return data


def _collect_dir(rel_dir: str) -> list:
    """递归收集仓库内某目录下的文件（相对仓库根的路径，posix 分隔，已排序）。"""
    base = os.path.join(ROOT, rel_dir)
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIR_NAMES)
        for name in sorted(filenames):
            if name.endswith(EXCLUDE_SUFFIXES):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, ROOT).replace(os.sep, '/')
            out.append(rel)
    return out


def collect_sources() -> list:
    """返回 [(zip 内条目名, 磁盘绝对路径)]，按条目名排序。"""
    entries = []
    for rel in SKILL_FILES:
        full = os.path.join(SKILL, rel.replace('/', os.sep))
        if not os.path.isfile(full):
            raise SystemExit(f'[错误] skill 文件缺失: {full}')
        entries.append((f'{TOP}/{rel}', full))
    for rel in TOOLKIT_FILES:
        full = os.path.join(ROOT, rel.replace('/', os.sep))
        if not os.path.isfile(full):
            raise SystemExit(f'[错误] 工具箱文件缺失: {full}')
        entries.append((f'{TOP}/{rel}', full))
    for rel_dir in TOOLKIT_DIRS:
        if not os.path.isdir(os.path.join(ROOT, rel_dir)):
            raise SystemExit(f'[错误] 目录缺失: {os.path.join(ROOT, rel_dir)}')
        for rel in _collect_dir(rel_dir):
            entries.append((f'{TOP}/{rel}', os.path.join(ROOT, rel.replace('/', os.sep))))
    entries.sort(key=lambda e: e[0])
    return entries


def _zipinfo(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0                     # 与平台无关
    info.external_attr = (0o644 << 16)         # 固定权限位
    return info


def build_bytes() -> bytes:
    """按确定性规则构造 zip 字节。"""
    entries = collect_sources()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # 顶层目录条目，显式声明单一顶部目录
        zf.writestr(_zipinfo(TOP + '/'), b'')
        for arcname, path in entries:
            with open(path, 'rb') as fh:
                zf.writestr(_zipinfo(arcname), _normalize_text_bytes(arcname, fh.read()))
    return buf.getvalue()


def _zip_entries(path: str) -> dict:
    """读取现有 zip 的 {条目名: sha256}；损坏则抛 ValueError。"""
    out = {}
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f'zip 内容损坏: {bad}')
        for info in zf.infolist():
            out[info.filename] = hashlib.sha256(zf.read(info)).hexdigest()
    return out


def _new_entries(data: bytes) -> dict:
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            out[info.filename] = hashlib.sha256(zf.read(info)).hexdigest()
    return out


def check(zip_path: str) -> int:
    """校验 zip 与源文件**内容**同步（条目名 + 解压后 sha256）。

    注意这里刻意不比较 zip 的原始字节：不同 Python 版本/平台的 zlib 压缩实现可能产生
    不同的 deflate 流（同内容不同字节），CI（Ubuntu Python 3.11/3.12）与本机（Windows
    Python 3.15）因此会误报。内容级校验与压缩实现无关，跨平台/跨版本都稳定；
    同一解释器下 `--write` 仍然是字节确定性的（固定时间戳/排序/权限）。
    """
    if not os.path.isfile(zip_path):
        print(f'FAIL  {zip_path} 不存在；先运行: python -X utf8 build_skill_zip.py')
        return 1
    try:
        old = _zip_entries(zip_path)
    except (ValueError, zipfile.BadZipFile) as exc:
        print(f'FAIL  {zip_path} 损坏: {exc}')
        return 1
    new = _new_entries(build_bytes())
    if old == new:
        print(f'PASS  {os.path.basename(zip_path)} 与源文件内容同步 '
              f'({len(new) - (1 if TOP + "/" in new else 0)} 个文件)')
        return 0
    missing = sorted(set(new) - set(old))
    extra = sorted(set(old) - set(new))
    changed = sorted(n for n in set(new) & set(old) if new[n] != old[n])
    print(f'FAIL  {os.path.basename(zip_path)} 已过期，请运行: python -X utf8 build_skill_zip.py')
    for label, names in (('新增/应包含', missing), ('多余/应删除', extra), ('内容变化', changed)):
        if names:
            print(f'  {label} {len(names)} 项:')
            for n in names[:20]:
                print(f'    {n}')
            if len(names) > 20:
                print(f'    ... 其余 {len(names) - 20} 项')
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description='生成/校验 skill.zip（技能 zip 安装包）')
    ap.add_argument('--check', action='store_true', help='只校验 zip 与源文件是否同步（CI 用）')
    ap.add_argument('--list', action='store_true', help='列出 zip 内条目后退出')
    ap.add_argument('--output', default=DEFAULT_ZIP, metavar='FILE', help='输出路径（默认仓库根 skill.zip）')
    args = ap.parse_args()

    if args.check:
        return check(args.output)

    data = build_bytes()
    entries = collect_sources()
    if args.list:
        print(f'{TOP}/  ({len(entries)} 个文件, {len(data):,} 字节)')
        for arcname, _path in entries:
            print('  ' + arcname)
        return 0

    tmp = args.output + '.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, args.output)
    print(f'[skill.zip] 已生成 {args.output}')
    print(f'            顶层目录 {TOP}/，{len(entries)} 个文件，{len(data):,} 字节')
    print(f'            自检: python -X utf8 build_skill_zip.py --check')
    return 0


if __name__ == '__main__':
    sys.exit(main())
