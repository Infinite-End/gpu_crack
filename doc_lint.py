#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""doc_lint.py — 文档与真实 CLI 的一致性检查（改文档/改 argparse 后跑它）。

检查项：
  1. 文档里出现的每个 `--flag` 都必须是真实存在的参数（各模块 + 全局参数）；
  2. 文档里 `ctf.py <word>` 的 <word> 必须是真实子命令；
  3. CLI_REFERENCE / kernels 覆盖全部 26 个功能子命令；
  4. 文档引用的本地文件存在；
  5. 文档/代码无乱码、TODO、FIXME 等残留；
  6. 文档版本头、退出码矩阵、稳定契约等关键章节存在；
  7. 文档声称的子命令数量与实际一致；
  8. 统一入口 help 列出全部子命令。

用法：
  python -X utf8 doc_lint.py
退出码：0 = ALL PASS；1 = 有失败项。
"""
import os
import re
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
# 可选：外部 skill 源目录；设置 CTF_SKILL_SRC 后额外校验仓库内 skill/ 镜像与之逐字节一致
SRC_SKILL = os.environ.get('CTF_SKILL_SRC', '')
# 仓库内优先用 skill/ 镜像，否则回退到 CTF_SKILL_DIR
SKILL = os.environ.get('CTF_SKILL_DIR', '')
if not SKILL:
    SKILL = os.path.join(ROOT, 'skill') if os.path.isdir(os.path.join(ROOT, 'skill')) else SRC_SKILL
CTF = os.path.join(SKILL, 'scripts', 'ctf.py')
DOCS = [os.path.join(ROOT, 'README.md'),
        os.path.join(ROOT, 'CLI_REFERENCE.md'),
        os.path.join(SKILL, 'SKILL.md'),
        os.path.join(SKILL, 'references', 'kernels.md')]
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')

FAIL = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (('  |  ' + str(detail)[:500]) if (detail and not cond) else ''))
    if not cond:
        FAIL.append(name)


def run(args, timeout=300):
    r = subprocess.run([PY, '-X', 'utf8'] + args, capture_output=True, text=True,
                       timeout=timeout, encoding='utf-8', errors='replace', env=ENV)
    return r.returncode, r.stdout + r.stderr


SCRIPTS = {'gpu_crack.py': ['bench', 'mtscan', 'sboxscan', 'ksscore', 'lfsrscan'],
           'ctf_tools.py': ['selftest', 'chacha', 'poly1305', 'rs63', 'crc32', 'xorbrute', 'aesctr'],
           'ctf_ext.py': ['selftest', 'tea', 'speck', 'rc4', 'lcg', 'md5', 'rsa', 'encode',
                          'mdext', 'bm', 'zipbrute', 'eccscan'],
           'ctf_misc.py': ['selftest', 'lsb', 'stegscan', 'imgdiff', 'bitplane', 'cyclic', 'cycfind',
                           'gadgets', 'elfinfo']}
FLAG_RE = re.compile(r'--[a-zA-Z][a-zA-Z0-9-]*')

# ---------- 0. 文档存在 ----------
for p in DOCS:
    check('doc exists: ' + p, os.path.exists(p))

doc_text = {}
for p in DOCS:
    if os.path.exists(p):
        doc_text[p] = open(p, encoding='utf-8').read()
print('skill doc base:', SKILL)

# ---------- 0b. 仓库 skill/ 镜像必须与 DSH skill 源一致 ----------
MIRROR_FILES = ('SKILL.md', 'references/kernels.md', 'scripts/ctf.py')
if os.path.isdir(SRC_SKILL) and os.path.isdir(os.path.join(ROOT, 'skill')) \
        and os.path.abspath(SKILL) == os.path.abspath(os.path.join(ROOT, 'skill')):
    for rel in MIRROR_FILES:
        src = os.path.join(SRC_SKILL, rel.replace('/', os.sep))
        dst = os.path.join(ROOT, 'skill', rel.replace('/', os.sep))
        same = os.path.exists(src) and os.path.exists(dst) and \
            open(src, encoding='utf-8').read() == open(dst, encoding='utf-8').read()
        check('skill mirror in sync: ' + rel, same, 'copy %s -> skill/%s' % (src, rel))

# ---------- 1. 实际参数集合 ----------
flags = set()
for script, subs in SCRIPTS.items():
    rc, out = run([os.path.join(ROOT, script), '--help'])
    flags |= set(FLAG_RE.findall(out))
    for sub in subs:
        rc, out = run([os.path.join(ROOT, script), sub, '--help'])
        flags |= set(FLAG_RE.findall(out))
flags.add('--help')

# 追加扫描根目录脚本里直接声明的选项（consistency.py --fuzz、bench --markdown 等
# 不在子命令 SCRIPTS 里的选项也要算作"真实存在"）
import re as _re
for _py in ('gpu_crack.py', 'ctf_tools.py', 'ctf_ext.py', 'ctf_misc.py', 'consistency.py'):
    _p = os.path.join(ROOT, _py)
    if os.path.exists(_p):
        with open(_p, encoding='utf-8') as _fh:
            for _opt in _re.findall(r"add_argument\(\s*['\"]--([\w-]+)", _fh.read()):
                flags.add('--' + _opt)

print('actual flags (%d)\n' % len(flags))

# ---------- 2. 文档中的参数必须存在 ----------
for doc, text in doc_text.items():
    bad = sorted({f for f in FLAG_RE.findall(text) if f not in flags})
    check('doc flags valid: ' + os.path.basename(doc), not bad, bad)

# ---------- 3. 文档中的模块名必须存在 ----------
VALID = {s for subs in SCRIPTS.values() for s in subs} | {'selftest', 'help', 'test', 'all'}
for doc, text in doc_text.items():
    refs = set(re.findall(r'ctf\.py\s+([a-zA-Z][\w-]*)', text))
    bad = sorted(r for r in refs if r not in VALID)
    check('doc ctf.py modules valid: ' + os.path.basename(doc), not bad, bad)

# ---------- 4. 26 个功能子命令覆盖 ----------
functional = [s for subs in SCRIPTS.values() for s in subs if s != 'selftest']
cli_ref = doc_text.get(os.path.join(ROOT, 'CLI_REFERENCE.md'), '')
kern = doc_text.get(os.path.join(SKILL, 'references', 'kernels.md'), '')
missing_ref = [s for s in functional if ('`%s`' % s) not in cli_ref]
missing_kern = [s for s in functional
                if not re.search(r'(?<![\w-])%s(?![\w-])' % re.escape(s), kern)]
check('CLI_REFERENCE lists all 30 functional subcommands', not missing_ref, missing_ref)
check('kernels.md lists all 30 functional subcommands', not missing_kern, missing_kern)

# ---------- 5. 引用的本地文件存在 ----------
file_refs = set()
for text in doc_text.values():
    file_refs |= set(re.findall(
        r'skill/[\w.\-/]+|(?<![\w/])references[/\\][\w.\-]+|(?<![\w/])scripts[/\\][\w.\-]+'
        r'|(?<![\w/])(?:README|CLI_REFERENCE)\.md', text))


def _resolve_ref(ref):
    """把文档中的相对引用解析为磁盘路径（skill/* -> 仓库根，references|scripts -> skill 目录）。"""
    r = ref.replace('\\', '/')
    if r.startswith('skill/'):
        return os.path.join(ROOT, r)
    if r.startswith(('references/', 'scripts/')):
        return os.path.join(SKILL, r)
    return os.path.join(ROOT, r)


bad_files = [r for r in sorted(file_refs) if not os.path.exists(_resolve_ref(r))]
check('referenced files exist', not bad_files, bad_files)

# ---------- 6. 乱码 / TODO 残留 ----------
BAD_TOKEN = ['\ufffd', 'TODO', 'FIXME', 'TBD', '???', '？？？']
for p in DOCS + [os.path.join(ROOT, f) for f in SCRIPTS] + [CTF]:
    if not os.path.exists(p):
        continue
    text = open(p, encoding='utf-8').read()
    hits = [t for t in BAD_TOKEN if t in text]
    check('clean text: ' + os.path.basename(p), not hits, hits)

# ---------- 7. 关键章节 ----------
for p, text in doc_text.items():
    check('has version header: ' + os.path.basename(p), ('v3.0' in text) or ('v2.3' in text) or ('v2.2' in text) or ('v2.1' in text) or ('v1.0' in text))
for token in ('退出码矩阵', '稳定契约', '★', 'UTF-8'):
    check('CLI_REFERENCE contains ' + token, token in cli_ref)
check('CLI_REFERENCE count 30', '30 个功能子命令' in cli_ref)
check('README count 30', '30 个功能子命令' in doc_text.get(os.path.join(ROOT, 'README.md'), ''))

# ---------- 8. 统一入口 help ----------
rc, out = run([CTF, 'help'])
missing_help = [s for s in functional if s not in out]
check('dispatcher help lists all subcommands', not missing_help, missing_help)

print()
print('DOC-LINT:', 'ALL PASS' if not FAIL else ('%d FAIL' % len(FAIL)))
if FAIL:
    print('failed:', FAIL)
sys.exit(0 if not FAIL else 1)
