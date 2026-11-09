#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""doc_lint.py — 文档与真实 CLI 的一致性检查（改文档/改 argparse 后跑它）。

检查项：
  1. 文档里出现的每个 `--flag` 都必须是真实存在的参数（各模块 + 全局参数）；
  2. 文档里 `ctf.py <word>` 的 <word> 必须是真实子命令；
  3. CLI_REFERENCE / kernels 覆盖全部功能子命令（数量从注册表派生）；
  4. 文档引用的本地文件存在；
  5. 文档/代码无乱码、TODO、FIXME 等残留；
  6. 文档版本头存在，且版本号 == 单一真源 `ctf_registry.VERSION`；
  7. 文档声称的子命令数量与实际一致（数量从注册表派生，不再写死 30）；
  8. 统一入口 help 列出全部子命令；
  9. 部署回退表一致性：`skill/scripts/ctf.py` 的 `_EMBEDDED_SCRIPTS` / `_EMBEDDED_GLOBALS`
     必须与 `ctf_registry` 的真源逐项一致（否则 skill 单独部署会静默少/多子命令）；
 10. CLI_REFERENCE 的全局参数矩阵（--backend/--chunk/--json/--checkpoint）列出的模块
     必须与注册表能力位一致；
 11. 示例插件 morse 在文档中被提及；
 12. 新增的验证入口（verify_all.py / fuzz_kernels.py）存在时，文档必须引用。

版本真源：ctf_registry.VERSION（本文件与 README/CLI_REFERENCE/SKILL/kernels.md 都读它，
不要在任何地方再写死版本号）。

用法：
  python -X utf8 doc_lint.py
退出码：0 = ALL PASS；1 = 有失败项。
"""
import ast
import glob
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
CTF_PY_SRC = os.path.join(ROOT, 'ctf_registry.py')

FAIL = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (('  |  ' + str(detail)[:500]) if (detail and not cond) else ''))
    if not cond:
        FAIL.append(name)


def run(args, timeout=300):
    r = subprocess.run([PY, '-X', 'utf8'] + args, capture_output=True, text=True,
                       timeout=timeout, encoding='utf-8', errors='replace', env=ENV)
    return r.returncode, r.stdout + r.stderr


def _literal_assign(path, name):
    """静态读取一个脚本里的模块级 `NAME = <字面量>` 赋值（不执行该脚本）。"""
    try:
        with open(path, encoding='utf-8') as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError, TypeError):
                    return None
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
              and node.target.id == name and node.value is not None):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                return None
    return None


# ---------------------------------------------------------------- 版本真源
VERSION = ''
REG = None
try:
    sys.path.insert(0, ROOT)
    import ctf_registry as _REG
    REG = _REG
    VERSION = str(getattr(_REG, 'VERSION', '') or '')
except Exception as exc:                              # noqa: BLE001
    print(f'[warn] 无法导入 ctf_registry（版本真源）: {exc}')

check('version source available (ctf_registry.VERSION)',
      bool(VERSION) and bool(re.fullmatch(r'\d+\.\d+', VERSION)), VERSION)
print('version source: ctf_registry.VERSION =', VERSION or '（缺失）')

# 子命令清单的真源在 ctf_registry.py；这里按脚本反查，避免两处名单漂移
try:
    SCRIPTS = {script: ['selftest'] + list(names)
               for script, names in REG.BY_SCRIPT.items()}
except Exception:                                     # 注册表缺失时退回内置表
    SCRIPTS = {'gpu_crack.py': ['bench', 'mtscan', 'sboxscan', 'ksscore', 'lfsrscan'],
               'ctf_tools.py': ['selftest', 'chacha', 'poly1305', 'rs63', 'crc32', 'xorbrute', 'aesctr'],
               'ctf_ext.py': ['selftest', 'tea', 'speck', 'rc4', 'lcg', 'md5', 'rsa', 'encode',
                              'mdext', 'bm', 'zipbrute', 'eccscan'],
               'ctf_misc.py': ['selftest', 'lsb', 'stegscan', 'imgdiff', 'bitplane', 'cyclic',
                               'cycfind', 'gadgets', 'elfinfo']}
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

# 追加扫描仓库脚本里直接声明的选项（consistency.py --fuzz / bench --markdown /
# verify_all.py --quick / examples --check 等不在子命令表里的选项也要算作"真实存在"）
_scan_files = sorted(glob.glob(os.path.join(ROOT, '*.py')))
_scan_files += [os.path.join(ROOT, 'examples', 'make_examples.py')]
for _py in _scan_files:
    if not os.path.exists(_py):
        continue
    with open(_py, encoding='utf-8') as _fh:
        for _opt in re.findall(r"add_argument\(\s*['\"]--([\w-]+)", _fh.read()):
            flags.add('--' + _opt)

print('actual flags (%d)\n' % len(flags))

# ---------- 2. 文档中的参数必须存在（含插件自身声明的参数） ----------
PLUGINS = {}
try:
    PLUGINS = {p.name: p.path for p in REG.discover_plugins(ROOT)}
except Exception:
    PLUGINS = {}
for name, path in PLUGINS.items():
    rc, out = run([path, '--help'], timeout=60)
    for _opt in re.findall(r'--[a-zA-Z][a-zA-Z0-9-]*', out):
        flags.add(_opt)

for doc, text in doc_text.items():
    bad = sorted({f for f in FLAG_RE.findall(text) if f not in flags})
    check('doc flags valid: ' + os.path.basename(doc), not bad, bad)

# ---------- 3. 文档中的模块名必须存在 ----------
VALID = ({s for subs in SCRIPTS.values() for s in subs}
         | set(PLUGINS) | {'selftest', 'selfcheck', 'help', 'test', 'all'})
for doc, text in doc_text.items():
    refs = set(re.findall(r'ctf\.py\s+([a-zA-Z][\w-]*)', text))
    bad = sorted(r for r in refs if r not in VALID)
    check('doc ctf.py modules valid: ' + os.path.basename(doc), not bad, bad)

# ---------- 4. 功能子命令覆盖（数量从注册表派生） ----------
functional = [s for subs in SCRIPTS.values() for s in subs if s != 'selftest']
n_functional = len(functional)
cli_ref = doc_text.get(os.path.join(ROOT, 'CLI_REFERENCE.md'), '')
kern = doc_text.get(os.path.join(SKILL, 'references', 'kernels.md'), '')
readme = doc_text.get(os.path.join(ROOT, 'README.md'), '')
missing_ref = [s for s in functional if ('`%s`' % s) not in cli_ref]
missing_kern = [s for s in functional
                if not re.search(r'(?<![\w-])%s(?![\w-])' % re.escape(s), kern)]
check(f'CLI_REFERENCE lists all {n_functional} functional subcommands', not missing_ref, missing_ref)
check(f'kernels.md lists all {n_functional} functional subcommands', not missing_kern, missing_kern)

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

# ---------- 7. 关键章节 + 版本真源一致性 ----------
VERSION_HEADER_RE = re.compile(r'文档版本[^\n]*?v(\d+\.\d+)')
for p, text in doc_text.items():
    m = VERSION_HEADER_RE.search(text)
    check('has version header: ' + os.path.basename(p), bool(m),
          '缺「文档版本 vX.Y」表头')
for p, text in doc_text.items():
    m = VERSION_HEADER_RE.search(text)
    got = m.group(1) if m else ''
    check('doc version matches source: ' + os.path.basename(p), got == VERSION,
          f'doc={got or "?"} source={VERSION}')
for token in ('退出码矩阵', '稳定契约', '★', 'UTF-8'):
    check('CLI_REFERENCE contains ' + token, token in cli_ref)
check(f'CLI_REFERENCE count {n_functional}', f'{n_functional} 个功能子命令' in cli_ref)
check(f'README count {n_functional}', f'{n_functional} 个功能子命令' in readme)

# 发布 tag 也必须指向当前版本（历史变更记录里的旧版本号不算）
tag_versions = set(re.findall(r'(?:tag|标签)\s*`v(\d+\.\d+)`', readme)) | \
    set(re.findall(r'(?:tag|标签)\s*`v(\d+\.\d+)`', cli_ref))
check('release tag matches source version', tag_versions <= {VERSION},
      sorted(tag_versions - {VERSION}))

# ---------- 8. 统一入口 help ----------
rc, out = run([CTF, 'help'])
missing_help = [s for s in functional if s not in out]
check('dispatcher help lists all subcommands', not missing_help, missing_help)

# ---------- 9. 部署回退表 == 注册表（单点真源，防 skill 单独部署漂移） ----------
embedded_scripts = _literal_assign(CTF, '_EMBEDDED_SCRIPTS')
embedded_globals = _literal_assign(CTF, '_EMBEDDED_GLOBALS')
if REG is not None:
    reg_scripts = dict(REG.SCRIPTS)
    reg_globals = {k: tuple(v) for k, v in REG.SCRIPT_GLOBALS.items()}
    check('dispatcher _EMBEDDED_SCRIPTS == registry SCRIPTS',
          isinstance(embedded_scripts, dict) and embedded_scripts == reg_scripts,
          {k: (embedded_scripts or {}).get(k) for k in set(reg_scripts) | set(embedded_scripts or {})
           if (embedded_scripts or {}).get(k) != reg_scripts.get(k)})
    check('dispatcher _EMBEDDED_GLOBALS == registry SCRIPT_GLOBALS',
          isinstance(embedded_globals, dict)
          and {k: tuple(v) for k, v in embedded_globals.items()} == reg_globals,
          (embedded_globals, reg_globals))
else:
    check('dispatcher fallback tables checked', False, 'ctf_registry 不可用')

# ---------- 10. CLI_REFERENCE 全局参数矩阵 == 注册表能力位 ----------
if REG is not None:
    def _matrix_modules(option):
        """从 CLI_REFERENCE 的全局参数表里取 `option` 行第 4 列列出的模块集合。"""
        for line in cli_ref.splitlines():
            if '`%s`' % option not in line or not line.strip().startswith('|'):
                continue
            cols = [c.strip().strip('\u3000') for c in line.strip().strip('|').split('|')]
            if len(cols) < 4 or f'`{option}`' not in cols[0]:
                continue
            mods = set()
            for tok in re.split(r'[,，、/\s]+', cols[3]):
                tok = tok.strip().strip('`')
                if tok in ('gpu_crack', 'ctf_tools', 'ctf_ext', 'ctf_misc'):
                    mods.add(tok + '.py')
            return mods
        return None

    for opt in ('--backend', '--chunk', '--json', '--checkpoint'):
        want = {s for s, glob in REG.SCRIPT_GLOBALS.items() if opt in glob}
        got = _matrix_modules(opt)
        check(f'CLI_REFERENCE {opt} matrix matches registry', got == want, (opt, got, want))
else:
    check('CLI_REFERENCE global matrix checked', False, 'ctf_registry 不可用')

# ---------- 11. 示例插件在文档中被提及 ----------
plugin_names = sorted(PLUGINS)
if plugin_names:
    all_doc = '\n'.join(doc_text.values())
    missing_plugins = [n for n in plugin_names if n not in all_doc]
    check('plugins mentioned in docs', not missing_plugins, missing_plugins)
else:
    check('plugins mentioned in docs', True, '')

# ---------- 12. 新增验证入口的文档引用（存在即要求被文档提到） ----------
for fname in ('verify_all.py', 'fuzz_kernels.py'):
    p = os.path.join(ROOT, fname)
    if os.path.exists(p):
        all_doc = '\n'.join(doc_text.values())
        check(f'docs reference {fname}', fname in all_doc, f'{fname} 存在但文档未提及')
    else:
        check(f'docs reference {fname}', True, '（文件尚不存在，跳过）')

print()
print('DOC-LINT:', 'ALL PASS' if not FAIL else ('%d FAIL' % len(FAIL)))
if FAIL:
    print('failed:', FAIL)
sys.exit(0 if not FAIL else 1)
