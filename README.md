# gpu_crack — GPU 批量爆破 / 校验工具箱

> English: a GPU-accelerated CTF toolkit for batch brute-force and triage — MT19937 seed scan, AES S-box
> fault scan, ChaCha20/RC4/AES-CTR keystream, RS(63,31), TEA/Speck, LCG recovery, MD5 cracking,
> RSA multi-attack, multi-layer decoding, image LSB/stegscan, ROP gadgets and ELF recon.
> Command-line only, backend auto-select (torch CUDA → CPU → numpy). Docs are in Chinese; see
> `CLI_REFERENCE.md` for every flag and exit code. Licensed MIT.

| 文档属性 | 值 |
|---|---|
| 文档版本 | v3.2（2026-11-09） |
| 工具版本 | gpu_crack toolkit v2 |
| 校验状态 | `python -X utf8 verify_all.py` 全绿（doc_lint 50 项 / 模块自测 13+31+23 项 / 主框架 42 项 / consistency 45/45 / fuzz 17 类 / 统一入口+示例 6/6） |
| 适用环境 | Windows + Python 3.15 + torch 2.14.1+cu130（无 torch 时自动回退 numpy） |
| GitHub | https://github.com/Infinite-End/gpu_crack（**public / MIT**，tag `v3.2`） |
| 配套文档 | [CLI_REFERENCE.md](CLI_REFERENCE.md)（**参数/输出/退出码以它为准**）、`skill/SKILL.md` 与 `skill/references/kernels.md` |
| 许可证 | [MIT](LICENSE) |

---

## 1. 项目定位

面向 CTF 密码学 / 取证 / Pwn 的本地批处理工具箱。核心思想：
**能并行的候选一律批量算，能用廉价判据的先算廉价判据，命中立刻早停。**

- 输入输出全部走命令行（或作为 Python 库 import），不依赖网络、不依赖交互。
- GPU 后端与 numpy 后端结果一致（`consistency.py` 交叉校验）。
- 每个功能子命令都有已知向量或「植入-找回」自测，改动后先跑自测。

## 2. 能力概览

| 模块 | 文件 | 能力 | 子命令 |
|---|---|---|---|
| 批量爆破 | `gpu_crack.py` | mt19937 种子、AES S-box 故障、比特一致度、LFSR | `bench` `mtscan` `sboxscan` `ksscore` `lfsrscan` |
| 密码学 | `ctf_tools.py` | ChaCha20、Poly1305、RS(63,31)、CRC32、XOR、AES-CTR | `chacha` `poly1305` `rs63` `crc32` `xorbrute` `aesctr` |
| 算法族 | `ctf_ext.py` | TEA/XTEA/XXTEA、Speck、RC4、LCG、MD5、RSA、多层解码、长度扩展、LFSR 反推、ZIP 爆破、ECC 小私钥 | `tea` `speck` `rc4` `lcg` `md5` `rsa` `encode` `mdext` `bm` `zipbrute` `eccscan` |
| 取证 & Pwn | `ctf_misc.py` | LSB 隐写、文件级 stego、图像差异、位平面、cyclic、ROP、ELF | `lsb` `stegscan` `imgdiff` `bitplane` `cyclic` `cycfind` `gadgets` `elfinfo` |

共 **30 个功能子命令** + 3 个模块级 `selftest`。完整参数说明见 [CLI_REFERENCE.md](CLI_REFERENCE.md)。

所有模块共用一套调度/输出参数：`--json`（单行 JSON 结果）、`--quiet`、`--no-progress`、
`--checkpoint FILE` / `--reset-checkpoint`（断点续跑）。CUDA 显存不足时 `--chunk` 会**自动减半重试**，
不需要人工调参。

## 3. 环境与解释器

### 3.1 依赖与安装

| 依赖 | 版本（本机） | 用途 |
|---|---|---|
| Python | 3.15.2 | 运行 |
| torch | 2.14.1+cu130 | GPU 后端（无则自动 numpy） |
| numpy | 2.5.4 | 数组/标量后端 |
| Pillow | 13.0.0 | `ctf_misc` 图像处理 |
| capstone | 5.0.9 | `gadgets` 反汇编 |
| pyelftools | 0.33 | `elfinfo` 解析 ELF |
| pycryptodome / cryptography | 4.2.0 / 50.0.2 | **仅** `consistency.py` 交叉校验用 |

安装（克隆后一次即可）：

```bash
git clone https://github.com/Infinite-End/gpu_crack.git
cd gpu_crack
python -m venv .venv
.venv\\Scripts\\activate        # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
pip install torch   # 可选：GPU 加速；CUDA 版安装命令见 https://pytorch.org/get-started/locally/
```

只有 `numpy` 是硬性要求：没有 torch 或没有 CUDA 也能跑，GPU 内核会自动回落 numpy 后端。

### 3.2 解释器选择算法（`ctf.py`）

候选解释器按下列顺序产出（全部自动发现，代码内**不含任何硬编码本机路径**）：

1. 环境变量 `CTF_PYTHON` —— **强制采纳**（只校验文件存在，不检查 CUDA；指向非 Python 文件时会告警并回退）；
2. 运行 `ctf.py` 的当前解释器 `sys.executable`；
3. PATH 上的 `python3`；
4. PATH 上的 `python`；
5. Windows `py -0p` 登记的每个版本；
6. 工具箱目录内的 `.venv` / `venv`。

除第 1 步外，逐个用 `python -c "import torch;assert torch.cuda.is_available()"` 探测，**取第一个通过者**；全都不通过时回退到当前解释器（GPU 内核自动走 `--backend numpy` 慢路径），不会报错退出。

探测结果会缓存到用户缓存目录（Windows `%LOCALAPPDATA%\gpu_crack\`，Linux/macOS `$XDG_CACHE_HOME/gpu_crack/` 或 `~/.cache/gpu_crack/`），
键包含候选解释器列表（绝对路径 + mtime_ns + size）+ 探测类型；命中前重新 stat 选中解释器，任一变化即失效重探测，
「全都没有 CUDA → 回落当前解释器」为负缓存（6 小时后复检），Windows `py -0p` 候选列表另缓存 1 小时。
缓存读-改-写用同目录临时文件 + `os.replace` 原子替换，最后写者胜、永不写坏；不可写/损坏时静默忽略。
实测：冷启动 dispatch ≈3.1 s → 热启动 ≈0.32 s，`ctf.py help` 2.7 s → 0.10 s（help 不再探测解释器）。
`CTF_NO_PY_CACHE=1` 关闭跨进程缓存，`CTF_PY_CACHE=<路径>` 指定缓存文件，`CTF_DEBUG_TIMING=1` 打印各阶段耗时。

### 3.3 路径解析与环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `CTF_TOOLKIT_DIR` | 自动探测：有效时优先，否则从 `ctf.py` 向上查找 `gpu_crack.py`，再退到向上两级 | 工具箱目录（`ctf.py` 找模块脚本用）；skill 目录被单独拷出仓库时须设为仓库根 |
| `CTF_PYTHON` | 见 3.2 | 强制指定解释器 |

skill 目录：仓库内 `skill/`（统一入口 `skill/scripts/ctf.py`、文档 `skill/SKILL.md`、`skill/references/kernels.md`）。

### 3.4 后端（backend）

`--backend` 仅 `gpu_crack.py` / `ctf_tools.py` / `ctf_ext.py` 支持（`ctf_misc.py` 纯 CPU）：

| 取值 | 行为 |
|---|---|
| `auto`（默认） | torch-CUDA → torch-CPU → numpy 依次回退 |
| `torch` | 同上（torch 不可用时静默回退 numpy） |
| `numpy` | 只用 numpy（慢但结果一致） |

- `--chunk`：每批候选数；`gpu_crack` 默认 65536、`ctf_tools` 默认 16384。
  8 GB 显存 OOM 时把它减半（如 `--chunk 8192`）。

## 4. 快速开始

```powershell
$py = 'python'                          # 需装有 torch + CUDA 的解释器
$ctf = '.\skill\scripts\ctf.py'

# 0) 先摸当前机器吞吐（换机器/驱动必跑）
& $py .\gpu_crack.py bench

# 1) 弱随机数：暴力 mt19937 种子（2^32 全空间约 8 分钟）
& $py .\gpu_crack.py mtscan --key-hex ac2f75c043fbc36709d315f2 --seed-count 4294967296

# 2) 统一入口（自动选解释器、自动把 --backend/--chunk 挪到正确位置）
python $ctf selftest
python $ctf rsa --n 0x... --e 65537 --c 0x...
python $ctf xorbrute --data-hex 1b37373331363f78 --max-len 1
```

最小可用样例（每条都可直接粘贴运行）：

```powershell
# XOR 单字节（经典向量）
python $ctf xorbrute --data-hex 1b37373331363f78151b7f2b783431333d --max-len 1

# 图像 LSB
python $ctf lsb --img chal.png

# RSA 自动攻击
python $ctf rsa --n 0x... --e 65537 --c 0x...

# TEA 单块（全零 key/明文的标准向量，输出小端显示）
python $ctf tea --mode tea --op encrypt --key-hex 00000000000000000000000000000000 --data-hex 0000000000000000
```

## 5. 统一入口 `ctf.py`

```
python ctf.py <module-or-subcommand> [参数...]
python ctf.py selftest      # 四个模块组合自测（gpu bench + 3×selftest）
python ctf.py selfcheck     # 统一入口自身自检（不跑内核、不需要 GPU）
python ctf.py help [module] # 总帮助 / 单个模块帮助（等价于 help <module>）
```

- 模块名即子命令名，见第 2 节；每个模块的 `--help` 可用 `python ctf.py <sub> --help` 查看（`ctf.py help <sub>` 等价）。
- `--backend/--chunk` 放在模块名前后都行；对 `ctf_misc.py` 这类不支持全局参数的模块会提示并忽略，
  对 `ctf_ext.py` 传 `--chunk` 也会提示「已忽略」而不是报错。`ctf.py -- <module> ...` 可显式结束全局参数解析。
- `selfcheck` 只跑调度器自身单元自检（参数搬运/退出码聚合/缓存/回退表/插件容错），不跑内核、不需要 GPU，约 2 s。
- 退出码与模块一致（见 [CLI_REFERENCE.md §6](CLI_REFERENCE.md#6-退出码矩阵)）；`selftest` 为四个模块退出码按位 OR 聚合。
- 解释器探测结果有缓存（见 §3.2），连续调用不再重复 `import torch` 探测。

## 6. CLI 通用契约（摘要）

> 完整规定（含每个子命令的参数表/输出形态/退出码）见 **[CLI_REFERENCE.md](CLI_REFERENCE.md)**。

| 项目 | 约定 |
|---|---|
| 参数格式 | 形如 `--key-hex <值>`；布尔开关写作 `--locate`。整数/hex 的具体格式按子命令注释（见 CLI_REFERENCE） |
| 输出 | stdout = 人类可读结果；stderr = 错误。命中行含 `★` 标记 |
| 退出码 | `0` 成功（未命中/未攻破也是 0，靠输出判断）；`1` 自测失败/校验不匹配；`2` 参数或输入错误 |
| 错误 | 不向终端抛 traceback（内部 bug 除外）；解析请以退出码为准，不要匹配中文文案 |
| 编码 | UTF-8；Windows 控制台建议统一入口或加 `-X utf8`（`PYTHONUTF8=1`） |
| 随机性 | `ksscore`、`chacha --brute` 不可复现；`lfsrscan --rng-seed` 固定即复现；其余确定性 |
| 全局参数位置 | `gpu_crack.py`：子命令前后均可；`ctf_tools.py` / `ctf_ext.py`：须写在子命令前（`ctf_ext.py` 的 `--chunk` 为「提示忽略」）；走 `ctf.py`：任意位置 |

## 7. 库 API（Python import）

```python
import sys; sys.path.insert(0, r'.')
import gpu_crack as G

G.init_backend('torch')            # 'auto' | 'torch' | 'numpy'
T = G.xp()                         # 当前张量库（torch 或 numpy）
DEV = G.DEV                        # torch.device('cuda') 或 None

# 批量 mt19937：种子 -> 密钥（nbytes 可选 1~24，只算需要的前缀字节）
seeds = T.arange(1_000_000, dtype=T.int64, device=DEV)
keys = G.mt19937_keys(seeds, twist=False)               # [N,24] uint8
keys12 = G.mt19937_keys(seeds, twist=False, nbytes=12)  # [N,12] uint8（更快）

# 比特一致度打分（1D 目标 -> [N]；2D 目标 -> [N,M]）
score = G.bit_agreement(keys, T.tensor([0xac,0xf7,0x5c,0x04], dtype=T.uint8, device=DEV))

# 批量 AES-192 单块（fast path：单密钥 + 每候选不同 S-box 破坏位）
ct = G.aes192_ecb(blocks, key_t, None, a_idx, b_val)          # blocks/key_t/a_idx/b_val 见函数 docstring

# 其他模块同理：ctf_tools / ctf_ext / ctf_misc 的 cmd_* 与内核函数均可直接 import
```

约定：内核函数返回 numpy（`G.from_dev()` 可把 torch 张量搬回 CPU numpy）；后端用 `init_backend()` 切换。
更细的签名见各模块 `cmd_*` 函数与 [CLI_REFERENCE.md](CLI_REFERENCE.md)。

## 8. 验证与回归

六层验证，**改代码后跑 `verify_all.py` 一键全跑**（从底层往高层，失败先修下面层）：

```
python -X utf8 verify_all.py                  # 全部层
python -X utf8 verify_all.py --backend numpy  # 无 GPU / Windows CI
python -X utf8 verify_all.py --quick --only doc,fuzz   # 快速/局部
```

`verify_all.py` 把 6 层展开为 **9 个执行步骤**（模块自测 3 步、dispatcher/examples 各 1 步），
结尾打印 `9/9 个步骤通过，覆盖 6 个验证层`。

| 层级 | 命令 | 期望 |
|---|---|---|
| 1. 文档一致性 | `python -X utf8 doc_lint.py` | `DOC-LINT: ALL PASS`（参数/模块/计数/版本/矩阵与真实 CLI 对齐） |
| 2. 模块自测 | `python -X utf8 ctf_tools.py selftest`（`ctf_ext.py` / `ctf_misc.py` 同理） | 全 PASS，退出码 0 |
| 3. 主框架自测 | `python -X utf8 selftest.py` | 42 项断言全 PASS，退出码 0（AES-128/192/256 NIST、mt19937 双后端、mtscan/sboxscan 植入-找回、断点/JSON 契约） |
| 4. 跨实现一致性 | `python -X utf8 consistency.py` | `通过 45/45`（原有 32 条 + 新增 13 条；torch/numpy/pycryptodome/cryptography/hashlib/zlib 交叉） |
| 5. 差分模糊测试 | `python -X utf8 fuzz_kernels.py --quick` | 17 类内核差分对拍（`--quick` 下 dual_backend 跳过，加 `--backend torch` 为 17/17；独立参考实现/第三方库） |
| 6. 组合自测 + 示例 | `python .\skill\scripts\ctf.py selftest`；`python examples/make_examples.py --check` | 四模块全 PASS；示例命令逐条复跑通过 |
| 附：入口/公共层自检 | `python .\skill\scripts\ctf.py selfcheck`；`python -X utf8 ctf_common.py` | 参数搬运/退出码/缓存/回退表全 PASS |

新增内核的验收标准：① 有已知向量或植入-找回；② 双后端一致；③ 参数错误走干净报错（rc=2/1，无 traceback）；
④ 在 `consistency.py`、`fuzz_kernels.py` 或模块 `selftest` 中留下断言。

## 9. 性能基线

> 数据为 2026-11-01 本机复测（RTX 5070 Laptop 8 GB，Python 3.15，torch 2.14.1+cu130）。
> 同一内核随驱动/温度最多相差 2~3 倍，**以 `bench` 的即时输出为准**，勿把历史数字当承诺。

| 内核 | 实测 | 用法 |
|---|---|---|
| `popcount_matrix` | ~1000 M/s | 比特一致度打分基元 |
| `mt19937_keys` | ~30 M 种子/s（bench，24B）；12B 稳态 ~63 M/s、3B ~294 M/s（按需截断） | 2^32 种子（24B）端到端 ≈ 3~4 分钟（20 M/s 量级） |
| `mtscan` 端到端 | 3M 种子 8.2 M/s；10M 种子 20.5 M/s（首块 ~0.32 s CUDA 初始化对短扫描影响大） | 先 `bench` 再估时 |
| `mt19937_keys`（numpy 后端） | 向量化后 ~2.5 M 种子/s（旧实现 ~2 K/s，约 1000×） | 无 GPU 时可用 |
| `aes192_ecb` | ~1.6 M 块/s（AES-128/192/256 分别约 2.1/1.8/1.5 M/s） | S-box 故障枚举 |
| `sboxscan` | 65,280 组合 ≈ 0.4~0.6 s（含全块批量验证） | 故障攻击 |
| ChaCha20 | ~590 K keys/s（×64B） | 40 万候选 ≈ 0.7 s |
| RS63 | ~69 K 码字/s | 批量 syndrome |
| CRC32 | ~3.4 M/s（×64B） | 匹配 |
| xorbrute | 3 字节 / 15B 密文 ≈ 6 s | 1~3 字节爆破 |
| `lsb` 大图 | 3000×3000 RGB ≈ 6.5 s；4000×4000 级 ≈ 14 s（8 GB 内存） | 大图先看 `stegscan` |

> 性能数字为参考量级；共享 GPU / 驱动与温度会造成 ±20% 波动，**以 `bench` 即时输出为准**。
> 短扫描（<10M 种子）的端到端吞吐被 ~0.32 s 的 CUDA 初始化摊薄，这是固有成本。

性能设计原则（选型时遵守）：
1. 先算廉价判据（前缀 XOR+popcount），再算完整 tag/加密；
2. 分块执行、命中早停，显存不足自动/手动降 `--chunk`；
3. 常量移出批量（如 AES round key 在 CPU 算一次再广播）；
4. 串行依赖截断（mt19937 只要前 24 个状态就只算 24 步）；
5. 测吞吐前先预热（`bench` 已内置 3 次预热）。

## 10. 已知限制

| 限制 | 说明 |
|---|---|
| 无 GPU 时 | 自动 numpy 回退，结果正确但慢（mt19937 已向量化，chacha/ksscore 仍明显慢） |
| `xorbrute` | 只支持 1~3 字节密钥；密文 <8 字节时打分不可靠；4 字节直接拒绝（内存爆炸） |
| `encode` | 只在结果含 flag 特征时返回「最终结果」；否则只打印中间候选（不硬凑） |
| `gadgets` | 只回溯结尾指令前 32 字节窗口（含结尾指令 33~35 字节）；`f3 c3`（rep ret）等前缀结尾不单独枚举；仅 x86-64/i386，不支持 arm64 |
| `stegscan` | JPEG 尾部用 `rfind(FFD9)`：尾部数据含 FFD9 时长度偏小、以 FFD9 结尾时**漏报**（PNG 已改为 chunk 链，无此问题）；不解析 iTXt/zTXt、不校验 PNG chunk CRC |
| `lsb` | 16-bit **彩色/带 alpha** PNG 受 Pillow 解码限制只能看高 8 位（16-bit 灰度已完整支持）；P 模式只扫调色板索引，不扫调色板 RGB 颜色值；只扫 bit0~2 |
| `elfinfo` | RELRO 只区分 Full（是/否）；不解析 partial RELRO 细分；Canary 用 `__stack_chk_fail` 字符串启发式 |
| RSA 共模/广播/p-1 素数幂 | 只有库函数，不在 CLI 五连中 |
| 平台 | 路径/命令示例按 Windows 写；Linux/macOS 上把 `python`/`\` 换成本机习惯即可 |

## 11. 目录结构

```
<仓库根>\
├─ gpu_crack.py       主框架（bench/mtscan/sboxscan/ksscore/lfsrscan + 批量内核库）
├─ ctf_tools.py       密码学内核（chacha/poly1305/rs63/crc32/xorbrute/aesctr）
├─ ctf_ext.py         算法族（tea/speck/rc4/lcg/md5/rsa/encode/mdext/bm/zipbrute/eccscan）
├─ ctf_common.py      共享调度（分块 + OOM 自动降档 + 进度 ETA + 断点续跑 + JSON 输出）；`python ctf_common.py` 自检
├─ ctf_registry.py    内核注册表 + `VERSION` 版本真源（子命令 → 脚本/分组/能力）；`python ctf_registry.py` 自检
├─ plugins/           插件内核目录（放 *.py + `KERNEL` 元数据即可，见 plugins/README.md）
├─ ctf_misc.py        取证 & Pwn（lsb/stegscan/imgdiff/bitplane/cyclic/cycfind/gadgets/elfinfo）
├─ selftest.py        主框架自测（42 项断言：AES-128/192/256 NIST / mt19937 双后端 / 植入-找回 / 断点/JSON 契约 / 性能）
├─ consistency.py     跨实现一致性校验（45 项固定向量 + `--fuzz N` 随机对拍）
├─ fuzz_kernels.py    差分模糊测试（独立参考实现/第三方库，17 类内核随机对拍，`--quick` 供 CI）
├─ verify_all.py      一键跑全部验证层（本表所有层，CI 与交付前用）
├─ doc_lint.py        文档一致性检查（参数/模块/计数/版本/矩阵必须与真实 CLI 对齐）
├─ docs/bench.md      本机吞吐基准（`bench --markdown` 生成，勿手改）
├─ examples/          示例输入生成器 + 逐条跑法（`--check` 逐条复跑；examples/out/ 不入库）
├─ .github/workflows/ci.yml  CI：doc_lint + 自测 + 对拍 + fuzz + 统一入口 + 示例复跑 + verify_all
├─ README.md          本文件（项目说明）
├─ CLI_REFERENCE.md   完整 CLI 参考（参数/输出/退出码/示例）
├─ requirements.txt    依赖清单（pip install -r requirements.txt）
├─ .gitignore / .gitattributes
└─ skill/             skill 定义同步副本（GitHub 版；Codex / DSH 可直接注册本目录）
   ├─ SKILL.md         中文完整版（文档一致性检查的对象）
   ├─ SKILL.en.md      英文精简版（对外/英文模型用，与 SKILL.md 同步维护）
   ├─ references/kernels.md
   ├─ agents/openai.yaml   Codex skill 元数据（display_name / short_description / default_prompt）
   └─ scripts/ctf.py    统一入口
```

## 12. 变更记录

### v3.2（2026-11-09，本轮：全面对抗测试 + 缺陷修复 + 第六层验证）

> 方法：6 个并行 owner 各负责一个模块/基础设施范围，先对抗复现、再修根因、最后把探针固化为回归断言；
> 独立 verifier 复核（见 `reports/`，工作产物不入库）。

- **验证体系升级**：
  - 新增 `fuzz_kernels.py`：17 类内核差分模糊测试（zlib/hashlib/pycryptodome/cryptography + 自写规范参考实现），
    `--quick` 约 0.4 s 供 CI，固定 seed 可复现，失败 rc=1、harness 自检失败 rc=2。
  - 新增 `verify_all.py`：六层验证一键执行（`--backend numpy` / `--quick` / `--only` / `--json`）。
  - `consistency.py` 32 → **45** 项固定向量；`selftest.py` 重写为 **42 项断言**（原先恒 rc=0、植入-找回不断言）；
    新增 `ctf.py selfcheck`（52 项）、`ctf_common.py`（27 项）、`ctf_registry.py`（25 项）自检入口；
    `examples/make_examples.py --check`（6 条示例逐条复跑）。CI 覆盖七层 + 解释器缓存冒烟。
- **基础设施**：
  - 解释器探测跨进程缓存（热调用 2.9~3.4 s → **0.32~0.36 s**，`ctf.py help` 2.7 s → **0.10 s**）；
    `CTF_NO_PY_CACHE=1` / `CTF_PY_CACHE=<file>` / `CTF_DEBUG_TIMING=1`；`CTF_PYTHON` 指向非 Python 会告警回退。
  - 插件元数据改 **AST 静态读取**（`ctf.py help`/doc_lint 不再执行插件顶层代码），exec 仅兜底；插件保留命令名拒绝。
  - 断点文件升级 v2：`meta` 任务指纹（同文件换目标不再 0 扫描/误报旧命中）；写入失败去重告警、原子写。
  - `--` 结束全局解析、`help <module>`、`selftest` 尾随全局参数；`--chunk` 对不支持的模块改为「提示忽略」。
  - 版本单一真源 `ctf_registry.VERSION`；`doc_lint` 35 → **50** 项（版本/tag/能力矩阵/部署回退表/新入口引用）。
- **gpu_crack**：torch 后端命中枚举只报 1 个（P1，`nonzero` 形状）→ 256/256 双后端一致；非法 hex/taps/负种子/
  种子越界全部 rc=2 单行报错；numpy mt19937 向量化（~1000×，负种子 uint32 回绕）；bench 支持 torch-cpu；
  `--json`（新 schema）/`--markdown`（纯表格）契约；全局参数子命令前后均可；`lfsrscan` 流式打分去掉 2 GiB 矩阵；
  `mt19937_keys(..., nbytes=)` 按需截断（mtscan 3M 6.8→8.2 M/s、10M 20.5 M/s）。
- **ctf_tools**：六命令 `--json` 全被吞（P1）→ 输出 hit/result/locate/error 事件；`chacha --brute`/`xorbrute`
  接入分块调度，`--chunk/--checkpoint` 不再空转（断点 v2 + 任务指纹，xorbrute 结果与 `--chunk` 无关）；
  keys-file 容忍 UTF-8 BOM；`chacha --brute` 进度改 stderr。
- **ctf_ext**：`eccscan` 命中退出码=私钥（P1）→ rc=0；`--json` 覆盖 4/11 → **11/11**；`mdext`/`zipbrute`
  输入校验（traceback/静默误报 → rc=2）；`encode` 识别大写 `FLAG{}`；`--quiet` 抑制 rsa/encode 诊断；
  Speck 放开 `--word-bits 64`（新增 Speck128/128、128/192、128/256）。
- **ctf_misc**：`lsb` 强转 RGB 丢失 16-bit 灰度/LA alpha 隐写（P1）→ 保留原生位深；`_grade` flag 窗口
  256 B → 64 KiB；CLI 参数错误 rc=2 → **1**（对齐契约）；`elfinfo` 截断 ELF 半解析后 rc=0 → rc=1；
  `find_magics` 短签名格式校验（45 MB PNG 误报 2956 → 2 条）；`gadgets` 32 B 窗口 off-by-one；`imgdiff`
  LSB 按像素计数；`lsb` 大图 18.1 → 6.5 s、`stegscan` 45 MB 3.4 → 1.5 s。
- **文档**：README / CLI_REFERENCE / SKILL.md / kernels.md 全量对齐（版本、参数、JSON 字段表、断点 v2、
  退出码、API `nbytes`、已知限制），`docs/bench.md` 重新生成。功能子命令数保持 **30**（无新增子命令）。

### v3.1（2026-11-09：插件化 + 单点注册表）

- 新增 `ctf_registry.py`：**唯一真源**记录「子命令 → 脚本 / 分组 / 是否支持 `--chunk`·`--json`」。
  原先这张表被抄在 `skill/scripts/ctf.py`、`doc_lint.py`、README、CLI_REFERENCE 四处，加内核要改四遍。
- 新增 `plugins/` 插件目录：放一个 `*.py`（含模块级 `KERNEL = dict(name=..., help=...)` + 可执行入口）
  即可被统一入口识别，**无需改仓库任何文件**；重名以内置优先并告警，插件加载失败不影响主程序。
- 自带示例插件 `plugins/morse_decode.py`（`morse` 子命令，摩斯解码）。
- `ctf.py` 改为从注册表取模块表，**脱离仓库部署时自动回退**到内置表（部署副本仍零配置可跑）；
  `help` 输出改为按 4 组分类 + 插件单列。
- `doc_lint.py` 的子命令清单改为从注册表反查，消除"文档与 CLI 名单漂移"的结构性风险。
- `python ctf_registry.py` 可直接打印注册表自检（内置数、分组、已发现插件）。

### v3.0（2026-11-09，本轮：工程化收尾）

- **CI**：`.github/workflows/ci.yml`，Ubuntu + Python 3.11/3.12，跑 `doc_lint`、四个模块 selftest、
  `consistency`（固定向量 + `--fuzz 25` 随机对拍）、统一入口冒烟、examples 生成。
- **examples/**：`make_examples.py` 生成 6 组可跑通示例（mt19937 密钥流、Fermat RSA、ECC 小私钥、
  LFSR 比特流、LSB PNG、假口令字典），`examples/README.md` 给出逐条命令；生成物 `examples/out/` 不入库。
- **`bench --markdown`**：吞吐基准可直接输出 Markdown 表格，本机数据生成到 `docs/bench.md`
  （新增 AES-128/256 行）。
- **`consistency.py --fuzz N`**：随机密钥长度/随机块/随机 nonce 对拍（torch ↔ numpy ↔ pycryptodome），
  默认 0 保持原有 32 项；`--fuzz 20` → 72/72。

### v2.3（2026-11-09，本轮：通用调度 + 4 个新内核）

- 新增 `ctf_common.py`：统一分块调度（CUDA OOM 自动减半重试）、进度/ETA、断点续跑、JSON 输出。
- 新增全局参数：`--json` / `--quiet` / `--no-progress` / `--checkpoint` / `--reset-checkpoint`（gpu_crack、ctf_tools、ctf_ext）。
- **AES 通用化**：`aes_ecb` 支持 128/192/256（轮数 10/12/14 按密钥长度自动判定），`sboxscan` 与
  `aesctr` 同步放开密钥长度；`aesctr` 新增 `--nonce-mode le|be` 与 12/16 字节 nonce。
- 新增 4 个内核（26 → 30 个功能子命令）：
  - `mdext`：MD5 / SHA-1 哈希长度扩展（纯 Python 压缩函数，自测对拍 hashlib）；
  - `bm`：Berlekamp-Massey 由比特流反推 LFSR 多项式与线性复杂度；
  - `zipbrute`：ZIP 口令爆破（字典 / 掩码，多进程）；
  - `eccscan`：ECC 小私钥区间搜索（secp256k1 / P-256，Jacobian 混合加法 + 交叉相乘判等，单核 ~1.1 M keys/s）。
- `mtscan` / `sboxscan` / `ksscore` 改走统一调度，获得 OOM 降档、进度 ETA、断点续跑。
- `doc_lint.py`：子命令清单与计数同步到 30。

### v2.2（2026-11-09，本轮：skill 部署化 + 可移植性）

- `skill/scripts/ctf.py`：
  - 非交互（管道/重定向）时强制 stdout/stderr 为 UTF-8，修复 harness/Codex 按 UTF-8 解码时的中文乱码；
  - 工具箱目录改为「`CTF_TOOLKIT_DIR`（校验有效）→ 从脚本位置向上查找 `gpu_crack.py` → 向上两级」，
    skill 目录嵌在仓库任意层级均可自洽，不再依赖外部环境变量；
  - 找不到 `gpu_crack.py` 时一次性明确报错并返回 2，不再静默输出四段空自测。
- `skill/SKILL.en.md`：新增英文精简版 skill 说明（面向英文模型 / 对外分发）。
- `skill/agents/openai.yaml`：新增 Codex skill 元数据。
- 文档版本统一到 v2.2（`README.md` / `CLI_REFERENCE.md` / `SKILL.md` / `kernels.md` 原先分别停在 v2.1 与 v1.0）。
- `doc_lint.py`：版本校验接受 v2.2。

### v2.1（2026-11-01，本轮：自测 + 修复 + 文档规范化）

- 修复：`gpu_crack.bit_agreement` numpy 分支漏求和（影响 numpy 后端 `ksscore`/`chacha` 最佳候选）；
  补输入校验与退出码传播。
- 修复 `ctf_tools.py` 14 处：xorbrute 打分回归（经典向量恢复）、numpy 后端 ChaCha20 崩溃、
  crc32 单字节尾爆错误、`rs63 --fcr`/`chacha --brute` 无效、keys-file 容错、poly1305/aesctr 校验等。
- 修复 `ctf_ext.py` 11 类：RSA 共模/广播/p-1/n=p²/结果校验、TEA 静默截断、Speck 参数校验、
  RC4/LCG/MD5/encode 的崩溃与无效参数。
- 修复 `ctf_misc.py` 9 组：stegscan PNG 尾数据 off-by-4 + IEND 陷阱、cyclic/cycfind 与 pwntools 不兼容、
  gadgets 漏报/误报、elfinfo NX/RELRO/PIE 判断、lsb 交织/alpha、imgdiff/bitplane 输出。
- 文档：新增 `CLI_REFERENCE.md`；`README.md` / skill 文档改为规范化结构。

### v2.0（2026-11-01）

- 基线：四模块 26 个功能子命令 + 三模块自测 + 32 项一致性校验；性能基线见第 9 节。

## 13. GitHub 仓库与跨机使用

- 仓库：https://github.com/Infinite-End/gpu_crack （**public / 开源**），标签 `v3.2`。
- 仓库根目录 = 本目录。`skill/` 是 skill 定义（`SKILL.md` / `SKILL.en.md` / `references/kernels.md` / `scripts/ctf.py` / `agents/openai.yaml`）的**同步副本**；
  修改 skill 源后需同步到 `skill/`，`doc_lint.py` 会校验镜像一致性。
- 在其它机器/其它模型上使用：
  1. `git clone` 后**无需任何路径配置**——`ctf.py` 先从 `CTF_TOOLKIT_DIR` 取工具箱目录，未设置时从脚本位置向上查找 `gpu_crack.py`；
     仅当 `skill/` 被**单独拷贝到仓库之外**（如 `~/.codex/skills/ctf-toolkit`）时，才需要把 `CTF_TOOLKIT_DIR` 指向仓库根；
  2. **解释器自动发现**：`CTF_PYTHON` → 当前解释器 → PATH 上的 `python3`/`python` → Windows `py -0p` 登记的各版本 → 仓库内 `.venv`/`venv`，取第一个 `torch.cuda.is_available()` 为真的；全都不行则回落当前解释器走 numpy 后端；
  3. 跑 `python -X utf8 doc_lint.py`（有 repo 内 `skill/` 时自动用镜像）与 `python -X utf8 consistency.py` 确认环境。

## 14. 许可证

本项目以 **MIT License** 开源（见 [LICENSE](LICENSE)）。可自由使用、修改、分发、再许可与商用，
保留版权声明与许可声明即可。第三方依赖（torch/numpy/Pillow/capstone/pycryptodome/cryptography）遵循各自许可证。

