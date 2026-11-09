# gpu_crack — GPU 批量爆破 / 校验工具箱

> English: a GPU-accelerated CTF toolkit for batch brute-force and triage — MT19937 seed scan, AES S-box
> fault scan, ChaCha20/RC4/AES-CTR keystream, RS(63,31), TEA/Speck, LCG recovery, MD5 cracking,
> RSA multi-attack, multi-layer decoding, image LSB/stegscan, ROP gadgets and ELF recon.
> Command-line only, backend auto-select (torch CUDA → CPU → numpy). Docs are in Chinese; see
> `CLI_REFERENCE.md` for every flag and exit code. Licensed MIT.

| 文档属性 | 值 |
|---|---|
| 文档版本 | v2.2（2026-11-09） |
| 工具版本 | gpu_crack toolkit v2 |
| 校验状态 | `ctf.py selftest` 全 PASS；`consistency.py` 32/32；`doc_lint.py` 文档一致性 35 项全 PASS |
| 适用环境 | Windows + Python 3.15 + torch 2.14.1+cu130（无 torch 时自动回退 numpy） |
| GitHub | https://github.com/Infinite-End/gpu_crack（**public / MIT**，tag `v2.2`） |
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
| 算法族 | `ctf_ext.py` | TEA/XTEA/XXTEA、Speck、RC4、LCG、MD5、RSA、多层解码 | `tea` `speck` `rc4` `lcg` `md5` `rsa` `encode` |
| 取证 & Pwn | `ctf_misc.py` | LSB 隐写、文件级 stego、图像差异、位平面、cyclic、ROP、ELF | `lsb` `stegscan` `imgdiff` `bitplane` `cyclic` `cycfind` `gadgets` `elfinfo` |

共 **26 个功能子命令** + 3 个模块级 `selftest`。完整参数说明见 [CLI_REFERENCE.md](CLI_REFERENCE.md)。

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

1. 环境变量 `CTF_PYTHON` —— **强制采纳**，只校验文件存在，不检查 CUDA；
2. 运行 `ctf.py` 的当前解释器 `sys.executable`；
3. PATH 上的 `python3`；
4. PATH 上的 `python`；
5. Windows `py -0p` 登记的每个版本；
6. 工具箱目录内的 `.venv` / `venv`。

除第 1 步外，逐个用 `python -c "import torch;assert torch.cuda.is_available()"` 探测，**取第一个通过者**；全都不通过时回退到当前解释器（GPU 内核自动走 `--backend numpy` 慢路径），不会报错退出。

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
python ctf.py help
```

- 模块名即子命令名，见第 2 节；每个模块的 `--help` 可用 `python ctf.py <sub> --help` 查看。
- `--backend/--chunk` 放在模块名前后都行；对 `ctf_misc.py` 这类不支持全局参数的模块会提示并忽略。
- 退出码与模块一致（见 [CLI_REFERENCE.md §6](CLI_REFERENCE.md#6-退出码矩阵)）；`selftest` 为四个模块退出码的 OR。

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
| 全局参数位置 | 直接调模块：必须在子命令前；走 `ctf.py`：任意位置 |

## 7. 库 API（Python import）

```python
import sys; sys.path.insert(0, r'.')
import gpu_crack as G

G.init_backend('torch')            # 'auto' | 'torch' | 'numpy'
T = G.xp()                         # 当前张量库（torch 或 numpy）
DEV = G.DEV                        # torch.device('cuda') 或 None

# 批量 mt19937：种子 -> 24B 密钥
seeds = T.arange(1_000_000, dtype=T.int64, device=DEV)
keys = G.mt19937_keys(seeds, twist=False)          # [N,24] uint8

# 比特一致度打分（1D 目标 -> [N]；2D 目标 -> [N,M]）
score = G.bit_agreement(keys, T.tensor([0xac,0xf7,0x5c,0x04], dtype=T.uint8, device=DEV))

# 批量 AES-192 单块（fast path：单密钥 + 每候选不同 S-box 破坏位）
ct = G.aes192_ecb(blocks, key_t, None, a_idx, b_val)          # blocks/key_t/a_idx/b_val 见函数 docstring

# 其他模块同理：ctf_tools / ctf_ext / ctf_misc 的 cmd_* 与内核函数均可直接 import
```

约定：内核函数返回 numpy（`G.from_dev()` 可把 torch 张量搬回 CPU numpy）；后端用 `init_backend()` 切换。
更细的签名见各模块 `cmd_*` 函数与 [CLI_REFERENCE.md](CLI_REFERENCE.md)。

## 8. 验证与回归

三层验证，**改代码后从下往上跑**：

| 层级 | 命令 | 期望 |
|---|---|---|
| 1. 模块自测 | `python -X utf8 ctf_tools.py selftest`（`ctf_ext.py` / `ctf_misc.py` 同理） | 全 PASS，退出码 0 |
| 2. 主框架自测 | `python -X utf8 selftest.py` | AES-192 NIST、mt19937 参考、mtscan/sboxscan 植入-找回全 PASS |
| 3. 跨实现一致性 | `python -X utf8 consistency.py` | `通过 32/32`（torch/numpy/pycryptodome/cryptography/hashlib/zlib 交叉） |
| 4. 组合自测 | `python .\skill\scripts\ctf.py selftest` | 四模块全 PASS，退出码 0 |
| 5. 文档一致性 | `python -X utf8 doc_lint.py` | `DOC-LINT: ALL PASS`（参数/模块/计数/章节与真实 CLI 对齐） |

新增内核的验收标准：① 有已知向量或植入-找回；② 双后端一致；③ 参数错误走干净报错（rc=2/1，无 traceback）；
④ 在 `consistency.py` 或模块 `selftest` 中留下断言。

## 9. 性能基线

> 数据为 2026-11-01 本机复测（RTX 5070 Laptop 8 GB，Python 3.15，torch 2.14.1+cu130）。
> 同一内核随驱动/温度最多相差 2~3 倍，**以 `bench` 的即时输出为准**，勿把历史数字当承诺。

| 内核 | 实测 | 用法 |
|---|---|---|
| `popcount_matrix` | ~900 M/s | 比特一致度打分基元 |
| `mt19937_keys` | ~35 M 种子/s（bench）；`mtscan` 端到端 ~9 M/s | 2^32 种子 ≈ 8 分钟 |
| `aes192_ecb` | ~1.6 M 块/s | S-box 故障枚举 |
| `sboxscan` | 65,280 组合 ≈ 0.4 s | 故障攻击 |
| ChaCha20 | ~590 K keys/s（×64B） | 40 万候选 ≈ 0.7 s |
| RS63 | ~69 K 码字/s | 批量 syndrome |
| CRC32 | ~3.4 M/s（×64B） | 匹配 |
| xorbrute | 3 字节 / 15B 密文 ≈ 6 s | 1~3 字节爆破 |

性能设计原则（选型时遵守）：
1. 先算廉价判据（前缀 XOR+popcount），再算完整 tag/加密；
2. 分块执行、命中早停，显存不足自动/手动降 `--chunk`；
3. 常量移出批量（如 AES round key 在 CPU 算一次再广播）；
4. 串行依赖截断（mt19937 只要前 24 个状态就只算 24 步）；
5. 测吞吐前先预热（`bench` 已内置 3 次预热）。

## 10. 已知限制

| 限制 | 说明 |
|---|---|
| 无 GPU 时 | 自动 numpy 回退，结果正确但慢（尤其 chacha/ksscore） |
| `xorbrute` | 只支持 1~3 字节密钥；密文 <8 字节时打分不可靠；4 字节直接拒绝（内存爆炸） |
| `encode` | 只在结果含 flag 特征时返回「最终结果」；否则只打印中间候选（不硬凑） |
| `gadgets` | 只回溯结尾指令前 32 字节窗口；`f3 c3`（rep ret）等前缀结尾不单独枚举 |
| `stegscan` | JPEG 尾部用 `rfind(FFD9)`（PNG 已改为 chunk 链，无此问题） |
| `elfinfo` | RELRO 只区分 Full（是/否）；不解析 partial RELRO 细分 |
| RSA 共模/广播/p-1 素数幂 | 只有库函数，不在 CLI 五连中 |
| 平台 | 路径/命令示例按 Windows 写；Linux/macOS 上把 `python`/`\` 换成本机习惯即可 |

## 11. 目录结构

```
<仓库根>\
├─ gpu_crack.py       主框架（bench/mtscan/sboxscan/ksscore/lfsrscan + 批量内核库）
├─ ctf_tools.py       密码学内核（chacha/poly1305/rs63/crc32/xorbrute/aesctr）
├─ ctf_ext.py         算法族（tea/speck/rc4/lcg/md5/rsa/encode）
├─ ctf_misc.py        取证 & Pwn（lsb/stegscan/imgdiff/bitplane/cyclic/cycfind/gadgets/elfinfo）
├─ selftest.py        主框架自测（AES-192 NIST / mt19937 参考 / 植入-找回 / 性能）
├─ consistency.py     跨实现一致性校验（32 项）
├─ doc_lint.py        文档一致性检查（文档中的参数/模块/计数必须与真实 CLI 对齐）
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

- 仓库：https://github.com/Infinite-End/gpu_crack （**public / 开源**），标签 `v2.2`。
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

