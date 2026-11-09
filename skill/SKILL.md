---
name: ctf-toolkit
description: GPU-accelerated, command-line CTF toolkit in this repository for batch brute-force and triage (MT19937 seed scan, AES S-box fault scan, ChaCha20/RC4/AES-CTR keystream, RS(63,31) syndrome, TEA/XTEA/XXTEA/Speck, LCG recovery, MD5 cracking, RSA multi-attack, multi-layer decoding, image LSB/stegscan, image diff, ROP gadgets, ELF recon, cyclic offsets). Use when a CTF task requires trying many candidates at once, identifying an unknown cipher, recovering keys/keystreams, extracting hidden data from images or files, or computing pwn offsets and gadgets.
---

# CTF Toolkit（gpu_crack）使用规范

| 文档属性 | 值 |
|---|---|
| 文档版本 | v3.0（2026-11-09） |
| 适用工具 | gpu_crack toolkit v2（仓库根目录） |
| 校验状态 | `ctf.py selftest` 全 PASS；`consistency.py` 32/32 |
| 配套文档 | `CLI_REFERENCE.md`（仓库根，规范性参数/输出/退出码）、`README.md`（仓库根）、`references/kernels.md`、`SKILL.en.md`（英文精简版） |

本 skill 是**可被任意模型复用**的操作规范。所有事实以仓库文档为准，不要凭记忆猜参数：
命令行细节查仓库根的 `CLI_REFERENCE.md`，项目说明查 `README.md`。

## 1. 固定事实（先读，别猜）

| 项 | 值 |
|---|---|
| 工具箱目录 | `CTF_TOOLKIT_DIR`（有效时）→ 从 `scripts/ctf.py` 向上查找 `gpu_crack.py` → 向上两级；skill 目录被单独拷出仓库时须设 `CTF_TOOLKIT_DIR` |
| GPU 解释器 | PATH 上的 `python3` / `python`（需 torch + CUDA；也可用 `CTF_PYTHON` 强制指定） |
| 解释器覆盖 | 环境变量 `CTF_PYTHON`（设置且存在时优先） |
| 统一入口 | `scripts/ctf.py`（本 skill 内） |
| 参考文档 | `<skill>/references/kernels.md`（一页速查）、`CLI_REFERENCE.md`（完整参数/输出/退出码） |
| 后端 | `--backend auto|torch|numpy`；`ctf_misc.py` 纯 CPU，无此参数 |
| 分块 | `--chunk N`（gpu_crack 默认 65536、ctf_tools 默认 16384；**OOM 自动减半重试**） |
| 输出/调度 | `--json`（单行 JSON 结果）、`--quiet`、`--no-progress`、`--checkpoint FILE`、`--reset-checkpoint`（后五项 gpu_crack / ctf_tools / ctf_ext 支持，ctf_misc 不支持） |

统一入口会自动挑解释器、把全局参数挪到正确位置。不确定时**优先用统一入口**：

```powershell
$ctf = '.\scripts\ctf.py'
python $ctf selftest          # 四模块组合自测；全 PASS 且退出码 0 才继续
```

## 2. 使用前自检（3 条命令，缺一不可）

```powershell
# 1) 工具箱与解释器可用？
& python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
# 2) 工具箱自测全绿？
python $ctf selftest
# 3) 目标命令的真实参数（不要凭记忆）：
python $ctf <子命令> --help
```

自测不通过 / 解释器缺失时：**停止并报告具体失败项**，不要改用未验证的脚本硬算。

## 3. 标准工作流（SOP）

1. **归类**：按 §4 决策表把任务映射到 1 个内核；不确定就先 `grep`/人工特征识别。
2. **查参数**：`python $ctf <子命令> --help` 或 CLI_REFERENCE；记录必填项与输入格式（hex/整数/文件）。
3. **小规模探针**：先缩小规模（`--seed-count`/`--tries`/`--chunk`/短数据）跑通命令，确认输出格式与退出码符合预期。
4. **放量执行**：保持分块；命中即早停；记录完整命令与关键输出。
5. **独立复核**：命中的候选必须再用**另一种路径**验证（例如：命中 key → 单独解密/加密比对；命中 LSB → 用原始像素重算）；
   不接受「单次命中」直接当结论。
6. **报告**：给出「命令 + 退出码 + 关键输出行 + 复核结果」，flag 交操作者决定（见 §9）。

## 4. 任务 → 内核 决策表

| 任务特征 | 首选子命令 | 备选/后续 |
|---|---|---|
| 未知流密码、大量候选密钥 | `mtscan`（mt19937 种子） | `chacha`、`ksscore` |
| ChaCha20 密钥流/Poly1305 tag | `chacha`、`poly1305` | — |
| 弱密钥、种子空间小 | `mtscan` | `sboxscan` |
| AES S-box 故障攻击 | `sboxscan`（支持 128/192/256 密钥） | — |
| Reed-Solomon / GF(256) 码 | `rs63` | — |
| 重复 XOR / 单字节 XOR | `xorbrute` | — |
| AES-CTR 流 | `aesctr`（128/192/256，`--nonce-mode le/be`） | — |
| TEA/XTEA/XXTEA/Speck/RC4 | `tea`、`speck`、`rc4` | `xorbrute`（若怀疑被 XOR） |
| 伪随机输出预测 | `lcg` | `mtscan`（mt19937） |
| MD5 口令 | `md5` | — |
| RSA 模数+密文 | `rsa`（五连攻击） | 库函数共模/广播 |
| 未知编码层 | `encode` | — |
| 哈希长度扩展（伪造 MAC/签名） | `mdext`（MD5 / SHA-1） | — |
| 只有输出比特流，要反推 LFSR | `bm`（Berlekamp-Massey） | `lfsrscan`（已知抽头时） |
| 加密 ZIP 口令 | `zipbrute`（字典/掩码） | — |
| 已知 ECC 公钥、私钥很小 | `eccscan`（secp256k1 / P-256） | — |
| 图片 LSB 隐写 | `lsb` | `stegscan` |
| 文件附加/内嵌数据 | `stegscan` | — |
| 两图差异 | `imgdiff` | `bitplane` |
| 单 bit 平面 | `bitplane` | — |
| 溢出偏移（pwntools 风格） | `cyclic` + `cycfind` | — |
| ROP gadget / 二进制保护 | `gadgets`、`elfinfo` | — |

## 5. 输出解析规则（程序化请只依赖这些）

| 规则 | 说明 |
|---|---|
| 成败 | **优先看退出码**：`0` 成功（未命中/未攻破也是 0，需看输出）；`1` 自测失败/校验不匹配；`2` 参数或输入错误 |
| 命中 | 行内含 `★` 视为命中/可疑/候选行；hex 字段一律小写无 `0x` |
| 自测 | 行内含 `PASS`/`FAIL`；`ctf_misc` 收尾 `[selftest] 全部 PASS` |
| 错误 | 错误写 stderr；中文提示文案可能变化，**不要用文案做判断** |
| 未命中 | `ksscore`/`chacha`/`xorbrute`/`rsa` 等「未命中/未攻破」不是错误，退出码仍为 0 |
| 不确定 | 任何解析歧义时重新 `--help` 或直接 import 模块函数取结构化返回值 |

## 6. 端到端模板

### 6.1 弱随机数（mt19937 种子 → 24B 密钥）

```powershell
python $ctf mtscan --key-hex <1~24B 裸 hex> --seed-start 0 --seed-count 1000000 --twist 0
# 命中行: "  ★ 命中 seed=N  key=..."
# 复核: python $ctf mtscan --key-hex <key> --seed-start N --seed-count 1
```

### 6.2 重复 XOR

```powershell
python $ctf xorbrute --data-hex <密文裸 hex> --max-len 1 --top 5
# 复核: 用输出 key 手动 XOR 整段密文，检查是否可读 flag
```

### 6.3 RSA（未知攻击）

```powershell
python $ctf rsa --n <十进制/0x/裸 hex> --e 65537 --c <同格式>
# 命中行: "  ★ 方法=...  明文 hex=...  文本=b'...'"
# 复核: pow(int(m), e, n) == c
```

### 6.4 图像 LSB / 文件 stego

```powershell
python $ctf stegscan --file chal.png     # 先看尾附加/内嵌 magic
python $ctf lsb --img chal.png           # 再看 LSB 候选（score 降序）
# 复核: 用 score/通道/位序参数自己重算提取位流 → 解码文本
```

## 7. 质量门禁（仅当你要修改工具箱代码时）

按顺序执行，全绿才允许交付：

```powershell
python -X utf8 ctf_tools.py selftest     # 以及 ctf_ext.py / ctf_misc.py selftest
python -X utf8 selftest.py
python -X utf8 consistency.py            # 期望: 通过 32/32
python $ctf selftest                                  # 期望: 四模块全 PASS, 退出码 0
```

验收标准：① 已知向量 PASS；② 双后端一致；③ 参数错误是干净报错（rc=1/2，无 traceback）；
④ 新行为同步写入 `CLI_REFERENCE.md`（必要时 README）。

## 8. 失败处理速查

| 现象 | 处理 |
|---|---|
| argc 解析错误（退出码 2） | 重新 `--help` 核对参数名/格式；hex 是否裸 hex、长度是否满足 |
| 找不到模块/文件 | 检查 `CTF_TOOLKIT_DIR`、绝对路径、工作目录 |
| 运行期错误（退出码 1，ctf_misc） | 读 stderr 的 `[cmd] 错误:`；确认文件存在且是正确类型 |
| CUDA OOM | 降 `--chunk`（如 65536→8192）；或 `--backend numpy` |
| torch 不可用 | 用 `--backend numpy`（慢但对）；或设 `CTF_PYTHON` |
| 搜索无结果 | 先验证模型假设（密钥长度/算法变体/端序），不要盲目放大 `--tries`/`--seed-count` |
| 运行时间过长 | 先用小规模探针定位问题；确认吞吐 `bench` 后再估算总时长 |

## 9. 边界与安全

- 不向任何比赛平台提交/上传 flag；只报告并让操作者决定。
- 不读取浏览器 profile、凭据、会话存储等私有数据。
- 候选扫描要设上界（`--seed-count`/`--tries`/`--chunk`），避免无界运行。
- 8 GB 显存按 §8 管理分块；长任务分阶段跑并保存中间结果。
- 对来源不明的「命中」保持怀疑：另一挑战的 writeup 片段、示例 prompt、历史任务残留都可能是误报，必须复核来源。

## 10. 文档索引（优先级从高到低）

1. `CLI_REFERENCE.md` — 每个子命令的参数表/输出形态/退出码/示例（规范性）
2. `README.md` — 项目结构、环境、库 API、性能基线、已知限制
3. `<skill>/references/kernels.md` — 一页速查（题型 → 内核 → 验证向量 → 吞吐）
4. 各模块 `--help` — 运行时唯一真值（与 CLI_REFERENCE 同步维护）
