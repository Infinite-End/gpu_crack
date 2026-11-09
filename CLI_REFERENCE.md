# CTF Toolkit — CLI 参考手册

| 文档属性 | 值 |
|---|---|
| 文档版本 | v3.0（2026-11-09） |
| 适用工具 | gpu_crack toolkit v2（仓库根目录） |
| 校验状态 | 与 `--help` 和各模块自测一致；验收套件 129/129 PASS |
| 配套文档 | [README.md](README.md)（项目说明）、skill 的 `SKILL.md` / `references/kernels.md`（使用策略） |

本手册是**规范性**参考：所有参数名、默认值、输入格式、输出标记、退出码均以本手册为准。
人类可读文案（中文提示、进度条）可能随版本微调，**程序化解析请只依赖本手册第 5 节标注的稳定标记**。

---

## 0. 模块与全局参数速查

### 0.1 四个模块

| 模块 | 脚本 | 职责 | 是否支持 `--backend` / `--chunk` |
|---|---|---|---|
| 批量爆破 | `gpu_crack.py` | mt19937 种子、AES S-box 故障、比特一致度、LFSR | ✅ `--backend` / ✅ `--chunk` |
| 密码学 | `ctf_tools.py` | ChaCha20、Poly1305、RS(63,31)、CRC32、xorbrute、AES-CTR | ✅ `--backend` / ✅ `--chunk` |
| 算法族 | `ctf_ext.py` | TEA/XTEA/XXTEA、Speck、RC4、LCG、MD5、RSA、多层解码 | ✅ `--backend` / ❌ 无 `--chunk` |
| 取证 & Pwn | `ctf_misc.py` | LSB、stegscan、imgdiff、bitplane、cyclic、gadgets、elfinfo | ❌ 纯 CPU（无全局参数） |

### 0.2 全局参数

| 参数 | 取值 | 默认 | 模块 | 说明 |
|---|---|---|---|---|
| `--backend` | `auto` / `torch` / `numpy` | `auto` | gpu_crack, ctf_tools, ctf_ext | `auto` = torch-cuda → torch-cpu → numpy 依次回退 |
| `--chunk` | 正整数 | gpu_crack `65536`；ctf_tools `16384` | gpu_crack, ctf_tools | 每批候选数；显存不足时**自动**减半重试 |
| `--json` | 开关 | 关 | gpu_crack, ctf_tools, ctf_ext | 结果行改为单行 JSON（`{"event":...}`），便于程序/agent 解析 |
| `--quiet` | 开关 | 关 | gpu_crack, ctf_tools, ctf_ext | 抑制横幅与进度，只留结果行 |
| `--no-progress` | 开关 | 关 | gpu_crack, ctf_tools, ctf_ext | 不打印进度/ETA（结果仍输出） |
| `--checkpoint` | 文件路径 | 无 | gpu_crack, ctf_tools, ctf_ext | 断点文件：每批落盘一次，中断后续跑 |
| `--reset-checkpoint` | 开关 | 关 | gpu_crack, ctf_tools, ctf_ext | 忽略已有断点，从头开始 |

> 直接调用模块脚本时，全局参数**必须**写在子命令之前：
> `python gpu_crack.py --backend numpy --chunk 8192 mtscan ...`
> 通过统一入口 `ctf.py` 调用时位置随意：`python ctf.py mtscan ... --backend numpy` 亦可。

### 0.3 子命令总表（30 个功能子命令 + 3 个模块自测）
| # | 模块 | 子命令 | 一句话用途 |
|---|---|---|---|
| 1 | gpu_crack | `bench` | 测当前机器/后端吞吐 |
| 2 | gpu_crack | `mtscan` | 暴力 mt19937 种子 → 24B 密钥匹配 |
| 3 | gpu_crack | `sboxscan` | 枚举 AES-128/192/256 S-box 破坏 (a,b) |
| 4 | gpu_crack | `ksscore` | 候选密钥流 vs 目标的比特一致度打分 |
| 5 | gpu_crack | `lfsrscan` | 多条 LFSR 并行仿真匹配比特流 |
| 6 | ctf_tools | `chacha` | ChaCha20 密钥流生成 / 候选密钥扫描 |
| 7 | ctf_tools | `poly1305` | Poly1305 / ChaCha20-Poly1305 tag 计算与校验 |
| 8 | ctf_tools | `rs63` | GF(256) RS(63,31) syndrome 校验 / 纠错定位 |
| 9 | ctf_tools | `crc32` | 批量 CRC32 / 尾部 1~2 字节爆破 |
| 10 | ctf_tools | `xorbrute` | 重复 XOR 爆破（1~3 字节） |
| 11 | ctf_tools | `aesctr` | AES-128/192/256-CTR 密钥流 / 加解密 |
| 12 | ctf_ext | `tea` | TEA / XTEA / XXTEA 加解密 |
| 13 | ctf_ext | `speck` | Speck32/64、Speck64/128 加解密 |
| 14 | ctf_ext | `rc4` | RC4 加解密 |
| 15 | ctf_ext | `lcg` | LCG 参数恢复 / 状态预测 |
| 16 | ctf_ext | `md5` | MD5 哈希 / 口令爆破 |
| 17 | ctf_ext | `rsa` | RSA 多攻击自动尝试 |
| 18 | ctf_ext | `encode` | 多层编码 BFS 自动解码 |
| 19 | ctf_ext | `mdext` | MD5 / SHA-1 哈希长度扩展 |
| 20 | ctf_ext | `bm` | Berlekamp-Massey 反推 LFSR 多项式 |
| 21 | ctf_ext | `zipbrute` | ZIP 口令爆破（字典 / 掩码） |
| 22 | ctf_ext | `eccscan` | ECC 小私钥区间搜索（k·G 匹配公钥） |
| 23 | ctf_misc | `lsb` | 图像 LSB 位平面穷举提取 |
| 24 | ctf_misc | `stegscan` | 文件级隐写/附加数据扫描 |
| 25 | ctf_misc | `imgdiff` | 两图差异包围盒 + 可视化 |
| 26 | ctf_misc | `bitplane` | 单 bit 平面导出 PNG |
| 27 | ctf_misc | `cyclic` | pwntools 兼容 de Bruijn 模式生成 |
| 28 | ctf_misc | `cycfind` | 溢出偏移定位 |
| 29 | ctf_misc | `gadgets` | ROP gadget 搜索 |
| 30 | ctf_misc | `elfinfo` | ELF 保护/导入/PLT/GOT 解析 |

另有 3 个模块级自测子命令：`ctf_tools.py selftest`、`ctf_ext.py selftest`、`ctf_misc.py selftest`
（`gpu_crack.py` 无 selftest 子命令，主框架自测在 `selftest.py`）。
统一入口的 `ctf.py selftest` 是**特殊命令**：一次跑 `bench + 3 个模块自测`，不是单个模块的子命令。

---

## 1. `gpu_crack.py` — 批量爆破

- 全局参数：`--backend`、`--chunk`（见 0.2）
- 统一入口：`python ctf.py <子命令> ...`

### 1.1 `bench` — 基准测试

```
python ctf.py bench
```

| 项 | 说明 |
|---|---|
| 输入 | 无；可选 `--markdown` 以 Markdown 表格输出（写入 `docs/bench.md` 用） |
| 输出 | `[backend] ...`、`[device ] ...`（torch 时）、`[popcount] ... M/s`、`[mt19937 ] ... K keys/s`、`[aes128/192/256] ... K blocks/s` |
| 退出码 | 0 |
| 备注 | 已内置 3 次预热以排除 JIT 编译，适合在正式搜索前先跑；`--markdown` 时只输出表格 |

### 1.2 `mtscan` — mt19937 种子暴力匹配

```
python ctf.py mtscan --key-hex <1~24B hex> [--seed-start 0] [--seed-count 16777216] [--twist {0,1}]
```

| 参数 | 必填 | 默认 | 格式 | 说明 |
|---|---|---|---|---|
| `--key-hex` | ✅ | — | 裸 hex，1~24 字节 | 目标密钥/密钥流前缀（mt19937 一次最多产 24B） |
| `--seed-start` | ❌ | `0` | 十进制整数 | 起始种子（含） |
| `--seed-count` | ❌ | `16777216`(2^24) | 正整数 | 扫描种子个数 |
| `--twist` | ❌ | `0` | `0`/`1` | `0`=init_genrand 后直接读状态（魔改版常见）；`1`=标准 `std::mt19937` |

| 输出 | 形态 |
|---|---|
| 起始 | `[backend] ...` |
| 进度 | `  {n:,} seeds  {rate} M/s`（约每 20×chunk 个种子，stdout flush） |
| 命中 | `  ★ 命中 seed={s}  key={hex}` |
| 未命中 | `  未命中` |
| 汇总 | `[done] {n:,} 个种子 / {t:.2f}s = {rate} M seeds/s` |

- 退出码：`0` 成功（未命中也是 0，靠输出判断）；`2` 参数/输入错误。
- 示例：`python ctf.py mtscan --key-hex ac2f75c043fbc36709d315f2 --seed-start 0 --seed-count 100000000`
- 备注：命中后会再算一次完整 key 并打印，便于复核。

### 1.3 `sboxscan` — AES-128/192/256 S-box 故障枚举

```
python ctf.py sboxscan --plain-hex <16B 倍数> --cipher-hex <等长> --key-hex <24B>
```

| 参数 | 必填 | 格式 | 说明 |
|---|---|---|---|
| `--plain-hex` | ✅ | 裸 hex，长度非 0 且为 16 的倍数 | 明文块序列 |
| `--cipher-hex` | ✅ | 裸 hex，与 plain 等长 | 目标密文 |
| `--key-hex` | ✅ | 裸 hex，24 字节 | AES-192 密钥 |

| 输出 | 形态 |
|---|---|
| 起始 | `[scan] {65280} 种 (a,b) 组合，{nblocks} 块` |
| 命中验证 | `  ★ (a,b)=({x},{y})  全块验证=通过`（或 `仅首块`） |
| 汇总 | `[done] {n} 组合 / {t:.2f}s = {k} K/s`；无命中时 `  未命中` |

- 退出码：`0` 成功；`2` 长度/格式错误。
- 示例：`python ctf.py sboxscan --plain-hex 67796c696b657b31323334353637387d --cipher-hex <32B hex> --key-hex <48B hex>`
- 备注：故障模型为「第 1 轮 SubBytes 后，输入字节等于 a 的位置输出替换为 `INV_SBOX[b]`」；搜索空间固定 65,280 个 (a,b)（a≠b）。

### 1.4 `ksscore` — 比特一致度打分

```
python ctf.py ksscore --target-hex <L B hex> [--candidates 1048576]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--target-hex` | ✅ | — | 目标密钥流，1 字节以上裸 hex |
| `--candidates` | ❌ | `1048576`(2^20) | 随机候选个数（用于测吞吐/统计淘汰基线） |

| 输出 | 形态 |
|---|---|
| 汇总 | `[done] {n:,} 候选 × {L}B / {t:.2f}s = {rate} M cand/s` |
| 最佳 | `  最佳比特一致度 {score}/{L*8}   候选 {hex}` |

- 退出码：`0`；`2`（空目标、`--candidates<=0`）。
- 备注：候选是随机生成的，结果不可复现；真实用途是先用本内核给「真实候选族」打分（调用库函数 `G.bit_agreement`）。

### 1.5 `lfsrscan` — 多 LFSR 并行匹配

```
python ctf.py lfsrscan --taps "25:10,11,24[;28:12,22,25 ...]" --bits <0/1 串> [--trials 65536] [--rng-seed 1]
```

| 参数 | 必填 | 默认 | 格式 | 说明 |
|---|---|---|---|---|
| `--taps` | ✅ | — | `n:tap1,tap2,...`，多组 `;` 分隔 | n=状态位数，tap 从 1 开始且 ≤ n |
| `--bits` | ✅ | — | 仅含 0/1 的字符串 | 目标输出比特流 |
| `--trials` | ❌ | `65536`(2^16) | 正整数 | 每组随机初始状态数 |
| `--rng-seed` | ❌ | `1` | 整数 | 随机种子（torch/numpy 后端结果一致） |

| 输出 | 形态 |
|---|---|
| 起始 | `[scan] {g} 组 LFSR，目标 {T} 拍` |
| 每组 | `  组{i} (n={n}, taps={[...]}): 最佳 {m}/{T} 位` |
| 汇总 | `[done] {trials*g:,} 组状态 × {T} 拍 / {t:.2f}s` |

- 退出码：`0`；`2`（bits 非 0/1、tap 越界、trials≤0、格式错误）。
- 备注：LFSR 约定为右移 Fibonacci：`fb = xor(选中位)`，输出 `fb`，状态右移并补 `fb` 到最高位。

---

## 2. `ctf_tools.py` — 密码学内核

- 全局参数：`--backend`、`--chunk`（见 0.2）
- 本模块所有 `*-hex` 参数均容忍 `0x` 前缀与内部空白。
- 输入错误统一输出到 stderr（`[错误] ...`）并退出码 `2`。

### 2.1 `selftest` — 已知向量自测

```
python ctf_tools.py [--backend {torch,numpy}] selftest
```

- 说明：`selftest` 是模块级子命令，必须直接运行模块脚本；统一入口的 `ctf.py selftest`
  会跑「四个模块」的组合自测（见第 7 节）。
- 输出：9 条 `[名称] PASS` + `[性能]` 三行。
- 退出码：全部 PASS → `0`；有 FAIL → `1`。
- 覆盖：ChaCha20 RFC8439 / 原版变体、Poly1305 RFC8439、ChaCha20-Poly1305 AEAD、
  RS(63,31) 合法/非法码字、CRC32 向量、AES-192 CTR、xorbrute 经典向量。

### 2.2 `chacha` — ChaCha20 密钥流 / 候选扫描

```
python ctf.py chacha --target-hex <ks hex> (--keys-file <path> | --brute)
          [--nonce-hex <12B 或 8B hex>] [--counter 0] [--tries 1048576] [--key-len 32]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--target-hex` | ✅ | — | 目标密钥流（≥1 字节），命中判定为完全一致 |
| `--nonce-hex` | ❌ | `000000000000004a00000000`（12B） | 12B → RFC7539；8B → 原版（64bit 计数器） |
| `--counter` | ❌ | `0` | 起始计数器（RFC7539 为 32bit，溢出拒绝） |
| `--keys-file` | 二选一 | — | 每行一个 32B hex 密钥；容忍 `0x` 前缀、`#` 注释、空行 |
| `--brute` | 二选一 | — | 随机密钥搜索演示（配 `--tries`/`--chunk`） |
| `--key-len` | ❌ | `32` | 仅支持 32（保留参数，其它值报错） |
| `--tries` | ❌ | `1048576`(2^20) | `--brute` 模式下最多尝试密钥数 |

| 输出（keys-file 模式） | 形态 |
|---|---|
| 起始 | `[scan] {n} 个候选密钥 × {L}B 目标 （...变体说明）` |
| 汇总 | `[done] {rate} K/s` |
| 命中 | `  ★ 命中 #{i} key={hex}` |
| 未命中 | `  未命中；最佳 {score}/{bits} bit (候选 #{i})` |

- 退出码：`0`（未命中也是 0）；`2` 输入错误。
- 示例：`python ctf.py chacha --target-hex 10f1e7e4d13b5915500fdd1fa32071c4 --nonce-hex 000000090000004a00000000 --counter 1 --keys-file keys.txt`
- 备注：`keys-file` 文件里每行必须正好 32 字节密钥；行内可带 `0x`。

### 2.3 `poly1305` — Poly1305 / AEAD tag

```
python ctf.py poly1305 --key-hex <32B> --msg-hex <hex> [--aad-hex <hex>]
          [--nonce-hex <12B>] [--mode {plain,aead}] [--expect-hex <16B>]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--key-hex` | ✅ | — | 32B：默认是 Poly1305 子密钥（otk‖r）；给了 `--nonce-hex` 时视为 ChaCha20 主密钥 |
| `--msg-hex` | ✅ | — | plain 模式=消息；aead 模式=密文 |
| `--aad-hex` | ❌ | 空 | AEAD 附加数据 |
| `--nonce-hex` | ❌ | None | 12B ChaCha20 nonce；自动由主密钥派生 otk（仅 aead 模式） |
| `--mode` | ❌ | `plain` | `plain`=原始 Poly1305；`aead`=ChaCha20-Poly1305 tag |
| `--expect-hex` | ❌ | None | 期望 tag（16B）；不匹配时退出码 1 |

| 输出 | 形态 |
|---|---|
| tag | `[poly1305/{mode}] tag = {hex}` |
| 校验 | `  期望 {hex}  →  匹配 ✅ / 不匹配 ❌` |
| 派生 | `[poly1305] 派生 otk = {hex}`（仅给 nonce 时） |

- 退出码：`0` 成功/匹配；`1` `--expect-hex` 不匹配；`2` 输入错误。
- 示例（RFC 8439 §2.8.2）：
  `python ctf.py poly1305 --mode aead --key-hex 808182...9e9f --nonce-hex 070000004041424344454647 --msg-hex <密文> --aad-hex 50515253c0c1c2c3c4c5c6c7 --expect-hex 1ae10b594f09e26a7e902ecbd0600691`

### 2.4 `rs63` — RS(63,31) 校验 / 纠错

```
python ctf.py rs63 --code-hex <1~255B hex> [--nsym 32] [--fcr N] [--order {auto,msb,lsb}] [--locate]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--code-hex` | ✅ | — | 码字（GF(256)，长度 1~255） |
| `--nsym` | ❌ | `32` | 校验符号数（RS(63,31) 为 32） |
| `--fcr` | ❌ | None | 固定 first consecutive root；不填则自动试 `{0,1}` |
| `--order` | ❌ | `auto` | 系数顺序 msb/lsb；`auto` 自动试两种 |
| `--locate` | ❌ | 关 | 非法码字时尝试单字节纠错定位 |

| 输出 | 形态 |
|---|---|
| 合法 | `[RS(63,31)] 长度 {n}  合法=True  (order={o} fcr={f} nsym={k})` |
| 非法 | `[RS(63,31)] 长度 {n}  合法=False` + `  syndrome = {hex}  (order=... )` |
| 定位 | `  ★ 修正（order=... fcr=...）：pos={p} {old:#04x} → {new:#04x}` 或 `  单字节无法修正（...）` |

- 退出码：`0`（无论合法与否）；`2` 输入错误。
- 示例：`python ctf.py rs63 --code-hex <63B hex> --locate`
- 备注：`--order`/`--fcr` 显式给出时该维度不再自动尝试；`auto` 下四种约定全试。

### 2.5 `crc32` — CRC32 / 尾部爆破

```
python ctf.py crc32 --data-hex <hex> [--target-hex <1~4B hex>]
```

| 参数 | 必填 | 说明 |
|---|---|---|
| `--data-hex` | ✅ | 原始数据（任意长度） |
| `--target-hex` | ❌ | 目标 CRC 值（1~4 字节，按**大端**解释）；给出时在数据尾部爆破 1~2 字节 |

| 输出 | 形态 |
|---|---|
| 无 target | `[crc32] {crc:08x}` |
| 有 target | `[crc32] 目标 {want:08x}，在原数据尾部爆破 1~2 字节` + `  ★ 命中：{data hex}{tail hex}`（可多条） + `  扫描 65792 变体完成`（无命中时另有 `  未命中`） |

- 退出码：`0`（未命中也是 0）；`2` 输入错误。
- 示例：`python ctf.py crc32 --data-hex 89504e47 --target-hex 42e619a2`

### 2.6 `xorbrute` — 重复 XOR 爆破

```
python ctf.py xorbrute --data-hex <hex> [--max-len 3] [--top 10]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--data-hex` | ✅ | — | 待爆破密文（≥1 字节） |
| `--max-len` | ❌ | `3` | 密钥长度上限，允许 1~3（4 字节会 OOM，直接拒绝） |
| `--top` | ❌ | `10` | 输出候选数 |

| 输出 | 形态 |
|---|---|
| 汇总 | `[xorbrute] {L}B, key 长度 ≤{K}, 用时 {t:.2f}s` |
| 候选 | `  score={s:5d} key={keyhex:<12} → {plaintext!r}`（按分数降序） |

- 退出码：`0`；`2` 输入错误。
- 备注：打分=词频权重 + flag 结构（已知前缀重奖）；密文 <8 字节时统计不可靠。

### 2.7 `aesctr` — AES-128/192/256-CTR

```
python ctf.py aesctr --key-hex <24B> --nonce-hex <8B> [--mode {ks,enc,dec}]
          [--data-hex <hex>] [--length 64] [--counter 0] [--out-text]
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--key-hex` | ✅ | — | AES-192 密钥（24B） |
| `--nonce-hex` | ✅ | — | 8B nonce；计数器块 = nonce ‖ counter64 小端 |
| `--mode` | ❌ | `dec` | `ks`=输出密钥流；`enc`/`dec`=对 data 异或 |
| `--data-hex` | ❌（enc/dec 必填） | 空 | 输入数据 |
| `--length` | ❌ | `64` | 仅 `ks` 模式使用：密钥流字节数 |
| `--counter` | ❌ | `0` | 起始 64bit 计数器 |
| `--out-text` | ❌ | 关 | 额外打印 `text = repr(out)` |

| 输出 | 形态 |
|---|---|
| ks | `[aesctr] keystream({N}B) = {hex}` |
| enc/dec | `[aesctr/{mode}] {hex}`（加 `--out-text` 时再一行 `  text = b'...'`） |

- 退出码：`0`；`2` 输入错误。
- 示例：`python ctf.py aesctr --key-hex 000102...17 --nonce-hex 0001020304050607 --mode ks --length 64 --counter 1`

---

## 3. `ctf_ext.py` — 算法族

- 全局参数：`--backend`（无 `--chunk`）。输入错误输出 `ctf_ext.py: error: ...` 到 stderr，退出码 `2`。
- **hex 约定**：`--n/--c/--e/--outs/--mod` 及 Speck 的 `--key-hex/--data-hex` 是**整数**，容忍 `0x` 前缀；
  TEA/RC4/encode 的 `--data-hex` 是**字节串**，必须裸 hex（不接受 `0x`）。

### 3.1 `selftest`

```
python ctf_ext.py [--backend numpy] selftest
```

- 输出：TEA/XTEA/XXTEA/Speck32/64/Speck64/128/RC4/LCG/MD5/RSA（含共模、广播、p-1）/自动解码 的 PASS 行。
- 退出码：`0`=全 PASS；`N`=失败项数（1..N）。注意：`2` 既可能表示 2 项失败，也可能是用法错误，需看输出。

### 3.2 `tea` — TEA / XTEA / XXTEA

```
python ctf.py tea [--mode {tea,xtea,xxtea}] [--op {encrypt,decrypt}] --key-hex <16B> --data-hex <hex>
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | `xtea` | 算法选择 |
| `--op` | `decrypt` | 加/解密 |
| `--key-hex` | 必填 | 必须 16 字节 |
| `--data-hex` | 必填 | tea/xtea：8 的倍数；xxtea 解密：4 的倍数且 ≥8B；xxtea 加密自动按 4 字节补齐 |

- 输出：`[{mode} {op}] {hex}`（输出按**小端**逐 4 字节写出）。
- 退出码：`0`；`2` 长度/格式错误。
- 示例：`python ctf.py tea --mode tea --op encrypt --key-hex 00000000000000000000000000000000 --data-hex 0000000000000000` → `0a3aea4140a9ba94`

### 3.3 `speck` — Speck 单块

```
python ctf.py speck --key-hex <hex> --data-hex <hex> [--op {encrypt,decrypt}]
          [--word-bits {16,32}] [--key-words 4] [--rounds N]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--key-hex` | 必填 | 小端打包（k0 在最低位）；宽度 ≤ word_bits×key_words |
| `--data-hex` | 必填 | 2×word_bits 位的大端整数 |
| `--op` | `encrypt` | 加/解密 |
| `--word-bits` | `32` | 16 或 32 |
| `--key-words` | `4` | 2/3/4；NSA 组合：16/4、32/3、32/4；其它组合必须显式 `--rounds` |
| `--rounds` | None | 轮数覆盖 |

- 输出：`[speck/{op}] w={w} keys={k}  {hex}  ({bytes!r})`
- 退出码：`0`；`2` 参数错误。
- 示例（NSA 向量）：`python ctf.py speck --key-hex 1918111009080100 --data-hex 6574694c --word-bits 16 --key-words 4` → `a86842f2`

### 3.4 `rc4`

```
python ctf.py rc4 --key-hex <1~256B hex> --data-hex <hex>
```

- 输出：`[rc4] {hex}`
- 退出码：`0`；`2`（空 key/超 256B/非法 hex）。

### 3.5 `lcg`

```
python ctf.py lcg [--mod 2**32] --outs x1,x2,x3[,x4...] [--predict 3]
```

| 参数 | 默认 | 格式 | 说明 |
|---|---|---|---|
| `--mod` | `2**32` | 十进制 / `0x...` / 裸 hex / `2**k` / `(2**k)` | 模数 |
| `--outs` | 必填 | 逗号分隔，每项可为十进制/0x/裸 hex | ≥3 个连续输出 |
| `--predict` | `3` | 非负整数 | 预测个数 |

- 输出：`[lcg] 恢复: a={a} c={c}` + `  预测 {n} 个: [ ... ]`
- 退出码：`0`；`2` 参数错误。

### 3.6 `md5`

```
python ctf.py md5 --target <32 hex> [--charset abc...] [--max-len 4]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--target` | 必填 | 32 字符 MD5 hex |
| `--charset` | `a-z0-9` | 爆破字符集 |
| `--max-len` | `4` | 口令最大长度（含空串，长度从 0 开始试） |

- 输出：`[md5] 尝试 {n:,} 次 / {t:.2f}s  命中: b'...'`（或 `未命中`）
- 退出码：`0`（未命中也是 0）；`2` 参数错误。
- 备注：`--max-len`/`--charset` 组合爆炸，注意规模（如 62^4≈1.5e7）。

### 3.7 `rsa` — 多攻击自动尝试

```
python ctf.py rsa --n <整数> [--e 65537] --c <整数>
```

| 参数 | 默认 | 格式 | 说明 |
|---|---|---|---|
| `--n` | 必填 | 十进制 / `0x...` / 裸 hex | 模数（≥3） |
| `--e` | `65537` | 十进制整数 | 公开指数（≥3 的奇数） |
| `--c` | 必填 | 同 `--n` | 密文（≥0） |

- 攻击顺序：small-factor(≤1e5 试除) → Wiener → Fermat(2e5) → Pollard rho → Pollard p-1。
- 输出：`[rsa] n={bits}bit  e={e}` + 各步 `  <method>: 无`；命中时
  `  ★ 方法={method}  明文 hex={hex}  文本={bytes!r}`；全失败 `  未攻破（试试共模/广播/Coppersmith）`。
- 退出码：`0`（未攻破也是 0）；`2` 参数错误。
- 备注：命中结果先用 `m^e ≡ c (mod n)` 校验；共模/广播/已知因子等接口是**库函数**（无 CLI）。

### 3.8 `encode` — 多层编码 BFS 解码

```
python ctf.py encode (--data-hex <hex> | --data <原始文本>)
```

- 解码器：hex / b64 / b32（大小写不敏感）/ rot13 / rot47 / url / 反转 / 单字节 XOR。
- 输出：`  depth {k}: ...` 候选行；命中时 `[decode] 路径={['b64','hex']}` + `  结果=b'flag{...}'`；
  无 flag 特征时结果可能为 None（只打印候选，属正常行为）。
- 退出码：`0`；`2` 参数错误。

---

### 3.9 `mdext` — MD5 / SHA-1 哈希长度扩展

```
python ctf_ext.py mdext --algo md5 --orig-len 23 --digest-hex <32位hex> --append-hex 2661646d696e3d31
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--algo` | | `md5` | `md5` / `sha1` |
| `--orig-len` | ✅ | — | **原消息字节数**（长度扩展的核心输入，错一位结果就废） |
| `--digest-hex` | ✅ | — | 原摘要：MD5 16 字节 / SHA-1 20 字节 |
| `--append-hex` | | 空 | 要追加的数据（如 `2661646d696e3d31` = `&admin=1`） |

输出：`glue`（要接在原消息后的填充）、`tail`（glue+append）、`new_digest`、`forged_len`。
伪造消息 = `原消息 || glue || append`，其摘要即 `new_digest`。

```output
[mdext/md5] 原长度 23 B，追加 8 B
  glue        = 80000000000000000000000000000000000000000000000000000000000000000000b800...
  new_digest  = 3f1c...（32 位 hex）
```

### 3.10 `bm` — Berlekamp-Massey 反推 LFSR

```
python ctf_ext.py bm --bits 1000100110101111...
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--bits` | ✅ | — | 输出比特流（仅 0/1，至少 2L 位才能唯一确定） |

输出最小多项式 `taps`（`b[n] = Σ c_i·b[n-i]`，tap 从 1 开始）、线性复杂度 `L`、预测下 32 位。
与 `lfsrscan` 互补：`bm` 已知比特流求抽头，`lfsrscan` 已知抽头扫初始状态。

### 3.11 `zipbrute` — ZIP 口令爆破

```
python ctf_ext.py zipbrute --zip chal.zip --wordlist rockyou.txt --jobs 8
python ctf_ext.py zipbrute --zip chal.zip --charset abc123 --min-len 1 --max-len 5
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--zip` | ✅ | — | zip 文件路径（取第一个非目录条目） |
| `--wordlist` | 二选一 | — | 字典文件，每行一个口令，`#` 开头为注释 |
| `--charset` | 二选一 | — | 掩码字符集（配合 `--min-len`/`--max-len`） |
| `--min-len` / `--max-len` | | `1` / `4` | 掩码长度区间 |
| `--jobs` | | `1` | 并行进程数（Windows 上 spawn 开销大，建议 ≤ 8） |
| `--max-bytes` | | `1048576` | 每个候选最多读取的明文字节（越小越快） |

输出 `/★ 口令 = '...'`；未命中输出 `[zipbrute] 未命中`。仅支持 ZipCrypto/无加密条目
（WinZip AES 走 PBKDF2，本项目未实现）。

### 3.12 `eccscan` — ECC 小私钥区间搜索

```
python ctf_ext.py eccscan --curve secp256k1 --pub-hex 04<64位x><64位y> --start 0 --count 1048576
python ctf_ext.py eccscan --curve p256 --px 0x... --py 0x... --jobs 4
```

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--curve` | | `secp256k1` | `secp256k1` / `p256` |
| `--pub-hex` | 二选一 | — | 未压缩公钥 `04‖x‖y`（65 字节 hex） |
| `--px` / `--py` | 二选一 | — | 公钥坐标（十进制 / `0x` / 裸 hex） |
| `--start` | | `0` | 起始私钥（含） |
| `--count` | | `1048576` | 搜索个数 |
| `--jobs` | | `1` | 并行进程数 |

实现：起始点一次标量乘（仿射 + 逆元），之后 Jacobian 混合加法逐点递增 G，
用交叉相乘 `X == x·Z²`、`Y == y·Z³` 判等，**全程不需逆元**。单核实测约 1.1 M keys/s。

## 4. `ctf_misc.py` — 取证 & Pwn

- 无全局参数（纯 CPU）。运行期错误（文件不存在/非法图像/解析失败）输出 `[{cmd}] 错误: ...` 到 stderr，退出码 `1`。
- 本模块除 `cycfind --data-hex`（裸 hex）外没有 hex 参数。

### 4.1 `selftest`

```
python ctf_misc.py selftest
```

- 输出：`[LSB ...] PASS` … `[selftest] 全部 PASS`。
- 退出码：`0` 全 PASS；`1` 有失败。

### 4.2 `lsb` — LSB 位平面穷举

```
python ctf.py lsb --img <图片路径>
```

- 扫描空间：7 种通道模板 × bit0~2 × 行/列序 × 序列/像素交织 × MSB/LSB。
- 输出：`[lsb] {path}  mode={m} size={wh}  命中 {n} 组` + 每条
  `  score={s:3d} ch={chs} bit={b} {row|col}/{seq|pix} {msb|lsb} → {info}` + `        head={bytes!r}`。
- 评分：`100` magic@0；`90` flag 特征；`80` 内嵌 magic(len≥3, offset<256)；`60` 可打印含 flag；`40` 前缀可打印；`20` 整体可打印。
- 退出码：`0`；`1` 图像无法识别/文件不存在。

### 4.3 `stegscan` — 文件级扫描

```
python ctf.py stegscan --file <文件路径>
```

- 输出块：
  - `[stegscan] {path}  {size:,} 字节`
  - `  内嵌 magic: [(offset, 'NAME'), ...]`（21 条签名）
  - PNG：`  PNG: IEND 之后还有 {n} 字节`（>0 时带 `★ 可疑`）+ `    尾部 head={bytes!r}` + 尾部 magic 提示；
    异常 chunk 时 `  ⚠ chunk @0x... ...`
  - JPEG：`  JPEG: FFD9 之后还有 {n} 字节`
  - zlib：`  ★ zlib@0x{off:x} 解出 {n} 字节: {head!r}`
  - 字符串：`  可打印串 {n} 段，可疑 {m} 段` + `    @0x{off:x} {bytes!r}`
- 退出码：`0`（有可疑也是 0）；`1` 文件不存在/不可读。

### 4.4 `imgdiff` — 两图差异

```
python ctf.py imgdiff --a <a.png> --b <b.png> [--out <vis.png>]
```

- 输出：`[imgdiff] 差异像素 {n} / {total}  最大差 {d}` + `  包围盒 x[{x0}..{x1}] y[{y0}..{y1}]` +
  `  LSB 层差异像素 {n}` + `  差异图已保存 {path}`（给了 `--out` 时）。
- 尺寸不同时自动裁剪到较小尺寸并提示。
- 退出码：`0`；`1` 图像无法读取。

### 4.5 `bitplane` — bit 平面导出

```
python ctf.py bitplane --img <图片> [--plane 0] [--channel 0] [--out <plane.png>]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--plane` | `0` | 0~7（越界报错） |
| `--channel` | `0` | 0~2（RGB 转换后；越界报错） |
| `--out` | None | **文件路径**（不是目录）；父目录自动创建 |

- 输出：`[bitplane] plane={p} ch={c} → {path}  非零像素 {n}`（无 `--out` 时省略箭头段）。
- 退出码：`0`；`1` 参数/图像错误。

### 4.6 `cyclic` / `cycfind` — pwntools 兼容模式

```
python ctf.py cyclic [--length 200]
python ctf.py cycfind --data-hex <裸 hex> [--length 4096]
```

- `cyclic`：输出 `b'...'`；上限为 de Bruijn 序列长度（字母表 26，n=4 → 26^4=456976），超长报错。
- `cycfind`：`[cycfind] 模式长 {L}  查找 {hex}` + `  偏移 = {i}`（找不到为 `-1`）。
- 兼容性：`cyclic(20) == b'aaaabaaacaaadaaaeaaa'`（与 pwntools 相同）。
- 退出码：`0`（找不到也是 0；偏移在 stdout，不要当退出码用）；`1` 参数/长度错误。

### 4.7 `gadgets` — ROP gadget 搜索

```
python ctf.py gadgets --file <binary> [--filter "pop rdi"] [--max-insn 6] [--bits {32,64}]
```

- 输出：`[gadgets] {path} 找到 {n} 条（过滤 "{filter}"）` + `  0x{addr:x}: insn ; insn ; ...`
- 约定：以 ret(c3)/retf(cb)/ret imm16(c2)/syscall(0f05) 为锚，向前回溯最多 32 字节；
  结尾指令之前不允许再出现控制流结尾；支持 x86-64 / i386。
- 退出码：`0`；`1` 文件不存在（未安装 capstone 时打印提示并返回 0 条）。

### 4.8 `elfinfo` — ELF 概览

```
python ctf.py elfinfo --file <elf>
```

- 输出：`[elfinfo] {path}` + `  arch={arch}  bits={bits}  type={type}  entry={hex}` +
  `  NX: {是/否} | PIE: {是/否} | Canary: {是/否} | RELRO(Full): {是/否}` +
  `  危险导入: [...]` + `  .plt: {addr} (size {n})` + `  .got: {addr} (size {n})`（节不存在则省略）。
- 退出码：`0` 解析成功；`1` 非 ELF / 文件不存在 / 截断。

---

## 5. 稳定契约（供程序化解析）

以下内容视为**稳定接口**；其它中文文案、进度行、耗时/吞吐数字不保证逐字不变。

| 类别 | 稳定约定 |
|---|---|
| 退出码 | 见第 6 节矩阵，**优先用它判断成败** |
| 错误 | 一律写 stderr；不会把 traceback 抛给终端用户（除内部 bug） |
| 命中标记 | 输出中含 `★` 的行表示「候选/命中/可疑」；命中行里不要依赖固定词序，用 hex 字段正则提取 |
| 自测 | 行内含 `PASS` / `FAIL`；ctf_misc 另有 `[selftest] 全部 PASS`；退出码 0/1 |
| 数值输出 | 十六进制一律小写、无 `0x` 前缀（如 `key=...`、`tag = ...`、`明文 hex=...`） |
| 编码 | stdout/stderr 均为 UTF-8；Windows 控制台建议 `-X utf8` 或设置 `PYTHONUTF8=1`（统一入口已代为设置） |
| 机器可复现 | `mtscan`/`sboxscan`/`xorbrute`/`rs63`/`crc32`/`rsa`/`speck`/`tea`/`rc4`/`cycl*`/`lsb`/`stegscan`/`gadgets`/`elfinfo` 确定性；`ksscore`/`chacha --brute` 用 RNG（不可复现），`lfsrscan --rng-seed` 固定即可复现 |

---

## 6. 退出码矩阵

| 情况 | gpu_crack | ctf_tools | ctf_ext | ctf_misc | 统一入口 ctf.py |
|---|---|---|---|---|---|
| 成功 | 0 | 0 | 0 | 0 | 转发子进程 |
| `selftest` 全 PASS | — | 0 | 0 | 0 | 0 |
| `selftest` 有失败 | — | 1 | 失败项数 N | 1 | 非 0（OR 聚合） |
| 参数/输入格式错误 | 2 | 2 | 2 | 1 | 转发（通常 2 或 1） |
| 校验不匹配 | — | `poly1305 --expect-hex` → 1 | — | — | 转发 |
| 运行期/文件错误 | — | — | — | 1 | 转发 |
| 未命中/未攻破/找不到 | 0 | 0 | 0 | 0 | 0 |
| 未知模块 / 无参数 | — | — | — | — | 2 / 1 |
| `help` | — | — | — | — | 0 |

> 结论：**解析结果用退出码 + 输出标记；`2` 与 `1` 的精确语义按模块见上表**。

---

## 7. 统一入口 `ctf.py`

位置：`skill\scripts\ctf.py`（仓库内镜像；skill 安装在别处时用 `CTF_TOOLKIT_DIR` 指回仓库根）

```
python ctf.py <module-or-subcommand> [参数...]
python ctf.py selftest                 # 一次跑四个模块（gpu bench + 3 个 selftest）
python ctf.py help                     # 打印帮助
```

- 模块名即子命令名（`bench`、`mtscan`、`chacha`、`rsa`、`lsb`…），见 0.3 总表。
- 自动挑选解释器：`CTF_PYTHON`（若设置且存在）→ 自动探测（当前解释器、PATH 上的 `python3`/`python`、Windows `py -0p` 登记的所有版本、仓库内 `.venv`/`venv`）中第一个 `import torch` 且 `torch.cuda.is_available()` 为真的 → 都不行则回落当前解释器（走 numpy 后端）。
- `CTF_TOOLKIT_DIR` 可覆盖工具箱目录（默认：`ctf.py` 向上两级的仓库根目录）。
- 全局参数 `--backend/--chunk` 放在模块名前后均可；对不支持它们的模块（ctf_misc）会提示忽略。
- 退出码：见第 6 节。
