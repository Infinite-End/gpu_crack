# 内核速查表（一页版）

| 文档属性 | 值 |
|---|---|
| 文档版本 | v3.1（2026-11-09） |
| 定位 | **选型速查**：题型 → 内核 → 验证向量 → 吞吐。完整参数/输出/退出码以仓库根 `CLI_REFERENCE.md` 为准 |
| 环境 | 仓库根目录；解释器 PATH 上的 `python3`/`python`（需 torch 2.14.1+cu130 才有 GPU，否则自动回落 numpy） |
| 性能口径 | RTX 5070 Laptop 8 GB 本机复测；随驱动/温度波动 2~3 倍，先跑 `bench` |

## 1. 题型 → 内核 决策表

| 任务特征 | 子命令 | 备注 |
|---|---|---|
| 弱随机数 / mt19937 种子空间 | `gpu_crack.py mtscan` | `--twist 0/1` 区分魔改与标准实现 |
| AES S-box 故障攻击 | `gpu_crack.py sboxscan` | 枚举 65,280 个 (a,b) |
| 候选密钥流统计淘汰 | `gpu_crack.py ksscore` / 库函数 `bit_agreement` | 先廉价判据再全量验证 |
| 多 LFSR 抽头穷举 | `gpu_crack.py lfsrscan` | `n:tap,...` 多组 |
| ChaCha20 密钥流 / 候选密钥 | `ctf_tools.py chacha` | 12B/8B nonce 自动选变体 |
| Poly1305 / ChaCha20-Poly1305 | `ctf_tools.py poly1305` | `--expect-hex` 不匹配 rc=1 |
| RS(63,31) / GF(256) 码 | `ctf_tools.py rs63` | `--locate` 单字节纠错 |
| CRC32 / 尾部校验 | `ctf_tools.py crc32` | 尾部 1~2 字节爆破 |
| 重复 XOR（1~3 字节） | `ctf_tools.py xorbrute` | 密文 <8B 时不可靠 |
| AES 密钥长度/模式但只知部分信息 | `ctf_tools.py aesctr` | 128/192/256；`--nonce-mode le/be`；计数器块 8/12/16B |
| TEA/XTEA/XXTEA | `ctf_ext.py tea` | 输出小端；key 固定 16B |
| Speck32/64、Speck64/128 | `ctf_ext.py speck` | `--word-bits/--key-words/--rounds` |
| RC4 | `ctf_ext.py rc4` | key 1~256B |
| LCG 参数恢复/预测 | `ctf_ext.py lcg` | `--mod` 支持 `2**32` |
| MD5 口令爆破 | `ctf_ext.py md5` | `--charset/--max-len`，含空口令 |
| RSA 未知攻击 | `ctf_ext.py rsa` | 五连：小因子→Wiener→Fermat→rho→p-1 |
| 未知编码层 | `ctf_ext.py encode` | BFS 深度 4；无 flag 特征只打印候选 |
| MAC/签名可伪造（哈希长度扩展） | `ctf_ext.py mdext` | 需已知原消息长度；MD5 16B / SHA-1 20B 摘要 |
| 已知输出比特流、要反推 LFSR | `ctf_ext.py bm` | Berlekamp-Massey，需 ≥2L 位；与 `lfsrscan` 互补 |
| 加密 ZIP 口令 | `ctf_ext.py zipbrute` | 字典或掩码；仅 ZipCrypto，WinZip AES 未支持 |
| 已知 ECC 公钥但私钥很小 | `ctf_ext.py eccscan` | secp256k1 / P-256，单核 ~1.1 M keys/s |
| 图片 LSB 隐写 | `ctf_misc.py lsb` | 7 通道模板 × bit0~2 × 序 × 位序 |
| 文件附加/内嵌数据 | `ctf_misc.py stegscan` | 21 条内嵌 magic；PNG/JPEG 尾部 |
| 两图差异 | `ctf_misc.py imgdiff` | 包围盒 + LSB 差异 + 可视化 |
| 单 bit 平面 | `ctf_misc.py bitplane` | `--out` 是文件路径 |
| 溢出偏移 | `ctf_misc.py cyclic` / `cycfind` | 与 pwntools n=4 一致 |
| ROP gadget | `ctf_misc.py gadgets` | ret/retf/syscall 结尾 |
| ELF 保护/导入 | `ctf_misc.py elfinfo` | NX/PIE/RELRO/PLT/GOT/导入 |

## 2. `gpu_crack.py` 内核

| 内核 | 用途 | 关键输入 | 验证向量 | 吞吐（本机） |
|---|---|---|---|---|
| `mt19937_keys` / `mtscan` | 种子 → 24B 密钥 | 24B 目标 hex；`--twist` | 纯 Python MT 参考（twist 0/1） | ~35 M 种子/s（bench）；端到端 ~9 M/s |
| `aes192_ecb` / `sboxscan` | 批量 AES-192 单块 / 故障枚举 | 明文/密文/24B key | NIST AES-192 向量 | ~1.6 M 块/s；65,280 组合 ≈ 0.4 s |
| `popcount_matrix` / `ksscore` | 比特一致度 | 目标密钥流 | 与逐位 popcount 参考一致 | ~900 M/s |
| `bit_agreement` | 候选 vs 目标打分 | `[N,L]` 候选 | torch == numpy | 随 GPU |
| `lfsrscan` | 多 LFSR 并行 | `taps`、`bits` | 双后端同 `--rng-seed` 结果一致 | 视状态数 |

## 3. `ctf_tools.py` 内核

| 内核 | 用途 | 验证向量 | 吞吐（×64B） |
|---|---|---|---|
| `chacha20_keystream` / `chacha` | ChaCha20 密钥流、候选扫描 | RFC 8439 §2.4.2；pycryptodome 原版变体 | ~590 K keys/s |
| `poly1305` / `poly1305_tag` | Poly1305、AEAD tag | RFC 8439 §2.5.2 / §2.8.2；cryptography AEAD | 标量（过滤后终验） |
| `rs63_syndromes` / `rs63_check` / `rs63` | RS(63,31) 校验与纠错 | 自建编码器零 syndrome + 单字节定位 | ~69 K 码字/s |
| `crc32_batch` / `crc32` | CRC32 / 尾部爆破 | zlib（含随机 97B） | ~3.4 M/s |
| `xorbrute` | 1~3 字节 XOR 爆破 | 经典 cryptopals 向量 + `flag{...}` 向量 | 3 字节/15B ≈ 6 s |
| `aesctr_keystream` / `aesctr` | AES-192-CTR | NIST SP800-38A F.5.3 | 随 GPU |

## 4. `ctf_ext.py` 内核

| 内核 | 用途 | 验证向量 | 备注 |
|---|---|---|---|
| `tea_encrypt_block` / `xtea_*` / `xxtea_*` / `tea` | TEA 家族 | TEA(0,0,zero key)=`41ea3a0a94baa940`；多组往返；XXTEA 多长度 | 输出小端；key 必须 16B |
| `speck_encrypt/decrypt` / `speck` | Speck | NSA：`a86842f2`、`8c6fa548454e028b` | NSA 组合 16/4、32/3、32/4 |
| `rc4` | RC4 | "Key"/"Plaintext"→`bbf316e8d940af0ad3` | key 1~256B |
| `lcg_recover` / `lcg_predict` / `lcg` | LCG | 1103515245/12345 恢复并预测 | ≥3 个连续输出 |
| `md5_batch` / `md5_crack` / `md5` | MD5 | "abc"→`900150983cd24fb0d6963f7d28e17f72` | 含空口令 |
| `rsa_attack`（+共模/广播/p-1/已知因子库函数） | RSA | 构造小密钥全攻击链；结果 `m^e≡c` 校验 | CLI 为五连 |
| `auto_decode` / `encode` | 多层解码 | b64+hex、hex+b64+b32 等 9 类链 | BFS 深度 4 |

## 5. `ctf_misc.py` 内核

| 内核 | 用途 | 验证方式 | 备注 |
|---|---|---|---|
| `lsb_scan` / `lsb` | LSB 提取 | 种植 payload 恢复 + 无假阳性 | 评分：magic100 > flag90 > 内嵌magic80 > 可打印 |
| `stegscan` | 文件扫描 | 尾附加长度/内嵌 magic/zlib 解压对照 ground truth | 21 条 magic；PNG chunk 链解析 |
| `imgdiff` | 两图差异 | 已知包围盒 + 像素精确输出 | 尺寸不同自动裁剪 |
| `bitplane` | 位平面导出 | 与 `((px>>plane)&1)*255` 逐像素一致 | `--plane 0..7 --channel 0..2` |
| `cyclic` / `cycfind` | de Bruijn | `cyclic(20)=b'aaaabaaacaaadaaaeaaa'`（pwntools） | 上限 26^4 |
| `find_gadgets` / `gadgets` | ROP | 手工机器码集合相等 + 真实 ELF 计数 | 32B 回溯窗；排除执行不到的假链 |
| `elfinfo` | ELF 解析 | 与 GNU readelf 对 9 个 ELF 零差异 | 解析 PT_GNU_STACK/RELRO/BIND_NOW/DF_1_PIE |

## 6. 通用约定

- **入口**：优先 `python scripts/ctf.py <子命令> ...`（本 skill 内）；参数不确认时先 `--help`。
- **hex 格式族**（重要，容易踩）：
  - `ctf_tools` 的所有 `*-hex` 容忍 `0x` 前缀与空白；
  - `ctf_ext` 的**整数**参数（`--n/--c/--e/--outs/--mod`、Speck 的 key/data）容忍 `0x`；**字节串**参数（tea/rc4/encode 的 data、`--target`）必须裸 hex；
  - `gpu_crack`、`ctf_misc` 全部要求裸 hex（无 `0x`、无空白）。
- **退出码**：`0` 成功（未命中/未攻破也是 0）；`1` 自测失败/校验不匹配；`2` 参数或输入错误。
- **解析**：只依赖退出码与 `★`/`PASS`/`FAIL` 等稳定标记；不要把中文提示文案当接口。
- **随机性**：`ksscore`、`chacha --brute` 不可复现；`lfsrscan --rng-seed` 固定即复现；其余确定性。
- **后端**：`--backend numpy` 结果应与 `torch` 一致；OOM 先减 `--chunk`。
- **组合自测**：`python ctf.py selftest` 必须全 PASS 且退出码 0，才可信赖后续结果。

## 7. 常见失败模式

| 现象 | 原因/处理 |
|---|---|
| `unrecognized arguments: --chunk` | 直接调模块脚本时全局参数必须在子命令前；或改用 `ctf.py` |
| `[错误] --key-hex ...` | hex 长度/格式不符；核对是否为裸 hex |
| `rs63` 一直 `合法=False` | 约定不对：加 `--locate` 让 auto 判定，或显式 `--order/--fcr` |
| `xorbrute` top1 不像明文 | 密文太短（<8B）或不是重复 XOR；换 `--max-len`/结合已知明文 |
| `mtscan` 扫不到 | 确认 `--twist` 0/1、目标是否只取前 k 字节、种子范围是否覆盖 |
| `gadgets` 为空 | 未装 capstone（会提示）；或二进制架构不对，加 `--bits 32` |
| `elfinfo` 退出码 1 | 输入不是 ELF / 被截断 |
| CUDA OOM | 减 `--chunk`（65536→8192→1024）或 `--backend numpy` |
| 首次调用慢 | torch 为 sm_120 现场编译 kernel；先 `bench` 预热再计时 |
