# examples —— 可直接跑通的示例输入

生成（离线、无网络依赖）：

```
python -X utf8 examples/make_examples.py --clean
```

生成物在 `examples/out/`（这些文件不提交进 git，见 `.gitignore`）。

**自动回归**：生成后逐条复跑下面 6 条示例命令并汇总退出码（无 GPU 的 CI 也能跑，
全部走 `--backend numpy`）：

```
python -X utf8 examples/make_examples.py --check
# [examples] --check: 6/6 PASS   （任一失败则退出码非 0）
```

`--check` 是 `examples/README.md`（本文件）命令的单一真源实现（见
`examples/make_examples.py` 的 `build_checks()`）：先验证生成幂等（重复生成哈希一致），
再用 `sys.executable` 子进程逐条执行、校验退出码与输出标记。下面命令均可直接粘贴，
在仓库根目录运行（值都是确定性生成的，可直接复现）。

## 1. mt19937 种子爆破 → `mt19937_keystream.txt`

种子 `987654` 的前 24 字节密钥流（hex）。离线可复现，跑起来必然命中。

```
python skill/scripts/ctf.py --backend numpy mtscan \
  --key-hex 959cf346958d25f951e767ca18b3da0e1afb1a535b1b645e \
  --seed-start 900000 --seed-count 200000
# ★ 命中 seed=987654
```

## 2. RSA 相近素数 → Fermat 分解 → `rsa_fermat.json`

`p=1000000007`、`q=1000000009`（相差 2），Fermat 分解秒解。

```
python skill/scripts/ctf.py --backend numpy rsa \
  --n 1000000016000000063 --e 65537 --c 247132336671311844
# ★ 方法=...  明文 hex=435446  文本=b'CTF'
```

## 3. ECC 小私钥 → `ecc_small_k.json`

secp256k1、私钥 `1337`、未压缩公钥 `04‖x‖y`。

```
python skill/scripts/ctf.py --backend numpy eccscan --curve secp256k1 \
  --pub-hex 04db0c51cc634a4096374b0b895584a3ca2fb3bea4fd0ee2361f8db63a650fcee67ec0bd2baea1ae184bd16fd397b0e64d5d28257f85836486367fe33cc5b6e6a0 \
  --start 0 --count 5000
# ★ 私钥 k = 1337
```

## 4. LFSR 反推抽头 → `lfsr_bits.txt`

Galois 右移 LFSR（抽头掩码 `0xB400`，初值 `0xACE1`）的 128 拍输出。`lfsr_bits_meta.txt`
里已经用 `bm` 自检过：`L=16`。

```
python skill/scripts/ctf.py --backend numpy bm \
  --bits 10000111001000110100011011011100101100001101110111101110111110001111110111000011010111001011111101011100010100110100010111101100
# 线性复杂度 L=16
```

## 5. 图像 LSB 隐写 → `lsb_flag.png`

64×64 RGB 图，flag 写进 R 通道最低位（MSB-first，最后一个字节补 `\x00`）。

```
python skill/scripts/ctf.py lsb --img examples/out/lsb_flag.png
# ★ 命中 3 组，其中 head 含 b'flag{example_lsb_ok}'
```

## 6. ZIP 口令爆破 → `locked.zip` + `wordlist.txt`

`make_examples.py` 用标准 ZipCrypto（PKWARE 传统加密）生成 `locked.zip`：单条目
`flag.txt`，口令 `hunter2`（`wordlist.txt` 第 8 行），因此示例无需自备加密 zip。

```
python skill/scripts/ctf.py --backend numpy zipbrute \
  --zip examples/out/locked.zip --wordlist examples/out/wordlist.txt --jobs 1
# ★ 口令 = 'hunter2'
```

## 说明

- 所有示例都是**合成数据**，不含任何真实比赛题目或 flag。
- `examples/out/` 已在 `.gitignore` 中排除，避免把生成物提交进仓库。
- 生成脚本自身做 LFSR/BM 自检与 ZipCrypto 往返校验（`--check`），属可验证的正确性检查。
- `zipbrute` 仅支持 ZipCrypto/无加密条目（WinZip AES 未实现），`locked.zip` 即 ZipCrypto 格式。
