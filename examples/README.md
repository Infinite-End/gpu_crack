# examples —— 可直接跑通的示例输入

生成（离线、无网络依赖）：

```
python examples/make_examples.py --clean
```

生成物在 `examples/out/`（这些文件不提交进 git，见 `.gitignore`）。

## 1. mt19937 种子爆破 → `mt19937_keystream.txt`

种子 `987654` 的前 24 字节密钥流（hex）。离线可复现，跑起来必然命中。

```
python skill/scripts/ctf.py mtscan --key-hex <文件内容> --seed-start 900000 --seed-count 200000
# ★ 命中 seed=987654
```

## 2. RSA 小指数/相近素数 → `rsa_fermat.json`

`p=1000000007`、`q=1000000009`（相差 2），Fermat 分解秒解。

```
python skill/scripts/ctf.py rsa --n <n> --e 65537 --c <c>
```

## 3. ECC 小私钥 → `ecc_small_k.json`

secp256k1、私钥 `1337`、未压缩公钥 `04‖x‖y`。

```
python skill/scripts/ctf.py eccscan --curve secp256k1 --pub-hex <pub_hex> --start 0 --count 5000
# ★ 私钥 k = 1337
```

## 4. LFSR 反推抽头 → `lfsr_bits.txt`

Galois 右移 LFSR（抽头掩码 `0xB400`，初值 `0xACE1`）的 128 拍输出。`lfsr_bits_meta.txt`
里已经用 `bm` 自检过：`L=16`、`taps=[1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1]`。

```
python skill/scripts/ctf.py bm --bits <文件内容>
```

## 5. 图像 LSB 隐写 → `lsb_flag.png`

64×64 RGB 图，flag 写进 R 通道最低位（MSB-first，最后一个字节补 `\x00`）。

```
python skill/scripts/ctf.py lsb --img examples/out/lsb_flag.png
# ★ 命中 3 组，其中 head 含 b'flag{example_lsb_ok}'
```

## 6. 口令字典 → `wordlist.txt`

8 行假口令，配合自带加密 zip 使用（仓库不内置加密 zip，需要自备）：

```
python skill/scripts/ctf.py zipbrute --zip your.zip --wordlist examples/out/wordlist.txt --jobs 4
```

## 说明

- 所有示例都是**合成数据**，不含任何真实比赛题目或 flag。
- `examples/out/` 已在 `.gitignore` 中排除，避免把生成物提交进仓库。
- 生成脚本自身会在 LFSR 一项上做 BM 自检，属可验证的正确性检查。
