# 吞吐基准

由下面这条命令生成，**不要手改**（表格走 stdout，设备信息走 stderr）：

```
python -X utf8 gpu_crack.py bench --markdown    # stdout 即下表；stderr 为 [backend]/[device ]
```

设备：NVIDIA GeForce RTX 5070 Laptop GPU  VRAM 7.9 GiB；批量 16384（AES）/ 65536（mt19937）/ 4194304（popcount）。
数字随驱动、温度与并发负载波动（本轮实测 ±20%），**以本机 `bench` 即时输出为准**。

## 本机实测

| 内核 | 吞吐 | 批量 | 耗时 |
|---|---|---|---|
| `popcount` | 899.9 M 个/s | 4194304 个 32bit | 5 ms |
| `mt19937` | 26814.7 K keys/s | 65536 个种子 | 2 ms |
| `aes128` | 1903.9 K blocks/s | 16384 块 | 9 ms |
| `aes192` | 1670.1 K blocks/s | 16384 块 | 10 ms |
| `aes256` | 1461.4 K blocks/s | 16384 块 | 11 ms |

## 其它内核（v3.2 复测，非同一次 bench）

| 内核 | 吞吐 | 条件 |
|---|---|---|
| `mt19937_keys`（nbytes=12 / 3） | ~62 M / ~294 M seeds/s | 262144 种子稳态 |
| `mtscan` 端到端 | 3M 种子 8.2 M/s；10M 种子 20.5 M/s | 首块 ~0.32 s CUDA 初始化为固定成本 |
| ChaCha20 | ~586 K keys/s（numpy）/ ~620 K（torch） | ×64B，min-of-5 |
| RS63 | ~69 K codewords/s | 8192×63B syndrome |
| CRC32 | ~3.56 M/s | ×64B |
| xorbrute | 3 字节 / 16B 密文 ≈ 6.7 s | 16.7M 候选 |
| `lsb` 大图 | 3000×3000 RGB ≈ 6.5 s | 8 GB 内存可跑 4000×4000 级 |
| `stegscan` 45 MB PNG | ≈ 1.5 s | 向量化 printable_runs |
