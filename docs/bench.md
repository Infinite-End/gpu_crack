# 吞吐基准

由下面这条命令生成，**不要手改**（表格走 stdout，横幅/设备信息走 stderr）：

```
python -X utf8 gpu_crack.py bench --markdown       # stdout 即下表
python -X utf8 gpu_crack.py bench --profile        # 附逐内核 GPU 时间/launch 占比
```

设备：NVIDIA GeForce RTX 5070 Laptop GPU  VRAM 7.9 GiB；批量 16384（AES）/ 65536（mt19937）/ 4194304（popcount）。
计时为预热 + min-of-5；数字随驱动、温度与并发负载波动（±20%），**以本机 `bench` 即时输出为准**。

## 本机实测（v3.5，2026-11-10）

| 内核 | 吞吐 | 批量 | 耗时 |
|---|---|---|---|
| `popcount` | 948.7 M 个/s | 4194304 个 32bit | 4 ms |
| `mt19937` | 37709.9 K keys/s | 65536 个种子 | 2 ms |
| `aes128` | 1909.5 K blocks/s | 16384 块 | 8 ms |
| `aes192` | 1614.6 K blocks/s | 16384 块 | 10 ms |
| `aes256` | 1615.2 K blocks/s | 16384 块 | 10 ms |

## `bench --profile`（v3.5 新增）

逐内核输出「launch 次数 / GPU 时间 / wall / GPU 占比」，用来判断 kernel 是否 launch-bound：

```output
[profile] popcount      30.00 次 launch/次  GPU    4.660 ms/次  wall    4.474 ms/次  GPU 占比 100.0%
[profile] mt19937      154.00 次 launch/次  GPU    0.863 ms/次  wall    1.947 ms/次  GPU 占比  44.4%
[profile] aes192       1045.00 次 launch/次  GPU    1.366 ms/次  wall   12.781 ms/次  GPU 占比  10.7%
[profile] native_mt_t0   4.00 次 launch/次  GPU    2.716 ms/次  wall    2.843 ms/次  GPU 占比  95.5%
[profile] native_mt_t1   4.00 次 launch/次  GPU    5.890 ms/次  wall    5.766 ms/次  GPU 占比 100.0%
[profile] native_lfsr    5.00 次 launch/次  GPU    0.101 ms/次  wall    0.213 ms/次  GPU 占比  47.5%
```

## 其它内核（v3.5 复测，非同一次 bench）

| 内核 | 吞吐 | 条件 |
|---|---|---|
| `mt19937_keys`（numpy 后端） | ~2.5 M 种子/s | 无 GPU 时 |
| `mtscan` native fuse（twist=0） | **16.3 G 种子/s**（50M=3.1 ms，1 次 launch） | 计算段，不含启动 |
| `mtscan` native fuse（twist=1） | **1.5 G 种子/s**（10M=6.8 ms） | 计算段；免 624 步串行 twist |
| `lfsrscan` native | 65536×512 拍 kernel **0.27 ms**（~322×） | u64 位打包 |
| ChaCha20 批量 | **10.9–18.3 M keys/s**（×64B） | v3.4 为 ~1.8 M/s |
| `xorbrute` 3 字节/16B | 扫描段 0.28–0.30 s（CPU 6.5–7.9 s） | GPU 路径 |
| `md5` 爆破 | 21–39 M 候选/s（CPU 2.4–2.55 M/s） | 有效候选 ≥2M 自动 GPU |
| RS63 | ~69 K codewords/s | 8192×63B syndrome |
| CRC32 | ~3.56 M/s | ×64B |
| `lsb` 大图 | 3000×3000 RGB ≈ 6.5 s | 8 GB 内存可跑 4000×4000 级 |
| `stegscan` 45 MB PNG | ≈ 1.5 s | 向量化 printable_runs |

## 端到端 before/after（v3.4 → v3.5，`gpu_profile.py` 安静窗口实测）

| 用例 | wall 前 | wall 后 | 加速 | 净计算（扣 ~2.3 s 启动） |
|---|---|---|---|---|
| mtscan twist1 10M | 20.23 s | **2.43 s** | **8.33×** | 17.88 → 0.15 s（119×，launch 42,493→9） |
| xorbrute 3 字节/16B | 9.26 s | **3.67 s** | **2.52×** | 6.96 → 1.41 s（4.9×） |
| chacha --brute 2M | 6.03 s | **2.94 s** | **2.06×** | 3.73 → 0.67 s（5.6×） |
| mtscan twist0 50M | 3.45 s | **2.44 s** | **1.41×** | 1.09 → 0.16 s（6.8×） |
| lfsrscan 65536×512 | 2.78 s | 2.42 s | 1.15× | 0.43 → 0.14 s（3.1×） |
| md5 爆破 3.257M | 3.61 s | 3.21 s | 1.12× | 内部 1.24 → 0.80 s |
| eccscan 1M（仍 CPU） | 6.24 s | 5.59 s | 1.12× | — |

> 端到端被 ~2.3 s 的 Python/torch 启动地板支配；净计算段才是本轮 GPU 利用率的真实提升。
> 复现：`python -X utf8 gpu_profile.py --subtract-startup --repeat 2 --compare <before.json>`。
