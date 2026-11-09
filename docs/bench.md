# 吞吐基准

由下面这条命令生成，**不要手改**：

```
python gpu_crack.py bench --markdown
```

设备：NVIDIA GeForce RTX 5070 Laptop GPU  VRAM 7.9 GiB；批量 16384（AES）/ 65536（mt19937）/ 4194304（popcount）。

## 本机实测

| 内核 | 吞吐 | 批量 | 耗时 |
|---|---|---|---|
| `popcount` | 1089.7 M 个/s | 4194304 个 32bit | 4 ms |
| `mt19937` | 16368.6 K keys/s | 65536 个种子 | 4 ms |
| `aes128` | 1479.6 K blocks/s | 16384 块 | 11 ms |
| `aes192` | 1245.3 K blocks/s | 16384 块 | 13 ms |
| `aes256` | 1065.6 K blocks/s | 16384 块 | 15 ms |
