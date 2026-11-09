# plugins —— 第三方内核插件

把 `*.py` 放进本目录即可被统一入口识别，**不需要修改仓库里的任何文件**。

## 协议

1. 模块级提供元数据字典：

```python
KERNEL = dict(name='morse', help='摩斯码解码（一句话说明）')
```

2. 脚本能被 `python plugins/<file>.py <args...>` 直接调用：结果写 stdout，出错写 stderr，
   成功返回 0、参数错误返回 2（与内置内核约定一致）。

3. 约束：
   - `name` 不得与内置内核重名（重名时内置优先，启动会打 `[warn]`）；
   - 文件名不要以 `_` 开头（会被跳过）；
   - 建议自包含，不要 import 仓库根的大模块（否则插件无法单独分发）；
   - `KERNEL` 读取阶段会**执行一次模块顶层代码**，所以顶层不要放有副作用的重活，
     入口一律放在 `if __name__ == '__main__':` 里。

## 调用方式

```bash
python skill/scripts/ctf.py morse --code "... --- ..."
python plugins/morse_decode.py --code "... --- ..."      # 也可以直接跑
```

## 校验

```bash
python ctf_registry.py     # 列出内置内核 + 已发现插件
```

插件不计入 `doc_lint` 的 30 个内置功能子命令统计，也不会让文档一致性检查失败。
