# plugins —— 第三方内核插件

把 `*.py` 放进本目录即可被统一入口识别，**不需要修改仓库里的任何文件**。

## 协议

1. 模块级提供元数据字典（字面量写法）：

```python
KERNEL = dict(name='morse', help='摩斯码解码（一句话说明）')
```

2. 脚本能被 `python plugins/<file>.py <args...>` 直接调用：结果写 stdout，出错写 stderr，
   成功返回 0、参数错误返回 2（与内置内核约定一致）。

3. 约束：
   - `name` 必须是**非空字符串**，不得与内置内核重名（重名时内置优先，启动会打 `[warn]`）；
   - `name` 也不得占用统一入口的保留命令名：`selftest` / `selfcheck` / `test` / `all` / `help`；
   - 文件名不要以 `_` 开头（会被跳过）；
   - 建议自包含，不要 import 仓库根的大模块（否则插件无法单独分发）；
   - **元数据是静态读取的**：读取 `KERNEL` 时不执行插件顶层代码（`ctf_registry` 用 AST 解析
     模块级的字面量赋值），所以 `ctf.py help` / `--help` / `doc_lint` 都不会触发插件副作用。
     为保持兼容，若 `KERNEL` 不是可静态解析的字面量（例如用变量拼接），会退回
     `exec_module` 读取一次（会执行顶层代码，stderr 有 `[warn]` 提示）——**推荐始终用
     字面量 `KERNEL = dict(...)`**。
   - 统一入口的全局参数（`--json/--quiet/--backend/--chunk/--checkpoint/...`）**不适用于插件**；
     写在插件名前的全局参数会被提示忽略，插件自己的参数请写在插件名之后。

4. 同名/坏插件（语法错误、`KERNEL` 不是 dict、`name` 非法）只会被跳过并告警，不会拖垮主程序。

## 调用方式

```bash
python skill/scripts/ctf.py morse --code "... --- ..."
python plugins/morse_decode.py --code "... --- ..."      # 也可以直接跑
```

## 校验

```bash
python ctf_registry.py                          # 列出内置内核 + 已发现插件 + 注册表自检
python skill/scripts/ctf.py selfcheck           # 统一入口自检（含插件发现/容错用例）
```

插件不计入 `doc_lint` 的功能子命令统计（数量由 `ctf_registry.py` 派生），也不会让文档一致性
检查失败；默认的静态读取路径同样不会在 `doc_lint` / `--help` 期间执行插件代码。
