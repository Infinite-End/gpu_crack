# dsh-skill-gpu-crack — DSH 插件：把 gpu_crack 装载为内置技能

这是一个 **DeepSeek Harness（DSH）plugin bundle**。它不复制技能文件，而是把本仓库的
[`../skill/`](../skill/) 目录注册成 DSH 的 `bundled` 技能 provider：

- 目录里的 `SKILL.md` 作为技能正文（自动去掉 YAML frontmatter，`description` 进发现目录）；
- `resourceBase` 指向该目录，技能里的相对资源（`scripts/ctf.py`、`references/kernels.md`）可解析；
- 每次加载都重新读文件，仓库更新后无需重装插件；
- 装载后模型会在技能目录里看到 `ctf-toolkit`，加载后按 `skill/` 里的规范使用工具箱。

## 结构

```
dsh-plugin/
├─ package.json          # npm 包元数据 + dsh.bundle.patch 挂载声明
├─ cordis.patch.yml      # bundle patch 层：插入 skill-gpu-crack 行（含 assetRoot 配置）
├─ lib/index.js          # cordis 插件：ctx.skills.registerProvider(...)
└─ test/smoke.mjs        # 离线冒烟测试（不启动 DSH，验证 provider 契约）
```

## 安装（本机 / 本地开发）

DSH 支持用 pnpm 的 `link:` 协议装载本地插件目录：

```powershell
dsh plugin --profile desktop add link:E:\gpu_crack\dsh-plugin
```

也可以让 DSH 的 plugin-manager 安装同一路径（安装 spec 用 `link:E:\gpu_crack\dsh-plugin`）。
`cordis.patch.yml` 已把 `assetRoot` 配成 `E:\gpu_crack\skill`；若仓库移动位置，改这一处即可
（或设置环境变量 `GPU_CRACK_SKILL_DIR` 作默认值）。

安装后无需重启即可在**新会话**的技能目录中看到 `ctf-toolkit`；若宿主未热加载 bundle，
重启 DeepSeek Harness 一次。

## 验证

```powershell
node dsh-plugin\test\smoke.mjs        # 期望 DSH-SKILL-GPU-CRACK: ALL PASS
```

装载后让模型调用 `skill` 工具（name=`ctf-toolkit`），或直接在新会话里提出一个 CTF 任务，
模型应能按技能规范运行 `skill\scripts\ctf.py`。

## 卸载

```powershell
dsh plugin --profile desktop remove dsh-skill-gpu-crack
```

或经 plugin-manager 移除该 bundle，然后重启 DSH。

## 说明

- 这是**技能提供者插件**，不新增模型工具；技能里的脚本通过 DSH 的 bash/pwsh 工具执行。
- 插件本身零运行期依赖（不 import `@deepseek-ai/*`），只需宿主提供 `skills` 服务。
- 仓库里的 `gpu_crack.py` / `ctf_*.py` 仍是工具箱本体，插件只负责“挂载技能”。
