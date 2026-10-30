# ABench 沙箱容器化计划（Docker，免 sudo）

## 背景

现在的隔离依赖 root，具体靠这几样：
- 两个系统用户 `abench-agent` 和 `abench-judge`，用 `sudo -u` 切换；
- 按 uid 生效的 iptables 防火墙（`egress_proxy.Firewall`）；
- judge 子进程通过 `unshare --net` 加 `setpriv` 启动（`oj_eval.judge_isolation`）；
- 预检时逐个确认禁止访问的路径读不到（`sandbox_runner.isolation_preflight`）。

后果：只能在 Linux 上以 root 运行；没有 sudo 的用户只能跑不计分的 `host_soft` 冒烟测试；只有一个 agent 用户，所以一台机器同一时间只能跑一个运行；macOS 也只能跑冒烟测试。

**目标**：有 Docker 权限的普通用户，不用 sudo 就能跑完 agent 和 judge 的完整流程，正式运行可以计分。Docker 权限指 Linux 上在 docker 组里、或用 rootless Docker，以及 macOS 的 Docker Desktop。整体结构借鉴 Harbor（环境抽象、agent 注册、oracle、并发 jobs、doctor 自检），同时保留 ABench 自己的评测要求。

**已确定的决定**：
- 只支持 Docker：Linux 上的 docker 组或 rootless Docker，加上 macOS Docker Desktop。
- 容器版验证可用后，删除 sudo 路径；保留不计分的 `host_soft` 冒烟模式，给没有 Docker 的机器用。
- 这一轮做容器化，加上 Harbor 式的命令行、并发 jobs 和 GPU 分配。
- 镜像预构建后推到镜像仓库，按 sha256 digest 锁定。
- 设计要能扩展：更多 harness（如 Codex）、更多模型和服务商（包括 DeepSeek 这类兼容接口）、更多 case；并支持按难度和资源分级、分类运行（§5、§8–§10）。这一轮只接 Claude Code，Codex 等放到后续。
- 分级沿用网站上的分类（与 `index.yaml` 一致），各 case 的开发预算暂不调整。
- 默认运行和汇总全部 case（`all`）；管道校准的 case 不计分。

## 借鉴 Harbor 什么、不借鉴什么

| Harbor 的做法 | ABench 的处理 |
|---|---|
| 环境抽象 `BaseEnvironment`：start / stop / exec / upload / download，加 preflight | **借鉴**。写成 `tools/containers.py`。所有数据进出容器都走上传和下载（tar），不依赖 bind mount |
| 预置 agent（claude-code 等）；oracle agent 运行 `solution/solve.sh` | **借鉴**。agent 注册表包含 claude、oracle、mock、command。oracle 用 `example_submission` 跑全流程，取代 `calibrate.sh` |
| jobs：`--n-concurrent`、结果目录、按错误类型决定是否重试 | **借鉴**。`abench jobs` 提供并发、GPU 分配和断点续跑，只对基础设施故障重试 |
| 独立的 verifier 沙箱；产物用 tar 交接，并经过 data filter 过滤 | **借鉴**。judge 的每个 stage 都在全新容器里跑；产物交接用安全的 tar 解包 |
| agent 的认证与接口地址管理（Agent auth resolution、Base URL management） | **借鉴**。模型配置独立于 harness，出网白名单跟着配置走，不再写死在策略文件里，见 §8 |
| 数据集带版本（如 `terminal-bench@2.0`），按名字选择任务集 | **借鉴**。suite 带版本号，固定 case 列表、case spec 的哈希和 profile 版本，见 §10 |
| ATIF 轨迹格式 | **借鉴**。harness 适配器把各家原始日志转成统一摘要，可以导出为 ATIF，见 §5 |
| `network_mode`：public / no-network / allowlist，allowlist 靠 nftables sidecar 实现 | **不照搬**。ABench 要的是按主机名放行的研究出网，加上每个 case 自己的封锁表和哈希链日志。所以 agent 容器完全不联网，只通过一个 Unix socket 连到 ABench 自己的代理（以 sidecar 形式运行）。这样不需要 root，也不依赖 nftables |
| verifier 把测试打进镜像，或复用 agent 的环境 | **不照搬**。留出集不能进任何镜像，也不能进 agent 用过的容器。只在计时结束后，以去掉标签的输入形式送进 judge 容器；标签一直留在主机上的父进程里 |
| 奖励由 `test.sh` 写入 | **不照搬**。保留 `metrics.py`、bootstrap 区间、gates、baseline 和可计分条件 |
| 运行时在容器里安装 agent | **不照搬**。出网受限，而且结果要可复现，所以 agent CLI 预装进镜像，并锁定 digest |

## 总体结构

```
主机（普通用户；需要 python3、numpy、pyyaml 和 docker CLI）   ./abench run | calibrate | jobs ...
├─ agent 阶段（tools/sandbox_runner.py）
│   ├─ egress sidecar：钉住版本的 python 镜像，跑 egress_proxy.py serve
│   │     连 bridge 网络；在共享卷里监听 Unix socket；日志写在 sidecar 自己里面
│   └─ agent 容器：agent 镜像（profile 镜像 + harness CLI）
│         --network none，用户 uid 1000，--cap-drop ALL，输入卷只读
│         容器内 127.0.0.1:3128 转发到 socket，再到 sidecar；agent 用 docker exec 启动
│   结束后：tar 下载 submission → 安全解包 → 白名单过滤 → 删除容器、卷和网络
└─ judge 阶段（tools/oj_eval.py）：主机进程持有密钥，负责解密，标签和参考值只放在内存里
    每个 stage × seed 起一个全新容器：case 的 profile 镜像，--network none，非 root
    R/T/G/P 字节走 docker start -ai 的 stdout/stdin；T 之后才把留出集输入传进去
    P 之后 tar 下载输出，然后 docker rm -f -v；metrics.py 在主机上运行（只依赖 numpy）
```

`./abench jobs` 先用统一的 case 目录（catalog），按 suite（默认 `all`）或筛选条件选出 case；调度器按资源档位分配 GPU；然后每个运行起一个独立子进程，走上面的流程（§6、§10）。

**数据进出的原则**：不把主机目录 bind mount 进容器，统一用命名卷，内容通过 tar 上传或下载。这样做有三个原因：
- rootless Docker 的 uid 映射会让 bind mount 的文件权限出问题；
- Docker Desktop 的文件共享和 Linux 上的行为不一样；
- 三种环境可以共用一套代码。

公开数据、冻结模型和论文索引做成按内容哈希命名的缓存卷（`abench-data-<sha>`），只读挂载，多次运行复用。数据量小的视图、提交和 harness，每次运行重新上传。

## 关键设计

### 1. 镜像与环境锁
- `docker/profiles/<profile>/Dockerfile` 由 `spec/env/profiles.yaml` 生成（`tools/images.py`），每个 profile 一个，共 7 个：cpu-sci、gpu-torch、gpu-torch-hf、gpu-torch-diffusion、gpu-torch-sim、gpu-torch-cuda-dev、pro-h100。`extends` 对应 `FROM` 父镜像，profile 只有这一个定义来源。
  - 依赖版本全部固定（每个 profile 一份解析好的 requirements 锁文件）；
  - 镜像内置用户 `abench`（uid 1000）、`/abench/*` 目录、出网转发器，以及 `record_environment.py`。
- **版本规则**：profile 发布后不再修改。改依赖就发新版本（如 `gpu-torch@2026.11`），case spec 固定引用某个版本，旧结果在旧版本下照样可比。
- harness 镜像（`docker/agents/<name>/Dockerfile`）`FROM` 对应的 profile 镜像，只加 harness 自己的运行时和 CLI，不改 Python 环境，详见 §5。
- egress sidecar 用钉住 digest 的 python slim 镜像。`egress_proxy.py` 在启动时从主机的 `tools/` 上传进去，所以 `tools` 目录树的哈希也覆盖到实际运行的代理代码。
- 新增三类锁文件：
  - `spec/env/images.yaml`：仓库命名空间、镜像名、版本；
  - `spec/env/images.lock.yaml`：每个镜像在每个平台上的 digest；
  - `spec/env/lock/<profile>.json`：在镜像里运行现有的 `tools/record_environment.py` 生成。
- 平台：cpu-sci 和它对应的 agent 镜像同时构建 amd64 和 arm64，macOS 可以原生跑 CPU case；GPU 镜像只构建 amd64。参考平台是 linux/amd64，在其他平台上的运行结果有效，但不计分。
- case 用哪个 profile（含版本）写进 case 的 spec，生成到 `oj.yaml`；`profiles.yaml` 只保留 profile 的定义。现在的 `cases:` 映射只覆盖 17 个 case，迁移时补全 73 个，并加一个测试保证每个 case 都有 profile。

### 2. agent 容器（取代 sudo 用户、预检和防火墙）
- **创建参数**：`docker create --network none --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 --init --pids-limit --memory --cpus [--gpus device=N] --label abench.run=<id> <image@sha256>`。资源取自 case 的 `resources` 声明（§9）；没写的项用 `spec/container_policy.yaml` 里按 profile 的默认值，并且不能超过策略上限。pro-h100 是 4 张卡，并加大 `--shm-size`。
- **容器内路径**：

  | 路径 | 内容 | 读写 |
  |---|---|---|
  | `/abench/view` | 每次运行的视图卷 | 只读 |
  | `/abench/view/data` | 缓存的数据卷 | 只读 |
  | `/abench/models`、`/abench/papers` | 冻结模型、论文索引 | 只读 |
  | `/abench/workspace` | 工作区卷，含 `.home`、`.tmp`、`submission`、`checkpoint` | 可写 |
  | `/run/abench-egress` | sidecar 的 socket 卷 | 只读 |

- `build_agent_view` 不变，仍在主机上构建视图，再上传。`agent_env`、`agent_prompt`、`harness_command` 改成使用容器内路径，代码里把"主机路径"和"容器内路径"分开管理。
- **核验（取代预检）**，从"逐个确认读不到"改成"确认实际运行的就是允许的配置"：
  - 用 `docker inspect` 读出实际生效的网络、caps、用户、挂载列表和镜像 digest，与 `spec/container_policy.yaml` 比对；
  - 在容器里跑一次探测：直连外网要失败，DNS 解析要失败，视图不可写，只能看到预期的挂载。
  - 两部分结果都写进 `agent_run.json`。
- **启动 agent**：用 `docker exec -i`，prompt 的送入方式由 harness 适配器决定（§5）。容器内用 `timeout` 控制截止时间，主机到点后再用 `docker kill` 兜底。API key 只用 `-e NAME` 的形式传，值不出现在命令行上。
- **计量**：`GpuMeter` 只统计分配给这个运行的 GPU（并发时必须这样）；CPU 时间和内存峰值从容器的 cgroup 读取。
- **收集**：
  1. tar 下载 submission、检索日志，以及适配器声明的额外文件；
  2. 安全解包：拒绝符号链接、设备文件、绝对路径和 `..`，边解包边检查大小上限；
  3. 复用 `collect_submission` 的白名单和自报结果过滤。
- **清理**：删除容器、卷和网络。所有对象都带 `abench.run` 标签，进程崩溃后用 `abench gc` 清理残留；收到 SIGINT 或 SIGTERM 时，按标签删除本次运行创建的对象。

### 3. 研究出网：sidecar + Unix socket
- 主机仍用 `policy_for_case`（`tools/egress_proxy.py:81`）计算策略，以 JSON 传给 sidecar。
- 模型 API 的主机不再写死在 `harness_allow` 里，而是从本次运行的模型配置推出来，只放行这一个主机，见 §8。
- `EgressProxy` 增加 AF_UNIX 监听模式，加一个 `serve` 子命令。另加 `summarize_log()`：主机下载日志后，用它校验哈希链并生成摘要。
- agent 容器只有 loopback，没有别的网络接口。容器内的转发器把 `127.0.0.1:3128` 接到 socket 上，并设置 `HTTP(S)_PROXY=http://127.0.0.1:3128`，`NO_PROXY=localhost,127.0.0.1`。
- 以下行为保留不变：按主机名放行或拒绝、拒绝私有地址和回环地址（sidecar 连着 bridge 网络，这条很关键）、上游代理 `ABENCH_EGRESS_UPSTREAM`、超过 20 MiB 的下载单独列出。
- 网络模式：
  - `none`：不起 sidecar，给 oracle、mock 和离线 agent 用；
  - `open`：普通 bridge 网络，永远不计分。

### 4. judge 容器与 R/T/G/P 协议
- **主机父进程的职责不变**：持有密钥、解密，标签和参考值只放在内存里，在主机上跑 `metrics.py`。我检查过，各 case 的 metrics 及其辅助模块只依赖 numpy，所以主机不需要 torch。
- **每个 stage × seed 起一个全新容器**：
  - 镜像用 case 的 profile 镜像 digest，`--network none`，非 root，GPU 按 `oj.yaml` 的设置；
  - 挂载三个只读卷：submission 和 harness 合成的卷、缓存数据卷、模型卷。
- **协议通道**改在 `abench_stage.py` 的子进程侧实现：子进程先复制一份 stdout 的文件描述符，专门用来写 R/T/P 字节，再把 fd 1 和 fd 2 重定向到日志文件。父进程从 `docker start -ai` 的 stdout 读字节，向 stdin 写 G。`selfcheck`（以本地子进程方式运行）也走同一个通道，所以 agent 自检时执行的子进程代码和评测时完全一样。
- **一个 stage 的流程**：
  1. T 之后，父进程把 `held_out.npz` 传进容器的 `post/`，再发 G；
  2. P 之后容器退出，父进程 tar 下载输出、meta 和日志；
  3. `docker rm -f -v`，留出集输入随容器一起删除；
  4. 超时就 `docker kill`。
  - `pre/` 在容器启动前上传；`carry` 先从上一个容器下载，再上传到下一个容器。
- 子进程也记录它自己看到的 R 和 T 时刻，和父进程的计时一起写进记录，用来发现传输延迟带来的偏差。
- **评测记录的变化**：
  - `judge_isolation` 记为 `container`，与现在的 `child_user_netns` 同等对待；
  - 环境检查改成比对镜像 digest 和锁文件；
  - 不再需要 `ABENCH_JUDGE_PYTHONPATH`；
  - 继续检查 tools 目录树的哈希（`ABENCH_JUDGE_TOOLS_SHA256`），它对应主机上运行的评测代码。

### 5. harness 适配器与 agent 注册表（`tools/agents/`）
- 每个 harness 一个适配器（`tools/agents/<name>.py`），实现 `tools/agents/base.py` 定义的接口。加一个 harness 就是：一个适配器文件、一个 Dockerfile 模板，再通过契约测试。这一轮只有 Claude Code 一个真实 harness，接口由 claude、oracle、mock、command 四个实现一起检验；以后接 Codex 时，如果接口不够用再调整。
- **适配器声明**：名字、固定的版本、支持的接口协议（anthropic / openai-chat / openai-responses / gemini）、prompt 的送入方式（stdin / 参数 / 文件）、默认模型、需不需要网络。
- **适配器负责**：
  - 把模型配置（§8）翻译成自己的环境变量或配置文件，比如 Codex 的 `~/.codex/config.toml`；
  - 生成非交互的启动命令，以及启动前检查用的最小请求；
  - 把原始日志转成统一的轨迹摘要：工具调用、轮数、token、费用、实际返回的模型名。原始日志原样保留，摘要可以导出为 Harbor 的 ATIF 格式；
  - 列出结束后要额外收集的文件，比如 Codex 的 session 日志。
- **镜像**：每个 harness 只写一个 Dockerfile 模板（`docker/agents/<name>/Dockerfile`，用 `ARG PROFILE_IMAGE` 指定基础镜像），`abench images build` 自动对所有 profile 构建。构建完在镜像里运行 `record_environment.py --check`，与 profile 的环境锁比对，Python 环境有任何差异就让构建失败。这样 agent 开发用的环境始终就是 judge 的环境。
- **契约测试**：用一个假的模型服务（按协议返回固定回复）把每个适配器跑通，检查 prompt 是否送达、环境变量和配置文件是否正确、轨迹能否解析。CI 不需要真实的 API 密钥。
- **注册表**：

| agent | 镜像 | 网络 | 用途 |
|---|---|---|---|
| `claude` | agent-claude | egress | Claude Code，正式运行；复用 `harness_command` 和 `summarize_stream`；默认关闭自动更新和非必要流量；可以接官方 API 或 DeepSeek 等兼容接口 |
| `oracle` | profile 镜像 | none | 把 `example_submission` 放进提交目录；checkpoint 类 case 在容器里运行 `train.py`，参数来自 `sandbox/calibration/cases.yaml`。取代 `calibrate.sh`，主机不再需要 torch |
| `mock` | profile 镜像 | none | 运行现有的 `sandbox/mock_agent.sh`，不消耗 token 的管道检查 |
| `command` | 用户自己的镜像 | 任选 | 自定义 agent；镜像不在锁文件里就不计分 |

### 6. Harbor 式命令行（`./abench` → `tools/abench.py`）
- `doctor`：检查 Docker 版本，是 rootless 还是 Docker Desktop，GPU 运行时（`--gpus` 试跑一次），镜像 digest 是否与锁一致，能否起无网络容器，Unix socket 出网能否连通，密钥文件和主机 Python 依赖；并报告这台机器能跑哪些资源档位（§10）。
- `images pull|build|push|lock|verify`。
- `cases list [筛选条件]`：列出符合条件的 case，并显示难度 × 资源的分布（§10）。
- `case check <ID>`：新增 case 时的检查（§9）。
- `run --case X --agent claude --model <模型配置名> [--base-url URL] [--gpus 0] [--network egress|none] [--smoke]`。
- `calibrate X [--data synthetic|generated|<raw dir>]`：先在主机上冻结数据（现有的 `prepare/run.py`），再跑 oracle，然后评测。
- `judge --run runs/X/<id>`。
- `jobs run|resume|status|report`：
  - 选 case：不指定时跑默认的 `all`；也可以用 `--suite <名字>@<版本>`，或用筛选条件 `--difficulty`、`--resources`、`--area`、`--type`、`--subset`、`--cases`（§10）；
  - 可以给出多组 harness 和模型，跑"harness × 模型"的组合矩阵；
  - `--attempts` 设每个组合跑几次，默认 1 次；
  - `--dry-run` 先估算总算力，`--max-gpu-hours` 设上限（§10）；
  - 每个运行是一个独立子进程；
  - 按资源档位分配：参考卡（24GB）、整机 4 × H100、纯 CPU（§10）；
  - `-n` 控制并发上限；
  - 已经有 `evaluation.json` 的运行自动跳过，所以可以断点续跑；
  - 只对基础设施错误重试，agent 自己失败不重试；
  - `jobs/<job-id>/job.json` 指向各个 `runs/<case>/<run>/`。运行记录的目录结构不变，`summarize.py` 和网站不用改；
  - `report`：按难度 × 资源、领域、harness × 模型汇总（§10）。
- `gc`、`summarize`。
- 原有的 `sandbox_runner.py` 和 `oj_eval.py` 命令行保留：`--backend docker|host` 取代 `--agent-wrapper`（`host` 只能用于冒烟），新增 `--gpus` 参数。

### 7. 记录与可计分条件
- `agent_run.json` 新增这些字段：
  - `sandbox`：后端，是否 rootless 或 Docker Desktop，镜像 digest，inspect 核验结果，容器内探测结果，资源限制，sidecar 的镜像和日志哈希；
  - `model_endpoint`：见 §8；
  - `run_config`：这次运行的身份，包括 case、harness 名字和版本、模型配置名、harness 选项、第几次尝试、镜像 digest、所属 suite 及版本；
  - `catalog`：运行时这个 case 的难度、资源档位、领域、subset、类型和所属榜单的快照。以后做汇总，不依赖之后被修改过的 `index.yaml`。
- `evaluation.json`：`judge_isolation: container`，并记录 judge 镜像的 digest。
- 可计分条件在原有基础上做三处替换，并新增两条：

  | 原条件 | 新条件 |
  |---|---|
  | wrapper 在白名单里，且防火墙生效 | 容器核验通过，网络为 `none` 或经 egress sidecar |
  | judge 子进程以 `abench-judge` 身份运行 | judge 容器核验通过 |
  | judge 依赖经过认证 | 镜像 digest 与锁文件一致 |
  | （新增） | 运行平台必须是参考平台 linux/amd64 |
  | （新增） | GPU 类别符合 case 的资源声明：Budget 版用参考卡，Pro 用 4 × H100 |

### 8. 模型配置与接口地址
- **问题**：现在模型配置和 harness 绑在一起。Claude Code 用 `ANTHROPIC_BASE_URL` 等变量，可以改接 DeepSeek 的 `https://api.deepseek.com/anthropic`；Codex 用 OpenAI 协议，提供方写在 `~/.codex/config.toml` 里。模型 API 的主机还写死在 `spec/egress_policy.yaml` 的 `harness_allow` 里，每换一家服务商都要改策略文件。
- **模型配置文件 `spec/models.yaml`**，和 harness 无关。每条写：接口协议、接口地址、模型名、密钥变量名，以及可选的参数（推理强度、最大输出长度等）和辅助模型（如 Claude Code 用的小模型）。同一个 DeepSeek 模型可以按两种协议各写一条：

  ```yaml
  deepseek-chat-anthropic:      # 给 Claude Code 用
    api: anthropic
    base_url: https://api.deepseek.com/anthropic
    model: deepseek-chat
    key_env: DEEPSEEK_API_KEY
  deepseek-chat-openai:         # 以后给 Codex 等用
    api: openai-chat
    base_url: https://api.deepseek.com/v1
    model: deepseek-chat
    key_env: DEEPSEEK_API_KEY
  ```

- **解析**：运行 `abench run --agent claude --model deepseek-chat-anthropic` 时：
  1. 查模型配置，检查这个 harness 是否支持它的协议，不支持就直接报错；
  2. 由适配器翻译成 harness 自己的配置；harness 不支持的参数也报错，不悄悄忽略；
  3. 没指定模型时用 harness 的默认模型。
  - 临时试用可以用 `--base-url` 覆盖接口地址；也兼容直接用环境变量配置（如 `ANTHROPIC_BASE_URL`）。这两种情况记录里都会标明来源。
- **密钥**：变量名取自 `key_env`，只用 `-e NAME` 传进容器。值不进命令行，也不进记录；记录里只有变量名。
- **出网白名单**（所有 harness 通用）：
  - runner 从接口地址解析出主机，只把这一个主机加进本次运行的白名单，类别记为 `model_api`。`harness_allow` 只保留各 harness 官方的默认主机；本地为试跑加的 `api.deepseek.com` 之后就不再需要了。
  - 接口地址必须是 https。主机如果命中全局 `deny` 或这个 case 的封锁表，就拒绝启动，不会悄悄放行。
  - 代理默认拒绝私有地址，所以本机或内网的模型网关（如 LiteLLM、vLLM）要加显式参数 `--allow-private-model-endpoint` 才能放行；只放行这一个地址和端口，并记录在案。
- **启动前检查**：正式运行前，用同样的地址、密钥和代理路径，在容器里先发一次很短的请求（要求回复 pong），确认接口能连通、能认证。失败就直接停止，不消耗运行时间和预算。现在 `sandbox/run_local_claude_tc_ft_001.sh` 的 preflight 做的就是这件事，搬进容器版，每个适配器提供自己的检查命令。
- **记录**：`model_endpoint` 字段记下模型配置名、协议、接口地址、主机、配置来源（模型配置文件、命令行覆盖还是环境变量），以及轨迹里实际返回的模型名。这样同一个 case 在不同模型和服务商上的结果可以区分开。

### 9. case 扩展
- **profile 版本**：见 §1。case spec 写明用哪个 profile 的哪个版本。
- **资源声明**：case spec 新增 `resources`，生成到 `oj.yaml`。
  - 包括 `spec/compute_budget.md` 要求必须一起出现的字段：`budget_tier: budget|pro`、`hardware.reference`、`wall_clock_limit_h`、`leaderboard`；
  - 以及容器资源：GPU 数量和类别、CPU、内存、共享内存；不能超过 `container_policy.yaml` 的上限；
  - 缺字段时 runner 拒绝启动。这是 `compute_budget.md` 本来就要求的，但现在还没实现。
- **新增 case 的流程**：写 spec → `build_standard_cases.py` 生成文件 → 手写 `stages.py` 和 `metrics.py` → 写数据准备模块 → 声明 profile 和 `resources` → `abench case check <ID>` → `abench calibrate <ID>`。
- **`abench case check <ID>`**：检查 spec 和生成的文件、profile 及版本、数据准备模块、出网策略，以及资源声明是否符合档位上限；最后用 oracle 把校准跑一遍。
- **需要多个服务的 case**（推理服务加压测端、模拟器等）这一轮不做；launcher 接口留出"一个 stage 附带服务容器"的扩展点。

### 10. 分级与分类运行
- **现状**：每个 case 在 `index.yaml` 和 `verify.csv` 里有难度和资源档位，定义在 `spec/compute_budget.md`；网站的 case 浏览也按领域、组、难度、资源筛选。目前 73 个 case 的分布：

  | 难度 \ 资源 | low | medium | high | pro | 合计 |
  |---|---|---|---|---|---|
  | low | 11 | 4 | 0 | 0 | 15 |
  | medium | 1 | 19 | 10 | 0 | 30 |
  | high | 3 | 12 | 7 | 6 | 28 |
  | 合计 | 15 | 35 | 17 | 6 | 73 |

- **分级以网站上的分类为准。** 网站数据由 `tools/build_site_data.py` 从 `index.yaml` 生成；我核对过，线上网站、本地网站数据和 `index.yaml` 三处 73 个 case 的难度和资源档位完全一致。各 case 的开发预算（`oj.yaml` 的 `agent_budget`）暂不调整，也不检查它是否超出档位上限。
- **顺手处理的两处数据问题**：
  1. `compute_budget.md` 要求的资源字段，所有 `oj.yaml` 里都还没有（§9）；
  2. 命名冲突：`index.yaml` 里有 9 个 case 写了 `budget_tier: low/medium`，和 `budget` 字段重复；而 `compute_budget.md` 里 `budget_tier` 的含义是 `budget|pro`。删掉 `index.yaml` 里这 9 个重复的字段。
  - `tools/verify_registry.py --check` 加上检查：`index.yaml`、`verify.csv` 和网站数据里的分级一致，资源字段齐全。
- **统一的 case 目录**：新建 `tools/catalog.py`，把 `index.yaml`、`verify.csv`、`spec/areas.yaml` 和各 case 的 `oj.yaml` 合成每个 case 一条记录。命令行筛选、调度、报告和网站数据（`tools/build_site_data.py`）都用它，各处的筛选结果保持一致。
- **suite**（带版本，写在 `spec/suites.yaml`）：一个 suite 版本固定 case 列表、每个 case spec 的哈希和 profile 版本，结果只在同一个 suite 版本内比较。初始只有两个：

  | suite | 内容 | 数量 |
  |---|---|---|
  | `all`（默认） | 全部 case；其中管道校准的 TC-FT-001、DS-PIPE-002 照常运行，但不计分 | 73 |
  | `calibration` | 给 oracle 校准 judge 用：TC-FT-001、DS-PIPE-002，加上用真实数据校准的 FN-LTSF-005 | 3 |

  - 每个 case 的 `leaderboard` 字段：管道校准的 case 写 `none`，其余写 `all`。
  - 其他组合不单独定义 suite，用筛选条件选；以后需要固定下来，再加进 `spec/suites.yaml`。
- **选 case**：不指定时跑 `all`；也可以用 `--suite <名字>@<版本>`，或按条件筛选：`--difficulty`、`--resources`、`--area`、`--type`、`--subset`、`--cases`。`abench cases list` 用同样的条件列出 case 和难度 × 资源的分布。
- **按资源档位调度**：
  - 调度器从 `nvidia-smi` 读出本机的 GPU，分成三类：参考卡（24GB，RTX 4090 级）、整机 4 × H100、没有 GPU。
  - Budget 版的 case 要用参考卡，因为墙钟类指标规定在参考卡上测量；在别的 GPU 上跑要加显式参数，结果不计分。Pro 的 case 要独占一台 4 × H100 机器 24 小时。CPU case 按核数分配。
  - `abench doctor` 报告这台机器能跑哪些档位。
  - 本机没有对应硬件的 case（比如没有 4 × H100 时的 Pro case）不算失败，而是跳过，并在 job 的状态和报告里列出跳过的 case 和原因。
  - 默认按资源从低到高安排运行顺序，问题能早点暴露；可以用 `--order` 修改。
- **算力估算**：`--dry-run` 按"（开发预算 + 每次评测的上限）× 尝试次数 × harness 和模型的组合数"估算总算力。Budget 版的参考卡 GPU 小时和 Pro 的 H100 小时分开汇总，与 `compute_budget.md` 和网站的口径一致。`--max-gpu-hours` 设上限，预计会超出时就不再启动新的运行。
- **报告**：`abench jobs report` 默认汇总全部计分的 case（`all`），同时按难度 × 资源的网格、领域、harness × 模型分组：每组里各种 verdict 的数量，以及主指标相对基准分的结果和区间。
  - 管道校准的 case 不计入；两种算力分开汇总。
  - 入门（难度 low）和 Pro（资源 pro）作为网格里的分组查看。`compute_budget.md` 写的是 Pro 只进单独的榜，`index.yaml` 的 entry_rules 写的是入门单独成榜、不与主集合并，这两处和默认 all 不一致，在 P5 一起更新。
  - ABench 的最终得分（M、D、Q、X 四项）由单独的评分系统计算，不在这一轮的范围内；报告提供它需要的输入。

## 实施阶段

**P0 先验证关键假设。** 在三种环境上各验证一遍：Linux rootful + GPU、Linux rootless + GPU、macOS Docker Desktop。
1. Unix socket 放在命名卷里，以只读方式挂进 `--network none` 的容器后，能否正常连接。
   - 不行的话改用：internal 网络 + 双网卡的 sidecar。
2. 文件进出：用 tar 往卷里和运行中的容器里写文件（`docker cp`，或 `docker exec tar -x`）后，属主和权限是什么；能否从已停止的容器里 `docker cp` 取出文件。
3. 计时精度：同一个无操作 stage，分别在主机进程和容器里跑，比较测到的 `timed_seconds`。
   - 目标：两者相差在 10 ms 以内；
   - 超出的话，在记录里同时标出子进程自己记的时间。
4. GPU：rootful 和 rootless 下 `--gpus device=N` 是否可用；主机 `nvidia-smi` 能否按 GPU 编号计量。
5. Claude Code 能否以 uid 1000 在容器里运行，并经转发器和代理完成认证、输出 stream-json。官方 API 和兼容接口（如 DeepSeek）各试一次。
6. 容器内能否读到 cgroup v2 的 `cpu.stat` 和 `memory.peak`。

验证结果写进 `spec/sandbox_runner.md`；用到的验证脚本之后改写成需要 Docker 才运行的测试。

**P1 镜像和容器基础。**
- 新建 `docker/`（profile 的 Dockerfile 由 `profiles.yaml` 生成，处理 `extends`）、`tools/images.py`、`spec/env/images.yaml`、`spec/env/images.lock.yaml`、`spec/env/lock/*.json`。
- profile 版本规则；case spec 写明 profile 和版本，补全 73 个 case。
- 新建 `tools/containers.py`：
  - 封装 Docker CLI（检测 rootless 和 Docker Desktop；create / start -ai / exec / cp / inspect / kill / rm）；
  - 卷、网络、标签的辅助函数；
  - 按内容哈希命名的缓存卷；
  - 安全的 tar 上传和解包：逐条检查成员，有 `filter='data'` 时一并使用，带大小和数量上限。
- 新建 `spec/container_policy.yaml` 和 `tools/container_policy.py`：定义资源上限和按 profile 的默认值，负责生成容器参数，并对照 inspect 结果核验。

**P2 judge 进容器。**
- `tools/harness_kit/abench_stage.py`：
  - 子进程侧改用 stdio 协议通道，并记录子进程自己的时间戳；
  - `run_stage` 把启动子进程、送入 `post/`、取回输出这三步抽成 launcher 接口，本地 launcher 保持现有行为；
  - `KIT_VERSION` 升为 v1.2，用 `tools/build_standard_cases.py` 重新生成 73 个 case 里的副本。
- 新建 `tools/judge_container.py`（容器版 launcher）。
- `tools/oj_eval.py`：
  - 容器隔离与核验，`_iso_record` 返回 `container`；
  - `_environment_check` 和 `_dependency_attestation` 改为基于镜像 digest；
  - `_evaluate_harness` 把 launcher 传给 `run_pipeline`。
- IC-SPD-001 有自己的 `speedrun_runner`，把它迁到共用的 stage 协议上（kind 改为 `harness`）；在迁完之前，它仍按现在的方式记为 `host_soft`、不计分。
- **里程碑**：不用 sudo，用 DS-PIPE-002 的合成数据完成校准，`judge_isolation` 为 `container`；在同一份冻结数据上，指标与旧路径一致。

**P3 agent 进容器，harness 适配器和模型配置。**
- `tools/sandbox_runner.py`：
  - 把 `run()` 拆成几个步骤，抽出 `AgentEnvironment` 接口，实现两个版本：`DockerAgentEnvironment`，以及只用于冒烟的 `HostSoftEnvironment`；
  - 主机路径和容器内路径分开管理；
  - 实现核验、计量、tar 收集和清理。
- `tools/egress_proxy.py`：AF_UNIX 监听、`serve` 子命令、`summarize_log()`。
- harness 适配器（§5）：`tools/agents/base.py` 定义接口，实现 claude、oracle、mock、command；写契约测试和假的模型服务。
- 模型配置（§8）：`spec/models.yaml`、解析与兼容性检查、接口地址白名单、启动前检查、`model_endpoint` 记录；`harness_allow` 只保留官方默认主机。
- `run_config` 记录（§7）。
- **里程碑**：在 Linux（rootful 和 rootless）和 macOS 上，都能不用 sudo 跑通 `abench run --case TC-FT-001 --agent mock` 和 `--agent claude --smoke`（官方 API 和 DeepSeek 兼容接口各一次）。

**P4 命令行、jobs 与分级运行。**
- 新建 `tools/abench.py`、仓库根目录的 `./abench` 和 `tools/jobs.py`（GPU 池、并发、续跑、重试分类、job.json）。
- 分级运行（§10）：`tools/catalog.py`、`spec/suites.yaml`、筛选条件、按 GPU 类别调度、`--dry-run` 算力估算、`--max-gpu-hours`、`jobs report`、`cases list`、`case check`，以及运行记录里的 `catalog` 快照。
- 数据（§9、§10）：
  - `compute_budget.md` 要求的资源字段写进 case spec 的 `resources`，再生成到 `oj.yaml`；
  - 删除 `index.yaml` 里重复的 `budget_tier`；
  - `verify_registry.py --check` 加上分级一致性和字段齐全的检查。
- `abench calibrate` 取代 `calibrate.sh`。
- **里程碑**：
  - `abench cases list --difficulty low --resources low` 输出正确的分布；
  - `abench jobs run --agent mock --dry-run` 对默认的 `all` 给出算力估算，并列出本机硬件跑不了、会被跳过的 case；
  - `abench jobs run --cases TC-FT-001,DS-PIPE-002 --agent mock -n 2` 两个运行并发完成。

**P5 删除 sudo 路径，更新文档和 CI。**
- 删除的代码：
  - `tools/wrapper_policy.py` 和 `spec/wrapper_policy.yaml`；
  - `sandbox_runner.py` 里的 `isolation_preflight`、`write_launcher`、`write_launcher_env`、`cleanup_user`、`stage_tree`，以及 wrapper 相关分支；
  - `egress_proxy.Firewall`（TCP 监听模式保留，给 `host_soft` 冒烟用）；
  - `oj_eval.py` 里的 `judge_isolation`（setpriv / unshare）和 `ABENCH_JUDGE_CHILD_USER` 等环境变量；
  - `run_child` 和 `oj_predict_child.py`：73 个 case 都不再使用 `tabular_classification`；
  - IC-SPD-001 迁移后不再需要的 `evaluate_timed`；
  - `abench_stage.py` 里的 `isolation` 参数和 `_give`。
- 删除的测试和脚本：`tests/test_isolation.py` 里需要 root 的部分、`test_egress.py` 里的防火墙测试、`tests/test_wrapper_policy.py`、`calibrate.sh`、`run_local_claude_tc_ft_001.sh`。
- 更新的文档：
  - `README.md`：改成免 sudo 的快速上手（安装 Docker → `./abench doctor` → `images pull` → `calibrate` → `run`）；
  - `spec/sandbox_runner.md`；
  - `spec/oj_contract.md` 的 §1、§4、§6；
  - `spec/compute_budget.md` 和 `index.yaml` 的 entry_rules：关于 Pro、入门单独成榜的说法，改成与默认 all 一致（§10）；
  - `tools/test_split.py`（重新生成 `TESTING.md`）；
  - `verify.csv` 的 next_step 文字；
  - 网站（`site/`，发布在 https://huicir.github.io/abench）：说明页里"运行模式与隔离"一节改成容器版；case 浏览的筛选项改由 `tools/catalog.py` 生成。涉及 `site/index.html`、`site/assets/app.js`、`tools/build_site_data.py`。
- CI：
  - `.github/workflows/tests.yml`：单元测试，加上用 cpu-sci 镜像跑的 Docker 测试；
  - `.github/workflows/images.yml`：打 tag 时构建并推送镜像，需要你提供镜像仓库的凭据。

**后续（不在这一轮）**：更多 harness（Codex、Gemini CLI、OpenHands 等）；多服务 case；云沙箱；分阶段的 jobs（比如先跑难度 low 的 case，过线再跑其余的）。

## 复用的现有代码
- `sandbox_runner.py`：`build_agent_view`（:160）、`agent_env`（:213）、`harness_command`（:250）、`agent_prompt`（:289）、`GpuMeter`（:320）、`budget_check`（:374）、`summarize_stream`（:395）、`collect_submission`（:483）、`collect_retrieval_log`。
- `egress_proxy.py`：`policy_for_case`（:81）、`decide`（:103）、`_Log` 哈希链（:119）、`_public_address`（:134），以及 `EgressProxy._handle` 的连接处理逻辑。
- `oj_eval.py`：`evaluate`（:623）、`_evaluate_harness`（:485），以及解密、`_primary_ci`、`_aggregate`、`_gates`。
- `abench_stage.py`：`run_pipeline`（:374）的 carry 与 stage 编排、`_env`（:200）。
- 分级与目录：`index.yaml`、`verify.csv`、`spec/areas.yaml`、`spec/compute_budget.md` 的档位定义；`tools/verify_registry.py`（加一致性检查）；`tools/build_site_data.py` 和 `tools/test_split.py`（改用 catalog）。
- 其他：`hidden_store.py`、`validate_submission.py`、`leak_check.py`、`record_environment.py`、`tree_hash`、`sandbox/mock_agent.sh`、`sandbox/calibration/cases.yaml`、`sandbox/run_local_claude_tc_ft_001.sh` 里的 preflight 逻辑，以及 `test_egress.py` 里的 `FakeUpstream`。

## 验证
1. 单元测试，不需要 Docker：`python3 -m unittest discover -s tests`。新增的测试覆盖：
   - 容器参数的安全选项；
   - 安全 tar 解包（符号链接、设备文件、`..`、超出大小）；
   - 用 AF_UNIX 监听的代理（放行、拒绝、哈希链）；
   - 本地 stdio 协议：伪造握手、finish 超时、T 之前看不到留出集；
   - 模型配置的解析和兼容性检查；接口地址的白名单：必须是 https，命中封锁表时拒绝启动，私有地址要加显式参数才放行；
   - harness 适配器的契约测试（假的模型服务）；
   - case 目录的一致性检查（分级在各处一致、必需字段齐全）、suite 的选择（默认 all）、调度器的 GPU 类别匹配、跳过本机跑不了的 case、算力估算；
   - jobs 的 GPU 分配与续跑。
2. 需要 Docker 的测试，在 Linux rootful、Linux rootless、macOS 上各跑一次：`tests/test_container_judge.py`、`tests/test_container_agent.py`。agent 容器里要验证：
   - 直连外网失败，`getaddrinfo` 失败；
   - `/abench/view` 不可写，只能看到预期的挂载；
   - 经代理访问 `huggingface.co` 返回 403，并且有日志；
   - 超时会杀掉 agent；
   - 收集时拒绝符号链接。
3. 端到端：
   - `./abench doctor`；
   - `./abench calibrate DS-PIPE-002 --data synthetic`：verdict 为 evaluated，`judge_isolation` 为 container，在同一份冻结数据上指标与旧路径一致；
   - `./abench calibrate FN-LTSF-005 --data <raw>`：MSE 比值约 0.999 [0.993, 1.007]；
   - 在 GPU 机器上：`./abench calibrate A-LIN-002 --gpus 0`；
   - `./abench run --case TC-FT-001 --agent mock`，以及 `--agent claude --smoke`。后者用官方 API 和 DeepSeek 兼容接口各跑一次，检查记录里的 `model_endpoint`，并确认出网日志里放行的模型主机只有配置的那一个；
   - `./abench cases list --difficulty low --resources low`，以及 `./abench jobs run --agent mock --dry-run`（默认 all）；
   - `./abench jobs run ... -n 2`。
4. 计时：容器与主机进程的 `timed_seconds` 差值在 P0 定下的范围内。
5. 仓库检查：`tools/validate_task_statement.py --strict`、`tools/build_standard_cases.py --check`、`tools/verify_registry.py --check`（含新增的分级一致性和字段检查）。

## 风险与已知限制
- **docker 组等同于主机 root。** 在多人共用的服务器上，推荐每个用户用 rootless Docker；`doctor` 会提示这一点。在 rootful Docker 下，同组的其他用户能读到正在运行的 judge 容器内容。
- **容器和主机共用内核。** 容器逃逸和 GPU 驱动漏洞不在威胁模型之内，这一点和现在的 sudo 方案一样。
- **rootless + GPU 需要管理员做一次配置**（NVIDIA 容器工具包设置 `no-cgroups`）。这是一次性的，不需要每次运行都有 sudo。`doctor` 会检测；如果 cgroup 资源限制不可用，记录写成"未强制"。
- **CUDA 镜像很大**，单个约 10 GB 以上，第一次拉取比较慢。
- **以下限制和现在一样，没有变化**：出网只按主机名过滤，不检查 TLS 内容；agent CLI 在服务商那边执行的网页搜索不经过代理。
- **macOS 没有 GPU。** CPU case 用 arm64 镜像运行，结果有效但不计分，因为不是参考平台。
- **这一轮只有一个真实 harness。** 适配器接口目前只经 Claude Code 检验，接 Codex 等时可能还要调整。
- **假的模型服务只能验证管道**（prompt 送达、配置、日志解析），验证不了模型本身的行为。每个 harness 仍要用真实接口跑一次冒烟。
- **harness 的 CLI 更新很快，参数常变。** 适配器固定 CLI 版本，升级当作新版本处理；记录里有 harness 的版本。
