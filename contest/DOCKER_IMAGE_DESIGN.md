# robot_lab Docker 镜像设计说明

本文以当前工作区的 `docker/Dockerfile`、`docker/docker-compose.yaml`、`docker/.env.base`、`docker/container.sh` 和根目录 `.dockerignore` 为准，说明镜像如何构建、容器如何运行，以及这些配置背后的取舍。项目是独立于 Isaac Lab 仓库的机器人强化学习扩展；Docker 的目标是复用 Isaac Sim/Isaac Lab 运行环境，同时把 `robot_lab` 接入同一个 Python 环境。

## 1. 整体结构

```text
本地构建的 isaac-lab-base 镜像
    └── docker/Dockerfile：复制本仓库，安装 Isaac Lab 与 robot_lab
          └── robot-lab 镜像
                └── docker-compose.yaml：挂载工作区、配置网络和交互终端
                      └── robot-lab 容器
```

这里的 `robot-lab` 不是从零安装 Isaac Sim 的镜像。`docker/.env.base` 默认指定 `ISAACLAB_BASE_IMAGE=isaac-lab-base`，该基础镜像需要先按 Isaac Lab 的部署说明在本地构建。README 示例对应 Isaac Lab 2.3.0、Isaac Sim 5.1.0、Python 3.11，但 Dockerfile 本身没有锁定这些版本；最终版本由实际使用的基础镜像和构建时解析到的依赖决定。

| 文件 | 职责 |
| --- | --- |
| `docker/.env.base` | 提供基础镜像名和仓库在容器中的路径。 |
| `docker/Dockerfile` | 在基础镜像上复制项目，安装 Isaac Lab 核心包和本项目 Python 包。 |
| `docker/docker-compose.yaml` | 定义构建上下文、镜像与容器名、源码挂载、网络和环境变量。 |
| `docker/container.sh` | 封装构建、启停、进入容器，并按宿主机情况生成 GPU/X11 覆盖配置。 |
| `.dockerignore` | 限制进入构建上下文的文件，排除版本库、文档、日志、运行结果等。 |

`.env.base` 在这里有两个用途：脚本的 `--env-file` 让 Compose 用它展开 `${ISAACLAB_BASE_IMAGE}` 和 `${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}`；服务的 `env_file: .env.base` 又把其中变量注入容器。Dockerfile 中的 `ARG` 是构建参数，而由 `ENV` 设置的扩展路径及代理参数会保存在镜像配置中。构建参数、Compose 插值变量与容器运行环境属于不同层次，修改其中一层不一定会同步改变其余层。

## 2. 镜像构建流程

Compose 的 `build.context` 是仓库根目录 `../`，Dockerfile 位于 `docker/Dockerfile`。因此 `COPY ./ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}` 复制的是**仓库根目录**，不是 `docker` 子目录。默认目标路径是 `/workspace/isaaclab_extension_template`；`.dockerignore` 中列出的目录和文件不会进入构建上下文，也不会被这条 `COPY` 复制。镜像包含构建时的项目快照。

Dockerfile 只有一个基于 `ISAACLAB_BASE_IMAGE_ARG` 的构建阶段（命名为 `base`），没有另设最终阶段。它以 `root` 身份执行后续操作，最后将工作目录设为 `/workspace`。构建逻辑可分为三步：

1. 复制仓库到扩展目录。
2. 通过 `${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh` 进入 Isaac Sim/Isaac Lab 使用的 Python 环境，先安装 `setuptools<82`，再以 `--no-build-isolation` 安装 `flatdict==4.0.1`，然后执行 `isaaclab.sh --install`，并用 `import flatdict; import isaaclab` 验证。这里单独处理 `flatdict`，是因为该版本构建时依赖 `pkg_resources`，而 setuptools 82 已移除它；预装依赖也避免基础镜像未预装可导入的 Isaac Lab 包。
3. 在复制后的 `source/robot_lab` 目录执行 `isaaclab.sh -p -m pip install -e .`。`-e` 将扩展以可编辑方式安装到**同一个** Python 环境，依赖由 `source/robot_lab/setup.py` 声明，包括 `psutil`、`colorama`、`xacrodoc`、`numpy`、`pandas`、`pinocchio` 和 `cusrl[all]`。

两段安装命令均通过 `bash -i -c` 启动、加载 `${HOME}/.bashrc`，以继承基础镜像中可能由交互式 shell 提供的 Isaac Sim 环境设置。此设计依赖基础镜像存在可用的 `isaaclab.sh`，默认位置为 `/workspace/isaaclab`；仅有一个同名镜像标签并不能保证满足这个约定。

镜像构建时已安装代码；容器运行时 Compose 又把宿主机仓库 bind mount 到**相同路径**。挂载会遮盖镜像内该路径的构建时副本，可编辑安装因而能够使用宿主机的当前源码和资源文件。通常修改 Python 源码无需重建镜像；修改 Dockerfile、基础镜像或安装依赖后，应重新构建。直接运行镜像而不挂载仓库时，使用的是镜像中的构建时快照。

## 3. 构建期与运行期代理

代理配置分两套传递路径，目的是让下载依赖的构建命令与实际运行的容器都能联网：

| 阶段 | 配置来源与用途 |
| --- | --- |
| 构建期 | Compose 把 `BUILD_*_PROXY_ARG` 和 `CLASH_*_ARG` 传给 Dockerfile。两段 `RUN` 优先采用显式 `BUILD_*` 代理；没有显式代理时，按 Clash 地址拼出代理 URL；两者都不可用时取消 shell 内代理变量。 |
| 运行期 | Compose 从启动它的 shell 读取大小写两组 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY`，注入容器环境。`container.sh` 会预先准备这些变量。 |

`container.sh` 的优先级是：`DISABLE_CLASH_PROXY=1` 清空代理配置；否则若当前 shell 已设置 HTTP/HTTPS/ALL_PROXY（大小写任一形式），沿用该配置；否则默认使用 Clash 的 `http://127.0.0.1:7890` 作为运行期代理，并使用 `host.docker.internal:7890` 作为构建期代理。端口、协议、主机和不走代理的地址可通过 `CLASH_PROXY_*` / `CLASH_NO_PROXY` 变量调整。脚本在显式代理模式下会把其值传给 `BUILD_*_PROXY_ARG`。

Compose 为构建和运行都配置了 `host.docker.internal:host-gateway`，构建还设置 `build.network: host`，容器设置 `network_mode: host`。运行期默认代理使用 `127.0.0.1`，依赖容器与宿主机共享网络命名空间；构建期使用 `host.docker.internal`，代理服务也必须能从该地址访问。若宿主机没有可访问的 Clash 代理，应使用自己的代理环境变量，或设置 `DISABLE_CLASH_PROXY=1`。Dockerfile 的代理构建参数又通过 `ENV` 写入镜像，因此不要把含凭据的代理 URL 作为构建参数传入；构建时环境也不应被视为临时秘密存储。

## 4. 容器运行设计

Compose 定义单个 `robot-lab` 服务，镜像名和容器名都为 `robot-lab`。`OMNI_KIT_ALLOW_ROOT=1` 配合 Dockerfile 的 root 用户，使 Omniverse Kit 可在容器内以 root 运行。入口为 `bash`，并启用 `stdin_open` 与 `tty`，适合保持开发容器运行并通过脚本进入交互 shell。仓库 bind mount 使代码、训练输出等直接落在宿主机相应目录；容器删除不会删除这些宿主机文件。`docker compose down` 删除容器，但保留镜像。

GPU 和图形界面不是写死在主 Compose 文件中的，而由 `container.sh` 在 `docker/.tmp/` 下生成覆盖文件并与主配置一起传给 Compose：

- GPU：脚本检查 NVIDIA 设备节点、`nvidia-smi` 命令及其执行结果；检测到后通常增加 `devices: ["nvidia.com/gpu=all"]`。若宿主机安装 Docker Snap，则改用 `runtime: nvidia` 和 `gpus` 的 `nvidia.runtime-hook` 驱动，避免 Docker Snap 与 apt 版 NVIDIA Container Toolkit 混用时的 CDI 挂载失败。`FORCE_GPU=1` 强制添加，`DISABLE_GPU=1` 禁用，禁用优先。宿主机仍需配置 NVIDIA 容器运行环境；仅能构建镜像不代表 GPU 仿真可运行。
- X11：存在 `DISPLAY` 且 `/tmp/.X11-unix` 目录可用时，添加 `DISPLAY`、`TERM`、`QT_X11_NO_MITSHM`、X11 socket 和只读时区挂载；若 `XAUTHORITY` 指向现有文件，还会只读挂载认证文件。启动前尝试通过 `xhost` 授予容器 root 本地访问权限。无 X11 时按无 GUI 模式启动，训练命令可使用 `--headless`。

主 Compose 文件没有单独声明 GPU/X11 配置。因此使用 `container.sh` 启动与直接执行 README 中的基础 `docker compose up`，在 GPU/X11 可用性方面可能不同；若直接用 Compose，需要自行叠加相应配置。

## 5. 使用流程

先构建 `isaac-lab-base`，并确认宿主机具备 Docker 和 Compose。正常情况下从仓库根目录运行：

```bash
./docker/container.sh config   # 查看合并后的 Compose 配置，包括自动检测的覆盖项
./docker/container.sh build    # 只构建 robot-lab 镜像
./docker/container.sh start    # 构建（必要时）并在后台启动容器
./docker/container.sh enter    # 进入正在运行的容器
./docker/container.sh stop     # 停止并删除容器
```

`start` 实际调用的是 `compose up -d robot-lab`，**没有 `--build`**。虽然脚本帮助文本写着“Build and start”，但镜像已存在时不能据此认为它会重新构建；Dockerfile 或依赖变化后应明确先执行 `build`。`restart` 执行 `stop` 后再执行 `start`，也不等于重新构建镜像。启动检查仅轮询容器是否处于 Running 状态，不能证明 Isaac Sim、GPU 或训练任务已经通过验证。

默认工作目录是 `/workspace`，不是项目目录，也不是 `/workspace/isaaclab`；Dockerfile 末尾注释所说的“Isaac Lab directory”与实际 `WORKDIR` 不完全一致。因此运行相对路径的脚本前需要切换到项目目录。进入容器后应通过 `isaaclab.sh -p` 运行项目脚本，以使用镜像构建时安装包的同一个解释器，例如：

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 2048 --headless
```

如果不使用辅助脚本，可在 `docker/` 目录执行 README 给出的 `docker compose --env-file .env.base --file docker-compose.yaml build robot-lab` 等命令；这些命令只使用主 Compose 文件，不会自动生成 GPU/X11 覆盖配置。

无代理环境可对各次操作显式设置变量：

```bash
DISABLE_CLASH_PROXY=1 ./docker/container.sh build
DISABLE_CLASH_PROXY=1 ./docker/container.sh start
DISABLE_CLASH_PROXY=1 ./docker/container.sh enter
```

对 `enter` 使用该开关，只会为新进入的 shell 删除大小写代理变量，并通过 `bash --noprofile --norc` 避免启动文件重新设置代理；它不会更改已运行容器或其他进程的环境。普通 `enter` 也不会重新执行代理配置逻辑，而是沿用容器环境。需要使用其他代理时，应在 `build`、`start` 等命令执行前导出对应环境变量。

## 6. 设计取舍与维护注意事项

- **分层复用**：Isaac Sim/Isaac Lab 的大型运行时由基础镜像提供，项目镜像主要负责扩展安装；更换基础镜像时需核对 `isaaclab.sh` 路径、Python 环境和版本兼容性。
- **开发效率**：构建时安装扩展保证镜像具备包元数据和依赖，运行时挂载源码便于迭代；新增或变更 Python 依赖仍要重建镜像，且宿主机挂载内容优先于镜像快照。
- **构建上下文**：`.dockerignore` 排除了常见的大型输出，但 `COPY ./` 仍以整个未忽略的仓库为输入。新增的大型数据、缓存或 `docker/.tmp` 文件若未被忽略，会增加构建上下文和镜像内容；维护时应检查忽略规则。
- **缓存效率**：完整仓库的 `COPY` 位于两段依赖安装之前，任何被复制文件的变化都可能使后续安装层的缓存失效。以后如果构建耗时明显，可以考虑将稳定的 Isaac Lab 安装步骤提前，或按依赖元数据与源码拆分复制步骤；这是优化方向，当前文件尚未这样组织。
- **可复现性**：默认基础镜像使用本地标签 `isaac-lab-base`，多数项目依赖未锁版本；相同 Dockerfile 在不同时间或机器上可能得到不同环境。需要精确复现训练时，应同时记录基础镜像标识和解析后的 Python 依赖版本。
- **宿主机耦合**：host 网络、bind mount、GPU 设备和 X11 转发服务于本地开发/训练，适合受信任的 Linux 工作站；迁移到其他网络、容器平台或远程显示环境时，需要调整 Compose 覆盖配置。
- **数据边界**：本项目 Compose 只为仓库声明了持久化挂载，没有单独配置训练缓存卷；写在仓库挂载目录之外的容器文件可能随容器删除而丢失，基础镜像自行声明的卷需另外核实。root 在绑定目录中创建的文件也可能在宿主机显示为 root 所有。
- **验证范围**：Dockerfile 的导入检查位于安装 `robot_lab` 之前，仅检查 `flatdict` 与 `isaaclab`。本说明依据配置和脚本静态分析编写，未实际构建镜像或运行训练；完整验证还需要在目标宿主机测试 GPU、仿真启动和项目任务。
