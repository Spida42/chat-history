# LIBERO 项目进度快照

> **生成时间**: 2026-03-16
> **目的**: 提供给新 AI 窗口，快速恢复上下文，继续指导工作

---

## 👤 用户背景

- **身份**: 研究生，正在学习 LIBERO（终身机器人学习 benchmark）
- **经验水平**: 编程基础一般，需要手把手指导，命令要可直接复制粘贴
- **目标**: 跑通 LIBERO benchmark，理解代码结构，后续做终身学习相关实验
- **沟通风格**: 喜欢一次性给全部步骤，减少来回对话（"和你聊天太贵了"）

---

## 🖥️ 服务器信息

| 项目 | 详情 |
|---|---|
| 平台 | AutoDL 云服务器 |
| GPU | **RTX 4090** |
| SSH 命令 | `ssh -p 15467 root@connect.cqa1.seetacloud.com` |
| 端口 | 15467（之前是 20530，会变） |
| 连接方式 | VSCode Remote-SSH，Host 别名 `autodl` |
| 系统盘 | `/root/` |
| 数据盘 | `/root/autodl-tmp/`（大容量存储） |

---

## 📁 项目目录结构

```
/root/LIBERO/                          # 项目根目录（git 仓库）
├── libero/
│   ├── libero/                        # 核心包（import libero.libero）
│   │   ├── __init__.py                # 路径管家，管理 ~/.libero/config.yaml
│   │   ├── benchmark/                 # benchmark 定义
│   │   ├── envs/                      # 环境
│   │   ├── utils/                     # 工具函数
│   │   ├── bddl_files/                # 任务描述文件
│   │   ├── init_files/                # 初始状态文件
│   │   ├── assets/                    # 资源文件
│   │   └── datasets -> /root/autodl-tmp/libero_datasets  # 软链接！
│   ├── lifelong/                      # 终身学习算法
│   │   └── main.py                    # 训练入口
│   └── configs/                       # hydra 配置文件
├── notebooks/                         # Jupyter notebooks
│   ├── quick_walkthrough.ipynb        # ⭐ 入门 notebook（正在跑）
│   ├── quick_guide_algo.ipynb         # 算法指南
│   ├── procedural_creation_walkthrough.ipynb
���   ├── custom_object_example.ipynb
│   └── custom_assets/
├── benchmark_scripts/
│   └── download_libero_datasets.py    # 数据集下载脚本
├── setup.py                           # pip install -e . 的配置
└── scripts/
```

---

## 🐍 环境信息

| 项目 | 详情 |
|---|---|
| conda 环境名 | `libero_env` |
| Python 版本 | 3.9.25 |
| LIBERO 安装方式 | `pip install -e .`（editable mode） |
| LIBERO 版本 | 0.1.0 |
| ipykernel | ✅ 已安装 |
| Jupyter 内核 | 已注册为 `libero_env`，显示名 `libero_env (Python 3.9.25)` |

### 激活环境命令
```bash
conda activate libero_env
```

---

## 📊 数据集状态

| 项目 | 详情 |
|---|---|
| 数据集 | **libero_spatial** |
| 存储位置 | `/root/autodl-tmp/libero_datasets/libero_spatial/` |
| 软链接 | `/root/LIBERO/libero/datasets` → `/root/autodl-tmp/libero_datasets` |
| 文件数量 | **10 个 hdf5 文件** |
| 总大小 | **5.9 GB** |
| 每个文件 | 50 条 demo，包含 `actions, dones, obs, rewards, robot_states, states` |
| 完整性 | ✅ 已验证，全部可正常读取 |
| 下载方式 | 使用镜像源 `export HF_ENDPOINT=https://hf-mirror.com`（Hugging Face 直连被墙） |

### 10 个任务文件
```
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5
pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5
```

---

## ⚠️ 已知问题与解决方案

### 1. Hugging Face 无法直连
- **现象**: `httpcore.ConnectError: [Errno 101] Network is unreachable`
- **解决**: 下载前设置镜像源
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

### 2. Notebook 里 `import libero` 报 ModuleNotFoundError
- **根因**: LIBERO 目录结构是两层 `libero/libero/`，Notebook 工作目录不对时，Python 会把顶层 `libero/` 当成命名空间包（namespace package），找不到真正的模块
- **解决**: 在 Notebook **第一个 cell** 运行：
  ```python
  import os
  os.chdir("/root/LIBERO")
  print("工作目录:", os.getcwd())

  import importlib
  import libero
  importlib.reload(libero)
  import libero.libero
  print("libero.libero 路径:", libero.libero.__file__)
  print("成功！")
  ```
- **预期输出**: `libero.libero 路径: /root/LIBERO/libero/libero/__init__.py`

### 3. `datasets` 软链接目标不存在
- **现象**: `FileExistsError` 或 `No such file or directory`
- **解决**: `mkdir -p /root/autodl-tmp/libero_datasets`
- **当前状态**: ✅ 已修复

### 4. VSCode Jupyter 内核警告
- **现象**: "不再支持与所选 kernel 关联的 Python 版本"
- **解决**: **忽略即可**，不影响运行

---

## ✅ 已完成的步骤

1. ✅ AutoDL 服务器开机，VSCode Remote-SSH 连接成功
2. ✅ conda 环境 `libero_env` 确认存在
3. ✅ LIBERO 包重新安装（`pip install -e .`）
4. ✅ Jupyter 内核注册（`python -m ipykernel install --user --name libero_env`）
5. ✅ 数据集 `libero_spatial` 下载完成并验证完整性
6. ✅ Notebook `quick_walkthrough.ipynb` 打开，内核选好，import 问题已解决

---

## 🔄 当前正在做的步骤

### 任务 A：运行 `quick_walkthrough.ipynb`（进行中）
- 已打开 notebook，选好内核 `libero_env (Python 3.9.25)`
- 已解决 `import libero` 报错问题
- **下一步**: 继续逐个 cell 运行，理解数据结构和场景可视化
- **注意**: AutoDL 无显示器，渲染/可视化相关的 cell 可能报错，跳过即可

### 任务 B：首次训练（待做）
完成 Notebook 后，在终端运行：
```bash
cd /root/LIBERO
conda activate libero_env
export HF_ENDPOINT=https://hf-mirror.com

python libero/lifelong/main.py \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_rnn_policy \
    lifelong=base \
    seed=42 \
    train.n_epochs=5
```
- `bc_rnn_policy` = 行为克隆 + RNN（最基础算法）
- `lifelong=base` = 无遗忘防护的 baseline
- `n_epochs=5` = 快速验证能跑通
- 预计 RTX 4090 上跑 10-30 分钟

---

## 📋 整体学习计划

| 周次 | 内容 | 状态 |
|---|---|---|
| 第 1 周 | 环境搭建 + 跑通 Notebook + 首次训练 | 🔄 进行中 |
| 第 2 周 | 理解代码结构 + 跑完整实验 | ⏳ 待开始 |
| 第 3 周 | 尝试不同终身学习算法 | ⏳ 待开始 |
| 第 4 周+ | 自定义实验 / 改进算法 | ⏳ 待开始 |

---

## 🔑 关键命令速查

```bash
# 连接服务器
ssh -p 15467 root@connect.cqa1.seetacloud.com

# 激活环境
conda activate libero_env

# Hugging Face 镜像（下载东西前先设置）
export HF_ENDPOINT=https://hf-mirror.com

# 验证 LIBERO
python -c "import libero; print('LIBERO OK')"

# 查看数据集
find /root/autodl-tmp/libero_datasets -name "*.hdf5" | wc -l

# 查看 LIBERO 配置
cat ~/.libero/config.yaml
```
