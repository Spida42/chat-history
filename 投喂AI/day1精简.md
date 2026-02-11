# LIBERO 毕设项目 — 对话记忆摘要（精简版）

**用户:** @Spida42
**身份:** 工科入门初学者，毕设需要用到 LIBERO 平台
**基准平台:** [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)


> ⚠️ **这是精简后的记忆摘要**，原始对话约 8000+ 字，压缩至约 1000 字。
> 精简掉的内容包括：仓库完整目录结构讲解、每个文件的功能详解表格、各种比喻性教学内容（厨房/汽车类比等）、Notebook 各 section 的详细说明、VSCode/SSH 配置的具体步骤教程、源码逐行解读。
> 这些内容都是教学性质的，Copilot 可以随时通过读仓库源码重新获取，不需要占用上下文。

---

## 🎯 当前阶段

- **正在学习 LIBERO 仓库的使用方式**，为后续毕设做铺垫
- 毕设具体任务书尚未提供，具体研究内容待定
- 目前已知方向：与机械臂增量学习相关，LIBERO 作为基准平台

---

## ✅ 已完成的事项

1. **环境搭建完成**
   - AutoDL 云服务器
   - conda 环境 `libero_env`（Python 3.9.25）
   - `pip install -e .` 安装 LIBERO 成功，`import libero` 验证通过

2. **VSCode 远程开发环境配置完成**
   - Remote-SSH 连接 AutoDL（Host: `connect.cqa1.seetacloud.com`, Port: `20530`）
   - Jupyter 插件 + `ipykernel` 已安装，内核指向 `libero_env`

3. **仓库结构已初步了解**（必要时可再次讲解目录结构）

---

## 🔜 当前进度 — 第 1 周中期

按照已制定的 4 周学习计划：

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 1 周 | 搭环境 + 下载数据 + 跑通 Notebook + 首次训练 | 🔄 进行中 |
| 第 2 周 | 精读训练流程（main.py、algos/base.py、configs） | ⏳ 未开始 |
| 第 3 周 | 精读已有算法实现（重点 EWC、ER，对比着看） | ⏳ 未开始 |
| 第 4 周 | 仿照已有算法，开始写自己的算法 | ⏳ 未开始 |

### 第 1 周剩余待办

- [ ] 下载 `libero_spatial` 数据集（只下这一个，约 3GB）
  ```bash
  python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface
  ```
- [ ] 运行 `notebooks/quick_walkthrough.ipynb`，重点关注 **2.4（可视化场景）** 和 **2.6（数据结构）**
- [ ] 首次训练运行：
  ```bash
  python libero/lifelong/main.py benchmark_name=LIBERO_SPATIAL policy=bc_rnn_policy lifelong=base seed=42 train.n_epochs=5
  ```

---

## 📌 用户作为新手已通过初步学习确认的关键认知

1. **`libero/libero/`（仿真环境）→ 只调用不改**
2. **`libero/lifelong/`（算法+模型）→ 大部分直接用，`algos/` 是后续开发自己算法的地方**
3. **`scripts/` → 独立工具脚本，不在代码中 import，入门阶段不需要用**
4. **`configs/` → 通过修改 YAML 或命令行参数控制实验**
5. **看训练效果 = 跑 `main.py` 看终端打印的成功率（succ），不是用 `create_dataset.py`**
6. **开发自己的算法 = 新建文件继承 `Sequential` 基类，重写 `observe`、`start_task`、`end_task`**

---

## ⚠️ 沟通偏好

- 用户是编程/ML 新手，需要用类比和大白话解释概念
- 使用中文交流
- AutoDL 服务器端口可能会变（按需更新 SSH config）
