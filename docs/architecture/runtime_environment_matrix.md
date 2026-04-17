# 运行环境矩阵

- document_id: arch_runtime_environment_matrix
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-18

## 1. 目标

FRCNet 0.1 采用单一训练代码路径，运行时自动解析为以下后端之一：

- `mps`
- `rocm`
- `cuda`
- `cpu`

默认解析优先级为：

```text
mps -> rocm -> cuda -> cpu
```

## 2. 支持范围

### 2.1 macOS

- 目标平台：Apple Silicon
- 加速后端：`torch.device("mps")`
- 回退策略：若 `MPS` 不可用，则回退到 `CPU`
- 0.1 不把 Intel Mac 作为必测目标

### 2.2 Linux with ROCm

- 目标平台：支持 ROCm 的 AMD GPU 环境
- 设备路径：`torch.device("cuda")`
- 识别方式：`torch.cuda.is_available()` 且 `torch.version.hip is not None`
- 元数据中必须标记为 `resolved_backend=rocm`

### 2.3 Linux with CUDA

- 目标平台：支持 CUDA 的 NVIDIA GPU 环境
- 设备路径：`torch.device("cuda")`
- 识别方式：`torch.cuda.is_available()` 且 `torch.version.hip is None`
- 元数据中必须标记为 `resolved_backend=cuda`

### 2.4 CPU Fallback

- 当未检测到可用 GPU / MPS 时使用
- 作为默认测试与最低保真兼容路径

## 3. 安装原则

- 项目代码只保留一套
- `pyproject.toml` 不绑定平台专属 wheel 索引
- 各平台的 `torch` 安装方式由使用者按环境选择

## 4. 推荐安装方式

### 4.1 Apple Silicon macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -e .
```

### 4.2 ROCm

根据本机 ROCm 版本，先按 PyTorch 官方说明安装匹配的 `torch` / `torchvision`，再安装项目：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 4.3 CUDA

根据本机 CUDA 版本，先按 PyTorch 官方说明安装匹配的 `torch` / `torchvision`，再安装项目：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## 5. 0.1 约束

- 默认 `dtype=float32`
- 默认 `amp_enabled=false`
- 不做平台专属混合精度调优
- `DataLoader` 默认 `num_workers=0`
- `pin_memory=auto` 时：
  - `cuda/rocm -> true`
  - `mps/cpu -> false`

