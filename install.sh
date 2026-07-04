#!/usr/bin/env bash
# 在仓库根目录执行: bash install.sh
#
# ------------------------------------------------------------------------------
# 用途
# ------------------------------------------------------------------------------
# 本脚本面向本仓库主线实验：vLLM 来自 `3rdparty/vllm` 内置源码目录。
# 该目录里已经包含本项目需要的 Qwen3.5 / HiF4 适配；用户 clone 后不需要
# 再手工修改 vLLM。
#
# 当前主线版本固定为 vLLM v0.19.1，并且走**真正源码编译**：
#   - 不使用 `VLLM_USE_PRECOMPILED=1`；
#   - 不拉 `wheels.vllm.ai` rolling wheel；
#   - 用当前 hif4 环境里的 torch 2.10.0 做 `--no-build-isolation` 编译，保证
#     `vllm/_C*.so` 和运行时 torch 的 c10 ABI 一致。
#
# 当前只保留 `install.sh` 作为安装入口。
# ------------------------------------------------------------------------------
#
# 要求：
#   - 当前已处于 conda env "hif4" 下（Python 3.11）
#   - 已经 clone 本仓库，且 `3rdparty/vllm` / `3rdparty/lighteval` 源码目录存在。
#   - 机器有可用的 NVIDIA driver；conda CUDA toolkit 只解决编译期 nvcc/headers，
#     不能替代内核驱动。
#
# 组件版本（与本仓库主线一致）：
#   vllm        = v0.19.1   （3rdparty/vllm 内置源码目录, editable, source build）
#   torch       = 2.10.0
#   torchvision = 0.25.0
#   torchaudio  = 2.10.0
#   transformers= 5.6.2
#   CUDA toolkit= 12.8      （优先使用 hif4 conda env 中的 cuda-toolkit）
#   lighteval   = v0.13.0   （3rdparty/lighteval 内置源码目录, editable, 含本仓库补丁）
#   HiFloat4    = 本仓库 HiFloat4/hif4_gpu CUDA 扩展
#   runtime deps= accelerate / datasets / safetensors / tqdm / inspect-ai 等
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAX_JOBS="${MAX_JOBS:-32}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8}"

# -------- 0. 基本检查 --------
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "hif4" ]]; then
    echo "[install.sh] 错误：当前 conda env 不是 hif4（实际: ${CONDA_DEFAULT_ENV:-none}）。"
    echo "[install.sh] 为避免影响其他环境，请先执行： conda activate hif4"
    exit 1
fi

echo "[install.sh] Python: $(which python) $(python --version)"
echo "[install.sh] Pip:    $(which pip)"
echo "[install.sh] MAX_JOBS=${MAX_JOBS}"

# -------- 1. 检查内置源码目录 --------
for src in 3rdparty/vllm 3rdparty/lighteval; do
    if [[ ! -d "$REPO_ROOT/$src" ]]; then
        echo "[install.sh] 错误：缺少源码目录 $src。请确认仓库 clone 完整。"
        exit 1
    fi
    echo "[install.sh] 找到源码目录: $src"
done

# -------- 2. CUDA toolkit（编译期） --------
# vLLM 0.19.1 / torch 2.10.0 默认使用 CUDA 12.8 这一代 wheel。系统 nvcc 偏旧时，
# 允许在 conda env 内安装 CUDA toolkit，并显式让构建系统使用它。
if ! command -v nvcc >/dev/null 2>&1 || ! nvcc --version | grep -q "release 12.8"; then
    echo "[install.sh] ==> 安装/确认 conda CUDA toolkit ${CUDA_TOOLKIT_VERSION}"
    conda install -n hif4 -c nvidia "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" -y
fi

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_ROOT="${CUDA_HOME}"
export CUDA_PATH="${CUDA_HOME}"
export CUDAToolkit_ROOT="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_TOOLKIT_ROOT_DIR="${CUDAToolkit_ROOT}"
export PATH="${CONDA_PREFIX}/bin:${CONDA_PREFIX}/nvvm/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDAToolkit_ROOT}/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export NVCC_PREPEND_FLAGS="-I${CUDAToolkit_ROOT}/include ${NVCC_PREPEND_FLAGS:-}"
echo "[install.sh] CUDA_HOME=${CUDA_HOME}"
echo "[install.sh] CUDAToolkit_ROOT=${CUDAToolkit_ROOT}"
echo "[install.sh] nvcc: $(command -v nvcc)"
nvcc --version

# -------- 3. torch / 构建依赖 --------
# 先装 torch 栈，再用 --no-build-isolation 编译 vLLM，确保编译期和运行期 ABI 一致。
echo "[install.sh] ==> 安装 torch 2.10.0 栈"
pip install "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0"

echo "[install.sh] ==> 安装 Transformers 5.6.2"
pip install "transformers==5.6.2"

echo "[install.sh] ==> 安装 vLLM build 依赖"
pip install -r "$REPO_ROOT/3rdparty/vllm/requirements/build.txt"

# -------- 4. vLLM（editable, source build） --------
echo "[install.sh] ==> 源码编译安装 vLLM v0.19.1 (editable, no build isolation)"
#########################################
# export http_proxy="${http_proxy:-http://127.0.0.1:7897}"
# export https_proxy="${https_proxy:-http://127.0.0.1:7897}"
# export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
# export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"

# git config --global http.proxy "${http_proxy}"
# git config --global https.proxy "${https_proxy}"
# git config --global http.version HTTP/1.1
# git config --global core.compression 0
# git config --global http.lowSpeedLimit 0
# git config --global http.lowSpeedTime 999999
########################################
pushd "$REPO_ROOT/3rdparty/vllm" >/dev/null
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.19.1" \
SETUPTOOLS_SCM_PRETEND_VERSION="0.19.1" \
MAX_JOBS="${MAX_JOBS}" \
CMAKE_ARGS="-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT} -DCUDA_TOOLKIT_ROOT_DIR=${CUDAToolkit_ROOT} -DCUDA_INCLUDE_DIRS=${CUDAToolkit_ROOT}/include -DCUDA_CUDART_LIBRARY=${CUDAToolkit_ROOT}/lib/libcudart.so" \
    pip install --editable . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.org/simple
popd >/dev/null

# -------- 5. lighteval（editable, 不带 vllm extras） --------
# lighteval[vllm] 会让 pip 重新解析并可能覆盖本地 editable vLLM；这里必须只装
# lighteval 本体。版本约束放宽依赖 3rdparty/lighteval 里的本地补丁。
echo "[install.sh] ==> 安装 lighteval (editable, 不带 [vllm] extras)"
pushd "$REPO_ROOT/3rdparty/lighteval" >/dev/null
pip install --editable .
popd >/dev/null

# -------- 6. HiFloat4 CUDA 扩展 --------
echo "[install.sh] ==> 编译 HiFloat4 CUDA 扩展"
export CUDA_HOME=/home/liuzhilei/anaconda3/envs/hif4/targets/x86_64-linux

export CPATH="$CUDA_HOME/include:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
pushd "$REPO_ROOT/HiFloat4/hif4_gpu" >/dev/null
bash build.sh
popd >/dev/null

# -------- 7. 本项目运行依赖 --------
echo "[install.sh] ==> 安装本项目运行依赖"
# accelerate / datasets / safetensors / tqdm : HiFloat4 量化与 PPL 测试链路要用
# inspect-ai                                : lighteval 0.13.0 基础依赖
# more_itertools                            : lighteval 跑 vllm 后端时 batch iter 要用
# langdetect                                : IFEval 的语言识别要用
pip install \
    accelerate \
    "datasets>=4.0.0" \
    safetensors \
    tqdm \
    inspect-ai \
    more_itertools \
    langdetect

# -------- 8. 安装后导入检查 --------
echo "[install.sh] ==> 安装后导入检查"
python - <<'PY'
import torch
import vllm
import vllm._C
import lighteval
import accelerate
import datasets
import safetensors
import tqdm
import inspect_ai
import more_itertools
import langdetect
import HiFloat4.main as hif4_main
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

print(
    "ok",
    "torch=" + torch.__version__,
    "torch_cuda=" + str(torch.version.cuda),
    "vllm=" + vllm.__version__,
    "vllm_path=" + vllm.__file__,
    "hif4_main=" + hif4_main.__file__,
    "qwen3_5=" + Qwen3_5ForCausalLM.__name__ + "/" + Qwen3_5TextConfig.__name__,
)
PY

echo "[install.sh] 完成。"
echo "[install.sh] 如果 MAX_JOBS=32 编译压力太大，可重试：MAX_JOBS=8 bash install.sh"
