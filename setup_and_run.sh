#!/bin/bash
# ============================================================================
# DrMAS 一键环境搭建 & 训练启动脚本
# 适用于: CUDA 12.4 + Python 3.10 + PyTorch 2.6 + vllm 0.8.x 基础镜像
# 默认使用 vllm 作为推理引擎（也可切换为 sglang，见下方注释）
# 用法:
#   bash setup_and_run.sh              # 完整流程：搭建环境 → 准备数据 → 启动训练
#   bash setup_and_run.sh setup        # 仅搭建环境（不训练）
#   bash setup_and_run.sh train        # 仅启动训练（跳过环境搭建）
#   bash setup_and_run.sh eval         # 仅启动评估
#   bash setup_and_run.sh retriever    # 仅启动检索服务器
# ============================================================================

set -e

# -------------------- 配置变量 --------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HOME/data"
RETRIEVER_ENV_NAME="retriever"
RETRIEVER_DATA_DIR="$DATA_DIR/searchR1"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -------------------- Stage 0: 确保 vllm 版本正确 --------------------
stage_check_vllm() {
    log_info "Stage 0: 检查 vllm 版本..."
    # 确保 torch==2.6.0（setup.py vllm extras 依赖）
    pip install torch==2.6.0 --quiet

    # 如果镜像有 vllm 0.8.x 但和 torch 2.6 不兼容，重新安装
    if ! python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
        log_warn "vllm 未安装或不可用，安装 vllm==0.8.3..."
        pip install vllm==0.8.3
    else
        vllm_ver=$(python -c "import vllm; print(vllm.__version__)")
        log_info "检测到 vllm 版本: $vllm_ver"
    fi
    log_info "Stage 0: vllm 检查完成"
}

# -------------------- Stage 1: 安装 DrMAS 核心依赖 --------------------
stage_install_core() {
    log_info "Stage 1: 安装 flash-attn + liger-kernel + vllm..."

    # flash-attn: setup.py GPU_REQUIRES 中的核心加速库（编译可能需要几分钟）
    # 如果镜像已安装，跳过；否则从预编译 wheel 安装
    if python -c "import flash_attn" 2>/dev/null; then
        log_info "flash-attn 已安装，跳过"
    else
        log_info "安装 flash-attn（编译可能需要几分钟，请耐心等待）..."
        pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
    fi

    # liger-kernel: setup.py GPU_REQUIRES 中的高效 kernel 融合库
    pip install liger-kernel

    # 安装 requirements.txt 中的依赖
    cd "$REPO_ROOT"
    pip install -r requirements.txt

    # 以 editable 模式安装 DrMAS/veRL
    # install_requires 包含: ray>=2.41.0,<2.50.0, transformers>=4.52.1,<=4.53.2,
    # packaging>=20.0, qwen-vl-utils[decord], tensordict<=0.6.2 等
    pip install -e .

    log_info "Stage 1: 核心依赖安装完成"

    # 如果之后想用 sglang 作为推理后端，可以：
    #   pip install "sglang[all]==0.4.6.post5" --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
    #   pip install torch-memory-saver
    # 并将 run_search.sh 中的 rollout.name=vllm 改为 rollout.name=sglang
}

# -------------------- Stage 2: 安装 Search 环境 --------------------
stage_install_search_env() {
    log_info "Stage 2: 安装 Search 环境..."
    cd "$REPO_ROOT/agent_system/environments/env_package/search/third_party"
    pip install -e .
    pip install gym==0.26.2
    cd "$REPO_ROOT"
    log_info "Stage 2: Search 环境安装完成"
}

# -------------------- Stage 3: 准备 Search 数据集 --------------------
stage_prepare_data() {
    log_info "Stage 3: 准备 Search 数据集..."
    cd "$REPO_ROOT"
    python examples/data_preprocess/drmas_search.py
    log_info "Stage 3: 数据集准备完成"
}

# -------------------- Stage 4: 搭建 Retriever 环境 --------------------
stage_setup_retriever_env() {
    log_info "Stage 4: 搭建 Retriever conda 环境..."
    if conda env list | grep -q "^${RETRIEVER_ENV_NAME} "; then
        log_warn "conda 环境 '${RETRIEVER_ENV_NAME}' 已存在，跳过创建"
    else
        conda create -n "$RETRIEVER_ENV_NAME" python=3.10 -y
    fi

    # 在 retriever 环境中安装依赖（通过 conda run 避免手动 activate）
    conda run -n "$RETRIEVER_ENV_NAME" conda install numpy==1.26.4 -y
    conda run -n "$RETRIEVER_ENV_NAME" pip install \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
    conda run -n "$RETRIEVER_ENV_NAME" pip install \
        transformers datasets pyserini huggingface_hub
    conda run -n "$RETRIEVER_ENV_NAME" conda install \
        faiss-gpu==1.8.0 -c pytorch -c nvidia -y
    conda run -n "$RETRIEVER_ENV_NAME" pip install uvicorn fastapi

    log_info "Stage 4: Retriever 环境搭建完成"
}

# -------------------- Stage 5: 下载检索索引 --------------------
stage_download_index() {
    log_info "Stage 5: 下载检索索引..."
    local local_dir="$RETRIEVER_DATA_DIR"

    if [ -f "$local_dir/e5_Flat.index" ] && [ -f "$local_dir/wiki-18.jsonl" ]; then
        log_warn "索引文件已存在，跳过下载"
        return 0
    fi

    mkdir -p "$local_dir"
    conda run -n "$RETRIEVER_ENV_NAME" python \
        examples/search/searchr1_download.py --local_dir "$local_dir"

    log_info "合并索引分片..."
    cat "$local_dir"/part_* > "$local_dir/e5_Flat.index"

    log_info "解压 wiki 语料..."
    gzip -d "$local_dir/wiki-18.jsonl.gz"

    log_info "Stage 5: 索引下载完成"
}

# -------------------- Stage 6: 启动检索服务器 --------------------
stage_start_retriever() {
    log_info "Stage 6: 启动检索服务器（后台运行）..."
    local logfile="$REPO_ROOT/retrieval_server.log"

    if pgrep -f "retrieval_server" > /dev/null 2>&1; then
        log_warn "检索服务器已在运行，跳过启动"
        return 0
    fi

    conda run -n "$RETRIEVER_ENV_NAME" --no-capture-output \
        bash examples/search/retriever/retrieval_launch.sh > "$logfile" 2>&1 &

    sleep 3
    if pgrep -f "retrieval_server" > /dev/null 2>&1; then
        log_info "检索服务器已启动，日志: $logfile"
    else
        log_error "检索服务器启动失败，请查看日志: $logfile"
    fi
}

# -------------------- Stage 7: 启动训练/评估 --------------------
stage_run_training() {
    local mode="${1:-train}"
    log_info "Stage 7: 启动 Search 训练/评估（mode=${mode}）..."
    cd "$REPO_ROOT"
    bash examples/drmas_trainer/run_search.sh "$mode"
}

# -------------------- 主流程 --------------------
main() {
    local action="${1:-all}"

    echo ""
    echo "============================================"
    echo "  DrMAS 一键环境搭建 & 训练脚本"
    echo "============================================"
    echo "  项目根目录: $REPO_ROOT"
    echo "  数据目录:   $DATA_DIR"
    echo "  操作模式:   $action"
    echo "============================================"
    echo ""

    cd "$REPO_ROOT"

    case "$action" in
        setup)
            # 仅环境搭建
            stage_check_vllm
            stage_install_core
            stage_install_search_env
            stage_prepare_data
            stage_setup_retriever_env
            stage_download_index
            log_info "========== 环境搭建全部完成！=========="
            log_info "下次运行: bash setup_and_run.sh retriever   # 启动检索服务器"
            log_info "          bash setup_and_run.sh train       # 启动训练"
            ;;

        retriever)
            # 仅启动检索服务器
            stage_start_retriever
            ;;

        train)
            # 仅训练
            stage_run_training "train"
            ;;

        eval|evaluation)
            # 仅评估
            stage_run_training "eval"
            ;;

        all|*)
            # 完整流程：搭建环境 → 准备数据 → 启动检索服务器 → 训练
            stage_check_vllm
            stage_install_core
            stage_install_search_env
            stage_prepare_data
            stage_setup_retriever_env
            stage_download_index
            stage_start_retriever
            log_info "========== 环境就绪，开始训练 =========="
            stage_run_training "train"
            ;;
    esac
}

main "$@"
