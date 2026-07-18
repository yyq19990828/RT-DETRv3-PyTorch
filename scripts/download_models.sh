#!/bin/bash

# RT-DETRv3 模型下载脚本 (Google Drive)
# 使用 gdown 工具从 Google Drive 下载预训练模型
#
# 使用方法:
#   bash scripts/download_models.sh [model_name]
#
# 可选的模型名称:
#   - r18: RT-DETRv3-R18
#   - r34: RT-DETRv3-R34
#   - r50: RT-DETRv3-R50
#   - all: 下载所有模型 (默认)

# 检查是否安装了 gdown
if ! command -v gdown >/dev/null 2>&1; then
    echo "错误: 未找到 gdown 工具"
    echo "请先安装开发依赖: uv sync --extra dev"
    exit 1
fi

# 创建模型保存目录
MODEL_DIR="pretrained_models/paddle"
mkdir -p "$MODEL_DIR"

# 定义模型下载信息
# 格式: 模型名称|Google Drive文件ID|保存文件名
declare -A MODELS
MODELS=(
    ["r18"]="1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk|rtdetrv3_r18vd_6x_coco.pdparams"
    ["r34"]="12-wqAF8i67eqbocaWPK33d4tFkN2wGi2|rtdetrv3_r34vd_6x_coco.pdparams"
    ["r50"]="1wfJE-QgdgqKE0IkiTuoD5HEbZwwZg3sQ|rtdetrv3_r50vd_6x_coco.pdparams"
)

# 下载函数
download_model() {
    local model_key="$1"
    local model_info="${MODELS[$model_key]}"

    if [ -z "$model_info" ]; then
        echo "错误: 未知的模型名称 '$model_key'"
        return 1
    fi

    IFS='|' read -r file_id filename <<< "$model_info"
    local output_path="$MODEL_DIR/$filename"

    echo "=========================================="
    echo "下载模型: RT-DETRv3-${model_key^^}"
    echo "保存路径: $output_path"
    echo "=========================================="

    # 检查文件是否已存在
    if [ -f "$output_path" ]; then
        echo "文件已存在,跳过下载: $output_path"
        return 0
    fi
    
    # 使用 gdown 下载
    if gdown "https://drive.google.com/uc?id=$file_id" -O "$output_path"; then
        echo "✓ 下载成功: $filename"
    else
        echo "✗ 下载失败: $filename"
        return 1
    fi
}

# 主程序
MODEL_NAME="${1:-all}"

case "$MODEL_NAME" in
    r18|r34|r50)
        download_model "$MODEL_NAME"
        ;;
    all)
        echo "开始下载所有模型..."
        for model_key in "${!MODELS[@]}"; do
            download_model "$model_key"
            echo ""
        done
        echo "所有模型下载完成!"
        ;;
    *)
        echo "错误: 未知的模型名称 '$MODEL_NAME'"
        echo ""
        echo "使用方法: bash scripts/download_models.sh [model_name]"
        echo ""
        echo "可选的模型名称:"
        echo "  r18  - RT-DETRv3-R18 (20M params, 48.1 AP)"
        echo "  r34  - RT-DETRv3-R34 (31M params, 49.9 AP)"
        echo "  r50  - RT-DETRv3-R50 (42M params, 53.4 AP)"
        echo "  all  - 下载所有模型 (默认)"
        exit 1
        ;;
esac

echo ""
echo "下载的模型保存在: $MODEL_DIR/"
echo "可以使用以下命令查看:"
echo "  ls -lh $MODEL_DIR/"
