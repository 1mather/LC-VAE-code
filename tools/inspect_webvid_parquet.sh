#!/bin/bash

# 示例脚本：检查 WebVid parquet 文件

PARQUET_PATH="/scratch/shareddata/dldata/webvid-10m/dataset/val/00000.parquet"

echo "检查 WebVid Parquet 文件..."
echo "========================================"

# 基本检查（快速），指定 url 列作为视频路径列
python tools/inspect_parquet.py \
    --parquet_path ${PARQUET_PATH} \
    --video_column "url" \
    --skip_details

# 如果需要详细信息，去掉 --skip_details 标志：
# python tools/inspect_parquet.py \
#     --parquet_path ${PARQUET_PATH} \
#     --video_column "url"

# 如果需要保存 JSON 摘要：
# python tools/inspect_parquet.py \
#     --parquet_path ${PARQUET_PATH} \
#     --video_column "url" \
#     --output_json parquet_summary.json

# 如果跳过视频加载测试（加快速度）：
# python tools/inspect_parquet.py \
#     --parquet_path ${PARQUET_PATH} \
#     --video_column "url" \
#     --skip_video_test \
#     --skip_details

