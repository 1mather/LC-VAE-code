#!/bin/bash
set -euo pipefail

SRC_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/results/channel4_pretrain_32_100k/validation_samples/step_0095001/generated"
DST_DIR="/scratch/cs/vidgen/guanjr/project/latent_wf_vae_new/results/channel4_pretrain_16/validation_samples/step_0095001/generated"
LOG_FILE="${DST_DIR}/copy_log.txt"

echo "----------------------------------------"
echo "Copying videos from:"
echo "  SRC: $SRC_DIR"
echo "  DST: $DST_DIR"
echo "  LOG: $LOG_FILE"
echo "----------------------------------------"

# 检查源与目标是否存在
if [ ! -d "$SRC_DIR" ]; then
    echo "❌ Source directory does not exist: $SRC_DIR"
    exit 1
fi
if [ ! -d "$DST_DIR" ]; then
    echo "❌ Destination directory does not exist: $DST_DIR"
    exit 1
fi

# 记录日志
echo "$(date '+%Y-%m-%d %H:%M:%S') - Start copying 0~999.mp4 to 1000~1999.mp4" >> "$LOG_FILE"

# 主循环
for i in $(seq 0 999); do
    src_file="${SRC_DIR}/${i}.mp4"
    dst_file="${DST_DIR}/$((i+1000)).mp4"

    if [ ! -f "$src_file" ]; then
        echo "⚠️  Missing source file: $src_file" >> "$LOG_FILE"
        continue
    fi

    # 如果目标文件已存在，则跳过（安全防护）
    if [ -f "$dst_file" ]; then
        echo "⚠️  Skipping existing file: $dst_file" >> "$LOG_FILE"
        continue
    fi

    cp "$src_file" "$dst_file"
    echo "✅ Copied $src_file -> $dst_file" >> "$LOG_FILE"
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - Copy finished!" >> "$LOG_FILE"
echo "✅ All done. Log written to $LOG_FILE"
