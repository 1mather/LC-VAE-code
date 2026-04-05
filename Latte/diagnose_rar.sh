#!/bin/bash
# 诊断 UCF101.rar 文件问题

echo "========================================"
echo "UCF101.rar 文件诊断工具"
echo "========================================"

RAR_FILE="${1:-UCF101.rar}"

echo ""
echo "1. 检查文件是否存在..."
if [ ! -f "$RAR_FILE" ]; then
    echo "❌ 文件不存在: $RAR_FILE"
    exit 1
fi
echo "✅ 文件存在"

echo ""
echo "2. 检查文件大小..."
FILE_SIZE=$(stat -c%s "$RAR_FILE" 2>/dev/null || stat -f%z "$RAR_FILE" 2>/dev/null)
EXPECTED_SIZE=6932971618  # 大约 6.5 GB
echo "   实际大小: $FILE_SIZE bytes ($(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo 'unknown'))"
echo "   期望大小: $EXPECTED_SIZE bytes (约 6.5 GB)"

if [ "$FILE_SIZE" -lt "$EXPECTED_SIZE" ]; then
    echo "⚠️  文件可能下载不完整！"
    echo ""
    echo "解决方案："
    echo "1. 删除不完整的文件: rm $RAR_FILE"
    echo "2. 重新下载: wget --continue --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    exit 1
else
    echo "✅ 文件大小正常"
fi

echo ""
echo "3. 检查文件类型..."
file "$RAR_FILE"

echo ""
echo "4. 检查文件头（前20字节）..."
xxd -l 20 "$RAR_FILE"

echo ""
echo "5. 尝试用 unrar 测试..."
if command -v unrar &> /dev/null; then
    unrar t "$RAR_FILE" 2>&1 | head -20
else
    echo "⚠️  unrar 未安装"
    echo "   安装方法: sudo yum install unrar"
fi

echo ""
echo "========================================"
echo "诊断完成"
echo "========================================"




