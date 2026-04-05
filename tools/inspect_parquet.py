"""
Parquet 数据集检查工具

用于检查和分析 parquet 文件中的视频数据结构和详细信息

Usage:
    python tools/inspect_parquet.py --parquet_path /path/to/file.parquet
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


def check_file_exists(parquet_path):
    """检查文件是否存在"""
    print("="*70)
    print("1. 检查 Parquet 文件")
    print("="*70)
    
    if os.path.exists(parquet_path):
        print(f"✓ Parquet 文件存在: {parquet_path}")
        file_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
        print(f"  文件大小: {file_size:.2f} MB")
        return True
    else:
        print(f"✗ Parquet 文件不存在: {parquet_path}")
        return False


def load_parquet(parquet_path):
    """加载 parquet 文件"""
    print("\n" + "="*70)
    print("2. 加载 Parquet 文件")
    print("="*70)
    
    print("正在加载 parquet 文件...")
    df = pd.read_parquet(parquet_path)
    print(f"✓ 成功加载，包含 {len(df):,} 条记录")
    return df


def show_basic_info(df):
    """显示基本信息"""
    print("\n" + "="*70)
    print("3. DataFrame 基本信息")
    print("="*70)
    print(f"总行数: {len(df):,}")
    print(f"总列数: {len(df.columns)}")
    print(f"\n列名和数据类型:")
    print("-"*70)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = len(df) - non_null
        print(f"  {col:25s} | {str(dtype):15s} | 非空: {non_null:8,} | 空值: {null_count:8,}")


def show_data_preview(df):
    """显示数据预览"""
    print("\n" + "="*70)
    print("4. 数据预览")
    print("="*70)
    print("\n前 10 行数据:")
    print(df.head(10).to_string())


def show_column_details(df):
    """显示每列的详细信息"""
    print("\n" + "="*70)
    print("5. 所有列的详细信息")
    print("="*70)
    
    for col in df.columns:
        print(f"\n{'='*70}")
        print(f"列名: {col}")
        print(f"{'='*70}")
        print(f"数据类型: {df[col].dtype}")
        print(f"非空值数量: {df[col].count():,}")
        print(f"空值数量: {df[col].isnull().sum():,}")
        print(f"唯一值数量: {df[col].nunique():,}")
        
        # 显示示例值
        print(f"\n前 5 个值:")
        for i, val in enumerate(df[col].head(5), 1):
            val_str = str(val)[:100]  # 限制显示长度
            print(f"  {i}. {val_str}")
        
        # 如果是数值类型，显示统计信息
        if df[col].dtype in ['int64', 'float64']:
            print(f"\n统计信息:")
            print(f"  最小值: {df[col].min()}")
            print(f"  最大值: {df[col].max()}")
            print(f"  平均值: {df[col].mean():.2f}")
            print(f"  中位数: {df[col].median():.2f}")
            print(f"  标准差: {df[col].std():.2f}")
        
        # 如果唯一值不多，显示值分布
        if df[col].nunique() <= 20 and df[col].nunique() > 1:
            print(f"\n值分布:")
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                print(f"  {val}: {count:,} ({count/len(df)*100:.2f}%)")


def test_video_loading(df, video_column=None):
    """测试加载视频文件"""
    print("\n" + "="*70)
    print("6. 视频路径分析和测试加载")
    print("="*70)
    
    # 查找可能的视频路径列
    if video_column is None:
        video_path_candidates = []
        for col in df.columns:
            col_lower = col.lower()
            # 只选择字符串类型的列作为候选
            if df[col].dtype == 'object' and ('path' in col_lower or 'url' in col_lower or 'file' in col_lower):
                video_path_candidates.append(col)
        
        print(f"可能的视频路径列: {video_path_candidates}")
        
        if not video_path_candidates:
            print("\n未找到明显的视频路径列")
            print("提示: 视频路径列应该是字符串类型，包含 'path', 'url' 或 'file' 关键字")
            return
        
        # 优先选择包含 'path' 或 'file' 的列，其次是 'url'
        video_col = None
        for keyword in ['path', 'file', 'url']:
            for col in video_path_candidates:
                if keyword in col.lower():
                    video_col = col
                    break
            if video_col:
                break
        
        if not video_col:
            video_col = video_path_candidates[0]
    else:
        video_col = video_column
        if video_col not in df.columns:
            print(f"错误: 列 '{video_col}' 不存在于数据中")
            return
    
    print(f"\n使用列 '{video_col}' 进行视频路径分析")
    print("="*70)
    
    # 显示示例路径
    print(f"\n前 10 个视频路径:")
    for i, path in enumerate(df[video_col].head(10), 1):
        path_str = str(path)[:150]  # 限制显示长度
        print(f"  {i}. {path_str}")
    
    # 检查是否是URL
    sample_path = str(df[video_col].iloc[0]) if len(df) > 0 else ""
    is_url = 'http' in sample_path.lower()
    
    # 检查文件是否存在（仅对本地路径）
    if not is_url:
        print(f"\n检查前 20 个文件是否存在...")
        exists_count = 0
        not_exists_count = 0
        sample_size = min(20, len(df))
        
        for path in df[video_col].head(sample_size):
            if pd.notna(path) and os.path.exists(str(path)):
                exists_count += 1
            else:
                not_exists_count += 1
                if not_exists_count <= 3:
                    print(f"  ✗ 不存在: {path}")
        
        print(f"\n文件存在情况: {exists_count}/{sample_size} 存在, {not_exists_count}/{sample_size} 不存在")
    else:
        print(f"\n路径为 URL 格式，跳过本地文件存在性检查")
    
    # 统计文件扩展名（如果是路径格式）
    print(f"\n路径格式分析:")
    try:
        # 检查是否是URL或本地路径
        sample_paths = df[video_col].dropna().head(5)
        is_url = any('http' in str(p).lower() for p in sample_paths)
        is_local_path = any('/' in str(p) or '\\' in str(p) for p in sample_paths)
        
        if is_url:
            print("  格式: URL 链接")
            # 统计 URL 扩展名
            def get_url_extension(url):
                try:
                    if pd.notna(url):
                        url_str = str(url)
                        # 从URL中提取文件名部分
                        if '.' in url_str:
                            return Path(url_str.split('?')[0]).suffix
                except:
                    pass
                return None
            
            extensions = df[video_col].apply(get_url_extension)
            ext_counts = extensions.value_counts()
            print(f"\n文件扩展名统计:")
            for ext, count in ext_counts.head(10).items():
                if ext:
                    print(f"  {ext}: {count:,} ({count/len(df)*100:.2f}%)")
        elif is_local_path:
            print("  格式: 本地文件路径")
            # 统计本地路径扩展名
            extensions = df[video_col].apply(lambda x: Path(str(x)).suffix if pd.notna(x) else None)
            ext_counts = extensions.value_counts()
            print(f"\n文件扩展名统计:")
            for ext, count in ext_counts.head(10).items():
                if ext:
                    print(f"  {ext}: {count:,} ({count/len(df)*100:.2f}%)")
        else:
            print("  格式: 无法识别的路径格式")
    except Exception as e:
        print(f"  路径格式分析失败: {e}")
    
    # 尝试使用 decord 加载第一个存在的视频（仅本地文件）
    if not is_url:
        try:
            import decord
            from decord import VideoReader, cpu
            
            test_video_path = None
            for path in df[video_col].head(50):
                if pd.notna(path) and os.path.exists(str(path)):
                    test_video_path = str(path)
                    break
            
            if test_video_path:
                print(f"\n测试加载视频: {test_video_path}")
                print("-"*70)
                
                vr = VideoReader(test_video_path, ctx=cpu(0))
                
                print(f"✓ 视频加载成功!")
                print(f"  总帧数: {len(vr)}")
                print(f"  FPS: {vr.get_avg_fps():.2f}")
                print(f"  时长: {len(vr) / vr.get_avg_fps():.2f} 秒")
                
                # 获取第一帧
                frame = vr[0].asnumpy()
                print(f"  帧形状: {frame.shape} (H x W x C)")
                print(f"  分辨率: {frame.shape[1]} x {frame.shape[0]}")
            else:
                print("\n未找到可用的视频文件进行测试")
                
        except ImportError:
            print("\n未安装 decord 库，跳过视频加载测试")
            print("安装命令: pip install decord")
        except Exception as e:
            print(f"\n测试视频加载时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n路径为 URL 格式，跳过本地视频加载测试")
        print(f"提示: URL 视频需要先下载到本地才能使用 decord 加载")


def generate_summary(df, parquet_path, output_json=None):
    """生成数据摘要"""
    print("\n" + "="*70)
    print("7. 完整数据摘要")
    print("="*70)
    
    print(f"\nParquet 文件: {parquet_path}")
    print(f"文件大小: {os.path.getsize(parquet_path) / (1024**2):.2f} MB")
    print(f"\n总记录数: {len(df):,}")
    print(f"总列数: {len(df.columns)}")
    print(f"\n列名: {', '.join(df.columns)}")
    print(f"\n内存占用: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # 创建摘要字典
    summary = {
        'parquet_file': parquet_path,
        'file_size_mb': round(os.path.getsize(parquet_path) / (1024 * 1024), 2),
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': {}
    }
    
    for col in df.columns:
        summary['columns'][col] = {
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].count()),
            'null_count': int(df[col].isnull().sum()),
            'unique_values': int(df[col].nunique()),
        }
    
    print("\n" + "-"*70)
    print("JSON 格式摘要:")
    print("-"*70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 保存到文件
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 摘要已保存到: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description='检查和分析 parquet 文件中的视频数据结构'
    )
    parser.add_argument(
        '--parquet_path',
        type=str,
        required=True,
        help='Parquet 文件路径'
    )
    parser.add_argument(
        '--video_column',
        type=str,
        default=None,
        help='视频路径列名（可选，自动检测）'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='输出 JSON 摘要文件路径（可选）'
    )
    parser.add_argument(
        '--skip_video_test',
        action='store_true',
        help='跳过视频加载测试'
    )
    parser.add_argument(
        '--skip_details',
        action='store_true',
        help='跳过详细列信息（加快速度）'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Parquet 数据集检查工具")
    print("="*70)
    
    # 1. 检查文件存在
    if not check_file_exists(args.parquet_path):
        return
    
    # 2. 加载文件
    df = load_parquet(args.parquet_path)
    
    # 3. 显示基本信息
    show_basic_info(df)
    
    # 4. 显示数据预览
    show_data_preview(df)
    
    # 5. 显示列详细信息
    if not args.skip_details:
        show_column_details(df)
    else:
        print("\n跳过详细列信息（使用 --skip_details）")
    
    # 6. 测试视频加载
    if not args.skip_video_test:
        test_video_loading(df, args.video_column)
    else:
        print("\n跳过视频加载测试（使用 --skip_video_test）")
    
    # 7. 生成摘要
    generate_summary(df, args.parquet_path, args.output_json)
    
    print("\n" + "="*70)
    print("✓ 检查完成!")
    print("="*70)


if __name__ == '__main__':
    main()

