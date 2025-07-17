#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

def check_csv_file(filepath, file_type):
    """检查单个CSV文件"""
    print(f"\n{'='*60}")
    print(f"检查 {file_type} 文件: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return False
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ 文件读取成功")
        print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
        print(f"列名: {list(df.columns)}")
        
        # 显示前几行数据
        print(f"\n前5行数据:")
        print(df.head())
        
        # 检查特定列
        if file_type == "MLA":
            required_cols = ['mean_sk', 's_q', 'b', 'varlen', 'h_q', 'latency']
            print(f"\n检查必需列:")
            for col in required_cols:
                if col in df.columns:
                    unique_values = df[col].unique()
                    print(f"✅ {col}: {len(unique_values)} 个唯一值")
                    if len(unique_values) <= 20:
                        print(f"   值: {sorted(unique_values)}")
                    else:
                        print(f"   范围: {min(unique_values)} ~ {max(unique_values)}")
                else:
                    print(f"❌ 缺少列: {col}")
        
        elif file_type in ["Dense GEMM", "Batch GEMM"]:
            required_cols = ['m', 'tp', 'matrix_idx', 'time_us']
            print(f"\n检查必需列:")
            for col in required_cols:
                if col in df.columns:
                    unique_values = df[col].unique()
                    print(f"✅ {col}: {len(unique_values)} 个唯一值")
                    if len(unique_values) <= 20:
                        print(f"   值: {sorted(unique_values)}")
                    else:
                        print(f"   范围: {min(unique_values)} ~ {max(unique_values)}")
                else:
                    print(f"❌ 缺少列: {col}")
        
        elif file_type == "Group GEMM":
            required_cols = ['d', 'b_mla', 'm_per_group', 'matrix_idx', 'time_us']
            print(f"\n检查必需列:")
            for col in required_cols:
                if col in df.columns:
                    unique_values = df[col].unique()
                    print(f"✅ {col}: {len(unique_values)} 个唯一值")
                    if len(unique_values) <= 20:
                        print(f"   值: {sorted(unique_values)}")
                    else:
                        print(f"   范围: {min(unique_values)} ~ {max(unique_values)}")
                else:
                    print(f"❌ 缺少列: {col}")
        
        return True
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("           CSV 数据检查工具")
    print("="*60)
    
    # 定义要检查的文件
    files_to_check = [
        ("results/TC260_dense_gemm.csv", "Dense GEMM"),
        ("results/TC260_group_gemm.csv", "Group GEMM"),
        ("results/TC260_batch_gemm.csv", "Batch GEMM"),
        ("results/TC260_mla.csv", "MLA"),
        # 备用文件名
        ("results/H20_dense_gemm.csv", "Dense GEMM (H20)"),
        ("results/H20_group_gemm.csv", "Group GEMM (H20)"),
        ("results/H20_batch_gemm.csv", "Batch GEMM (H20)"),
        ("results/H20_mla.csv", "MLA (H20)"),
    ]
    
    found_files = 0
    
    for filepath, file_type in files_to_check:
        if os.path.exists(filepath):
            check_csv_file(filepath, file_type)
            found_files += 1
    
    if found_files == 0:
        print(f"\n❌ 没有找到任何CSV文件")
        print(f"请确保以下文件存在于 results/ 目录中:")
        for filepath, file_type in files_to_check[:4]:
            print(f"  - {filepath}")
        print(f"\n建议运行 'python extract_excel_data.py' 先提取Excel数据")
    else:
        print(f"\n🎉 检查完成! 找到 {found_files} 个文件")

if __name__ == "__main__":
    main() 