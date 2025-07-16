#!/usr/bin/env python3
"""
专门用于运行 process_table.py 的 Python 脚本
替代 bash 脚本，适用于 Windows 环境
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("DeepSeek Simulator - Process Table Runner")
    print("=" * 60)
    
    # 定义文件路径
    input_dir = "test_result"
    output_dir = "test_result"
    prefix = "H800-"
    
    # 输入文件列表
    input_files = {
        "dense_gemm": f"{input_dir}/H800_dense_gemm.csv",
        "group_gemm": f"{input_dir}/H800_group_gemm.csv", 
        "batch_gemm": f"{input_dir}/H800_batch_gemm.csv",
        "mla": f"{input_dir}/H800_mla.csv"
    }
    
    # 检查输入文件是否存在
    print(f"📂 检查输入文件...")
    missing_files = []
    for name, filepath in input_files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {name}: {filepath} ({size} bytes)")
        else:
            print(f"  ❌ {name}: {filepath} (文件不存在)")
            missing_files.append(filepath)
    
    if missing_files:
        print(f"\n❌ 错误：缺少以下输入文件：")
        for file in missing_files:
            print(f"     {file}")
        print("请确保这些文件存在后再运行。")
        return 1
    
    # 构建命令
    cmd = [
        "python", "python/process_table.py",
        "--dense_gemm", input_files["dense_gemm"],
        "--group_gemm", input_files["group_gemm"],
        "--batch_gemm", input_files["batch_gemm"],
        "--mla", input_files["mla"],
        "--output_path", output_dir,
        "--output_prefix", prefix
    ]
    
    print(f"\n🚀 运行命令:")
    print(f"   {' '.join(cmd)}")
    print()
    
    # 运行命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        # 显示输出
        if result.stdout:
            print("📄 程序输出:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  错误信息:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ 程序运行成功!")
            
            # 检查输出文件
            output_files = [
                f"{output_dir}/{prefix}two-microbatch-overlapping.csv",
                f"{output_dir}/{prefix}single-batch-comp-comm-overlapping.csv"
            ]
            
            print(f"\n📊 检查输出文件:")
            for filepath in output_files:
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"  ✅ {filepath} ({size} bytes)")
                else:
                    print(f"  ❌ {filepath} (未生成)")
            
            print(f"\n🎉 任务完成！输出文件已保存在 {output_dir}/ 目录下。")
            return 0
        else:
            print(f"❌ 程序运行失败，退出码: {result.returncode}")
            return result.returncode
            
    except Exception as e:
        print(f"❌ 运行时发生错误: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n按 Enter 键退出...")
    input()
    sys.exit(exit_code) 