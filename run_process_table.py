#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from datetime import datetime


sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

try:
    from common import GPUSpec  # type: ignore
except ImportError:
    # 如果导入失败，定义一个基本的GPU配置
    GPUSpec = {
        "TC260-1dNet": {"volume": 72, "sequence_length": 5000, "gpus_per_node": 16, "ar_bw": 75, "a2a_bw": 150},
        "TC260X-32": {"volume": 72, "sequence_length": 10000, "gpus_per_node": 32, "ar_bw": 75, "a2a_bw": 75},
        "TC260X-64": {"volume": 72, "sequence_length": 20000, "gpus_per_node": 64, "ar_bw": 75, "a2a_bw": 75},
        "TC260X-128": {"volume": 72, "sequence_length": 40000, "gpus_per_node": 128, "ar_bw": 75, "a2a_bw": 75},
        "TC260X-288": {"volume": 72, "sequence_length": 80000, "gpus_per_node": 288, "ar_bw": 75, "a2a_bw": 75},
        "TC260-2dNet": {"volume": 72, "sequence_length": 5000, "gpus_per_node": 16, "ar_bw": 75, "a2a_bw": 75},
        "H20-96": {"volume": 96, "sequence_length": 5000, "gpus_per_node": 8, "ar_bw": 180, "a2a_bw": 180},
        "H800-80": {"volume": 80, "sequence_length": 5000, "gpus_per_node": 8, "ar_bw": 180, "a2a_bw": 180},
    }

def print_gpu_menu():
    """显示GPU选择菜单"""
    print("\n" + "="*60)
    print("           DeepSeek 模拟器 - GPU 性能测试")
    print("="*60)
    print("请选择要测试的GPU类型:")
    print()
    
    gpu_list = list(GPUSpec.keys())
    for i, gpu_type in enumerate(gpu_list, 1):
        gpu_info = GPUSpec[gpu_type]
        sequence_len = gpu_info.get('sequence_length', 'N/A')
        volume = gpu_info.get('volume', 'N/A')
        print(f"{i:2}. {gpu_type:<15} - {volume}GB 显存, 序列长度: {sequence_len}")
    
    print(f"{len(gpu_list)+1:2}. 退出程序")
    print("-"*60)
    return gpu_list

def create_output_directory(gpu_type):
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_result/{gpu_type}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    return output_dir

def get_input_files():
    """获取输入文件路径"""
    base_dir = "data"
    files = {
        "dense_gemm": f"{base_dir}/H800_dense_gemm.csv",
        "group_gemm": f"{base_dir}/H800_group_gemm.csv", 
        "batch_gemm": f"{base_dir}/H800_batch_gemm.csv",
        "mla": f"{base_dir}/H800_mla.csv"
    }
    
    
    return files

def run_process_table(gpu_type, output_dir, input_files):
    """运行process_table.py脚本"""
    cmd = [
        sys.executable, "python/process_table.py",
        "--dense_gemm", input_files["dense_gemm"],
        "--group_gemm", input_files["group_gemm"],
        "--batch_gemm", input_files["batch_gemm"],
        "--mla", input_files["mla"],
        "--output_path", output_dir,
        "--gpu_type", gpu_type
    ]
    
    print(f"\n正在运行命令:")
    print(" ".join(cmd))
    print("\n" + "-"*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("执行成功!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("执行失败!")
        print(f"错误信息: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def main():
    """主函数"""
    while True:
        gpu_list = print_gpu_menu()
        
        try:
            choice = input("请输入选择 (1-{}): ".format(len(gpu_list)+1)).strip()
            choice_num = int(choice)
            
            if choice_num == len(gpu_list) + 1:
                print("退出程序...")
                break
            elif 1 <= choice_num <= len(gpu_list):
                gpu_type = gpu_list[choice_num - 1]
                print(f"\n已选择 GPU: {gpu_type}")
                
                # 显示GPU详细信息
                gpu_info = GPUSpec[gpu_type]
                print(f"显存容量: {gpu_info.get('volume', 'N/A')} GB")
                print(f"序列长度: {gpu_info.get('sequence_length', 'N/A')}")
                print(f"每节点GPU数量: {gpu_info.get('gpus_per_node', 'N/A')}")
                print(f"AllReduce带宽: {gpu_info.get('ar_bw', 'N/A')} GB/s")
                print(f"AllToAll带宽: {gpu_info.get('a2a_bw', 'N/A')} GB/s")
                
                # 确认执行
                confirm = input("\n确认执行测试? (y/N): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    # 创建输出目录
                    output_dir = create_output_directory(gpu_type)
                    
                    # 获取输入文件
                    input_files = get_input_files()
                    
                    # 检查必要文件
                    missing_files = [f for f in input_files.values() if not os.path.exists(f)]
                    if missing_files:
                        print(f"\n缺少以下必要文件:")
                        for f in missing_files:
                            print(f"  - {f}")
                        print("\n请先确保所有必要的CSV文件存在于results目录中。")
                        input("按回车键继续...")
                        continue
                    
                    # 运行测试
                    success = run_process_table(gpu_type, output_dir, input_files)
                    
                    if success:
                        print(f"\n测试完成! 结果已保存到: {output_dir}")
                        print(f"您可以在该目录中找到以下文件:")
                        print(f"  - {gpu_type}-single-batch-comp-comm-overlapping.csv")
                        print(f"  - {gpu_type}-two-microbatch-overlapping.csv")
                    else:
                        print(f"\n测试失败，请检查错误信息。")
                    
                    input("\n按回车键继续...")
                else:
                    print("取消执行。")
                    
            else:
                print("无效选择，请重新输入!")
                
        except ValueError:
            print("请输入有效的数字!")
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，退出...")
            break
        except Exception as e:
            print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main() 