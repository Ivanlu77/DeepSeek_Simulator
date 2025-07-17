import argparse
import pandas as pd
import math
import os
from dataclasses import dataclass
from typing import List, Tuple
from itertools import chain
from common import TestConfig, GPUSpec


def process_data(dense_gemm_file: str, group_gemm_file: str, batch_gemm_file: str, mla_file: str, output_path: str, output_prefix: str, gpu_type: str = "TC260X-64"):
    # 读取输入文件
    dense_df = pd.read_csv(dense_gemm_file)
    group_df = pd.read_csv(group_gemm_file)
    batch_df = pd.read_csv(batch_gemm_file)
    mla_df = pd.read_csv(mla_file)

    # 使用指定的GPU类型创建配置，并从GPU配置中获取序列长度
    sequence_length = GPUSpec[gpu_type]['sequence_length']
    config = TestConfig(gpu=gpu_type, s=sequence_length)
    
    # 添加类型断言确保这些值不为None
    assert config.tp_nums is not None
    assert config.model_config is not None
    assert config.device_nums is not None
    
    configs = config.generate_b_and_m_per_groups()
    results = []
    for d, tp, num_groups, b_mla, m_per_group in configs:
        # 获取各个组件的时间
        tp_index = config.tp_nums.index(tp)
        d_index = config.b_mla_and_device_pair[tp_index]['device_nums'].index(d)
        b_mla_peak = config.b_mla_and_device_pair[tp_index]['b_mla_peak'][d_index]

        # 获取QKV时间，添加错误检查
        try:
            qkv_1 = dense_df[(dense_df['m'] == b_mla) & (dense_df['tp'] == 1) & (dense_df['matrix_idx'] == 1)]['time_us']
            qkv_2 = dense_df[(dense_df['m'] == b_mla) & (dense_df['tp'] == tp) & (dense_df['matrix_idx'] == 2)]['time_us']
            qkv_3 = batch_df[(batch_df['m'] == b_mla) & (batch_df['tp'] == tp) & (batch_df['matrix_idx'] == 3)]['time_us']
            
            if len(qkv_1) == 0 or len(qkv_2) == 0 or len(qkv_3) == 0:
                print(f"警告: 找不到QKV数据 (b_mla={b_mla}, tp={tp})")
                continue
                
            qkv_time = int(qkv_1.iloc[0]) + int(qkv_2.iloc[0]) + int(qkv_3.iloc[0])  # type: ignore
        except (IndexError, KeyError) as e:
            print(f"警告: QKV数据访问失败 (b_mla={b_mla}, tp={tp}): {e}")
            continue

        # 获取注意力时间，添加错误检查
        try:
            h_q_target = config.model_config.q_head / tp
            attn_filter = (mla_df['mean_sk'] == config.s) & \
                         (mla_df['s_q'] == 1) & \
                         (mla_df['b'] == b_mla) & \
                         (mla_df['varlen'] == True) & \
                         (mla_df['h_q'] == h_q_target)
            
            attn_data = mla_df[attn_filter]['latency']
            
            if len(attn_data) == 0:
                print(f"警告: 找不到MLA数据 (s={config.s}, b={b_mla}, h_q={h_q_target})")
                print(f"可用的s值: {mla_df['mean_sk'].unique()}")
                print(f"可用的b值: {mla_df['b'].unique()}")
                print(f"可用的h_q值: {mla_df['h_q'].unique()}")
                continue
                
            attn_time = int(attn_data.iloc[0] * 1000)  # type: ignore
        except (IndexError, KeyError) as e:
            print(f"警告: MLA数据访问失败 (b_mla={b_mla}, tp={tp}): {e}")
            continue

        # 获取输出时间，添加错误检查
        try:
            o_1 = dense_df[(dense_df['m'] == b_mla) & (dense_df['tp'] == tp) & (dense_df['matrix_idx'] == 4)]['time_us']
            o_2 = batch_df[(batch_df['m'] == b_mla) & (batch_df['tp'] == tp) & (batch_df['matrix_idx'] == 9)]['time_us']
            
            if len(o_1) == 0 or len(o_2) == 0:
                print(f"警告: 找不到输出数据 (b_mla={b_mla}, tp={tp})")
                continue
                
            o_time = int(o_1.iloc[0]) + int(o_2.iloc[0])  # type: ignore
        except (IndexError, KeyError) as e:
            print(f"警告: 输出数据访问失败 (b_mla={b_mla}, tp={tp}): {e}")
            continue

        # 获取共享时间
        try:
            shared_time = int(dense_df[(dense_df['m'] == b_mla) & 
                                     (dense_df['matrix_idx'].isin([5, 6]))]['time_us'].sum())
        except Exception as e:
            print(f"警告: 共享数据访问失败 (b_mla={b_mla}): {e}")
            shared_time = 0

        # 获取组GEMM时间，添加错误检查
        try:
            up_data = group_df[(group_df['d'] == d) & (group_df['b_mla'] == b_mla) & 
                              (group_df['m_per_group'] == m_per_group) & (group_df['matrix_idx'] == 7)]['time_us']
            down_data = group_df[(group_df['d'] == d) & (group_df['m_per_group'] == m_per_group) & 
                                (group_df['b_mla'] == b_mla) & (group_df['matrix_idx'] == 8)]['time_us']
            
            if len(up_data) == 0 or len(down_data) == 0:
                print(f"警告: 找不到Group GEMM数据 (d={d}, b_mla={b_mla}, m_per_group={m_per_group})")
                continue
                
            up_gemm = int(up_data.iloc[0])  # type: ignore
            down_gemm = int(down_data.iloc[0])  # type: ignore
        except (IndexError, KeyError) as e:
            print(f"警告: Group GEMM数据访问失败 (d={d}, b_mla={b_mla}, m_per_group={m_per_group}): {e}")
            continue

        dispatch_alltoall = int(
            config.calculate_alltoall_time(d, tp, b_mla, True))
        combine_alltoall = int(
            config.calculate_alltoall_time(d, tp, b_mla, False))

        allreduce = int(config.calculate_allreduce_time(
            tp, b_mla)) if tp > 1 else 0

        # 计算两种模式下的层时间
        # Two microbatch overlapping
        t_moe_layer_two = int(2 * (max(dispatch_alltoall, shared_time + qkv_time) +
                              up_gemm + down_gemm +
                              max(attn_time + o_time + allreduce, combine_alltoall)))
        t_dense_layer_two = int(
            2 * (shared_time + qkv_time + up_gemm + down_gemm + attn_time + o_time + allreduce))
        # Single batch comp-compute overlapping
        t_moe_layer_single = int(max(dispatch_alltoall, shared_time) + qkv_time + up_gemm +
                                 max(down_gemm, combine_alltoall) + attn_time + o_time + allreduce)
        t_dense_layer_single = int(
            shared_time + qkv_time + up_gemm + down_gemm + attn_time + o_time + allreduce)

        # 计算TPOT和吞吐量
        tpot_two = int((t_moe_layer_two * 58 + t_dense_layer_two * 3) / 1000)
        tpot_single = int(
            (t_moe_layer_single * 58 + t_dense_layer_single * 3) / 1000)

        throughput_two = int(b_mla * 2 * 1000 / tp / tpot_two)
        throughput_single = int(b_mla * 1000 / tp / tpot_single)

        # 存储结果
        base_result = {
            'd': d,
            'tp': tp,
            'b_mla': b_mla,
            'sequence_length': sequence_length,  # 添加序列长度信息
            'gpu_type': gpu_type,                # 添加GPU类型信息
            'QKV(us)': qkv_time,
            'ATTN(us)': attn_time,
            'O(us)': o_time,
            'Shared(us)': shared_time,
            'Up_Gemm(us)': up_gemm,
            'Down_Gemm(us)': down_gemm,
            'Dispatch_AlltoAll(us)': dispatch_alltoall,
            'Combine_AlltoAll(us)': combine_alltoall,
            'AllReduce(us)': allreduce,

        }

        if b_mla * 2 < b_mla_peak * 0.9:
            results.append({
                **base_result,
                't_{dense_layer}(us)': t_dense_layer_two,
                't_{moe_layer}(us)': t_moe_layer_two,
                'TPOT(ms)': tpot_two,
                'Single-Device Throughput(Tokens/s)': throughput_two,
                'mode': 'two-microbatch'
            })
        if b_mla < b_mla_peak * 0.9:
            results.append({
                **base_result,
                't_{dense_layer}(us)': t_dense_layer_single,
                't_{moe_layer}(us)': t_moe_layer_single,
                'TPOT(ms)': tpot_single,
                'Single-Device Throughput(Tokens/s)': throughput_single,
                'mode': 'single-batch'
            })

    # 创建DataFrame并保存结果
    results_df = pd.DataFrame(results)

    # 分别保存两种模式的结果
    two_microbatch_df = results_df[results_df['mode']
                                   == 'two-microbatch'].drop('mode', axis=1)
    single_batch_df = results_df[results_df['mode']
                                 == 'single-batch'].drop('mode', axis=1)

    def float_format(x): return '{:.2f}'.format(
        x) if isinstance(x, float) else x

    # 更新输出文件名，包含GPU类型信息
    two_microbatch_outfile = os.path.join(
        output_path, f"{output_prefix}{gpu_type}-two-microbatch-overlapping.csv")
    single_batch_outfile = os.path.join(
        output_path, f"{output_prefix}{gpu_type}-single-batch-comp-comm-overlapping.csv")
    two_microbatch_df.to_csv(two_microbatch_outfile,
                             index=False, float_format=float_format)
    single_batch_df.to_csv(single_batch_outfile,
                           index=False, float_format=float_format)
    
    print(f"已生成结果文件:")
    print(f"  - {two_microbatch_outfile}")
    print(f"  - {single_batch_outfile}")
    print(f"使用GPU类型: {gpu_type}, 序列长度: {sequence_length}")


def main():
    parser = argparse.ArgumentParser(
        description='Process performance data files')
    parser.add_argument('--dense_gemm', required=True,
                        help='Path to dense_gemm.csv')
    parser.add_argument('--group_gemm', required=True,
                        help='Path to group_gemm.csv')
    parser.add_argument('--batch_gemm', required=True,
                        help='Path to batch_gemm.csv')
    parser.add_argument('--mla', required=True, help='Path to mla.csv')
    parser.add_argument('--output_path',
                        default='.',
                        help='Output directory path (default: current directory)')
    parser.add_argument('--output_prefix',
                        default='',
                        help='Prefix for output files (default: no prefix)')
    parser.add_argument('--gpu_type',
                        default='TC260X-64',
                        choices=list(GPUSpec.keys()),
                        help=f'GPU type to use (choices: {list(GPUSpec.keys())}, default: TC260X-64)')

    args = parser.parse_args()

    process_data(args.dense_gemm, args.group_gemm, args.batch_gemm,
                 args.mla, args.output_path, args.output_prefix, args.gpu_type)


if __name__ == '__main__':
    main()
