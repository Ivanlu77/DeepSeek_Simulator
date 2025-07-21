import pandas as pd
import os

def extract_excel_to_csv(excel_file="DSV3_MLA_GEMM_260.xlsx", output_dir="data"):
    """
    从Excel文件中提取batch、dense、flash、group数据并转换为CSV格式
    """
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 读取Excel文件的所有工作表
        excel_data = pd.read_excel(excel_file, sheet_name=None)
        
        print(f"找到的工作表: {list(excel_data.keys())}")
        
        # 遍历每个工作表
        for sheet_name, df in excel_data.items():
            print(f"\n处理工作表: {sheet_name}")
            print(f"形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            # 显示前几行数据
            print("前5行数据:")
            print(df.head())
            
            # 根据工作表名称或内容判断数据类型
            if 'batch' in sheet_name.lower():
                output_file = os.path.join(output_dir, "TC260_batch_gemm.csv")
                df.to_csv(output_file, index=False)
                print(f"Batch GEMM 数据已保存到: {output_file}")
                
            elif 'dense' in sheet_name.lower():
                output_file = os.path.join(output_dir, "TC260_dense_gemm.csv")
                df.to_csv(output_file, index=False)
                print(f"Dense GEMM 数据已保存到: {output_file}")
                
            elif 'flash' in sheet_name.lower() or 'mla' in sheet_name.lower():
                output_file = os.path.join(output_dir, "TC260_mla.csv")
                df.to_csv(output_file, index=False)
                print(f"Flash/MLA 数据已保存到: {output_file}")
                
            elif 'group' in sheet_name.lower():
                output_file = os.path.join(output_dir, "TC260_group_gemm.csv")
                df.to_csv(output_file, index=False)
                print(f"Group GEMM 数据已保存到: {output_file}")
            
            else:
                # 如果工作表名称不明确，根据内容尝试判断
                output_file = os.path.join(output_dir, f"TC260_{sheet_name.lower().replace(' ', '_')}.csv")
                df.to_csv(output_file, index=False)
                print(f"数据已保存到: {output_file}")
    
    except Exception as e:
        print(f"处理Excel文件时出错: {e}")
        # 尝试其他方法读取
        try:
            # 尝试读取第一个工作表
            df = pd.read_excel(excel_file)
            print(f"使用默认方法读取到数据，形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            # 保存为通用CSV
            output_file = os.path.join(output_dir, "TC260_data.csv")
            df.to_csv(output_file, index=False)
            print(f"数据已保存到: {output_file}")
            
        except Exception as e2:
            print(f"备用方法也失败: {e2}")

if __name__ == "__main__":
    extract_excel_to_csv() 