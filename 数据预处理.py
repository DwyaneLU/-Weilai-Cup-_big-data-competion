import pandas as pd
import numpy as np
from pathlib import Path


# 1. 数据读取与合并
def load_and_merge_data(folder_path):
    all_data = []


    for month in range(1, 10):
        file_name = f"month_{month:02d}.xlsx"
        file_path = Path(folder_path) / file_name

        try:

            df = pd.read_excel(file_path, sheet_name=f'month_{month:02d}', engine='openpyxl')


            if len(df.columns) != 21:

                df = df.dropna(subset=df.columns[:21], how='any')

            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")

    merged_df = pd.concat(all_data, ignore_index=True)
    return merged_df


# 2. 数据预处理
def preprocess_data(df):
    # 转换时间列为datetime类型
    df['数据采集时间'] = pd.to_datetime(df['数据采集时间'])
    df = df.sort_values('数据采集时间').reset_index(drop=True)

    # 设置时间索引用于插值
    df.set_index('数据采集时间', inplace=True)

    # 缺失值处理 - 线性插值
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    # 重置索引
    df = df.reset_index()

    # 定义合理范围
    valid_ranges = {
        '车速': (0, 150),
        '总电流': (-200, 200),
        'SOC': (0, 100),
        '电池单体电压最高值': (3.0, 4.2),
        '电池单体电压最低值': (3.0, 4.2)
    }

    # 异常值修正
    for col, (min_val, max_val) in valid_ranges.items():
        # 修正负车速
        if col == '车速':
            df[col] = np.where(df[col] < 0, 0, df[col])
        # 使用边界值修正
        df[col] = df[col].clip(lower=min_val, upper=max_val)

    return df


# 3. 执行预处理流程
if __name__ == "__main__":
    # 假设数据文件存放在当前目录的data文件夹中
    raw_data = load_and_merge_data('./data')

    print(f"原始数据量: {len(raw_data)}行")

    # 执行预处理
    processed_data = preprocess_data(raw_data)

    # 保存处理后的数据
    processed_data.to_csv('processed_vehicle_data.csv', index=False)
    print(f"处理后数据量: {len(processed_data)}行")
    print("数据已保存为 processed_vehicle_data.csv")


# 4. 数据验证
def data_validation(df):
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values[missing_values > 0])

    # 检查异常值
    print("\n数据范围验证:")
    print(f"车速范围: {df['车速'].min()} - {df['车速'].max()}")
    print(f"总电流范围: {df['总电流'].min()} - {df['总电流'].max()}")
    print(f"SOC范围: {df['SOC'].min()} - {df['SOC'].max()}")
    print(f"单体电压范围: {df['电池单体电压最低值'].min()} - {df['电池单体电压最高值'].max()}")


# 执行验证
data_validation(processed_data)