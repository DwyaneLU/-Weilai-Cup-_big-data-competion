import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 配置参数
DATA_DIR = "./data"  # 数据存放目录
OUTPUT_DIR = "./output"  # 输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------
# 通用处理函数
# --------------------------
def process_month(file_name):
    """处理单月数据并返回清洗后数据及异常报告"""
    # 读取数据
    file_path = os.path.join(DATA_DIR, file_name)
    if file_name.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='gbk')


    df.columns = [
        'timestamp', 'vehicle_status', 'charging_status', 'speed', 'total_mileage',
        'total_voltage', 'total_current', 'soc', 'dc_dc_status', 'insulation_resistance',
        'motor_controller_temp', 'motor_speed', 'motor_torque', 'motor_temp',
        'controller_input_voltage', 'controller_current', 'max_cell_voltage',
        'min_cell_voltage', 'max_temp', 'min_temp', 'alarm_level'
    ]

    # 预处理
    df = preprocess(df)

    # 异常检测
    df = detect_anomalies(df)

    # 生成报告
    report = generate_report(df, file_name)

    return df, report


def preprocess(df):
    """数据预处理"""
    # 处理时间字段
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 处理特殊字段格式
    temp_cols = ['motor_controller_temp', 'motor_temp', 'max_temp', 'min_temp']
    for col in temp_cols:
        df[col] = df[col].apply(lambda x: float(str(x).split(':')[1]) if ':' in str(x) else float(x))

    # 处理缺失值
    df.dropna(subset=['timestamp', 'total_voltage', 'total_current'], inplace=True)
    numeric_cols = ['speed', 'total_mileage', 'total_voltage', 'total_current', 'soc', 'insulation_resistance']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def detect_anomalies(df):
    """复杂异常检测"""
    # Z-Score检测
    zscore_threshold = 3
    for col in ['total_voltage', 'total_current', 'soc', 'max_cell_voltage']:
        z_scores = np.abs(stats.zscore(df[col]))
        df[f'zscore_{col}'] = z_scores > zscore_threshold

    # IQR检测
    iqr_factor = 1.5
    for col in ['motor_temp', 'max_temp']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[f'iqr_{col}'] = (df[col] < (q1 - iqr_factor * iqr)) | (df[col] > (q3 + iqr_factor * iqr))

    # 孤立森林
    iso_cols = ['total_voltage', 'total_current', 'soc', 'motor_temp']
    iso_data = StandardScaler().fit_transform(df[iso_cols])
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    df['isolation_forest'] = iso_model.fit_predict(iso_data) == -1

    # 时间序列分析
    window_size = 10
    df['soc_ma'] = df['soc'].rolling(window=window_size).mean()
    df['soc_std'] = df['soc'].rolling(window=window_size).std()
    df['ts_soc'] = (np.abs(df['soc'] - df['soc_ma']) > 2 * df['soc_std'])

    # 综合异常标记
    df['combined_anomaly'] = df[['zscore_total_voltage', 'iqr_motor_temp', 'ts_soc']].any(axis=1)

    return df


def generate_report(df, file_name):
    """生成异常报告"""
    critical = df[df['combined_anomaly']].copy()
    critical['anomaly_type'] = ""

    conditions = [
        (critical['zscore_total_voltage'], "电压异常"),
        (critical['iqr_motor_temp'], "温度异常"),
        (critical['ts_soc'], "SOC突变")
    ]

    for condition, label in conditions:
        critical.loc[condition, 'anomaly_type'] += label + ";"

    # 保存报告
    month = file_name.split('_')[1].split('.')[0]
    report_path = os.path.join(OUTPUT_DIR, f'anomaly_report_month_{month}.csv')
    critical[['timestamp', 'total_voltage', 'total_current', 'soc', 'anomaly_type']].to_csv(report_path, index=False)

    return critical


# --------------------------
# 批量处理所有月份
# --------------------------
all_data = []
file_list = [f'month_{str(i).zfill(2)}.xlsx' if i != 1 else 'month_01.xlsx' for i in range(1, 10)]

for file in file_list:
    print(f"Processing {file}...")
    cleaned_df, report = process_month(file)
    print(f"发现异常记录 {len(report)} 条")
    all_data.append(cleaned_df)

# 合并所有数据
full_df = pd.concat(all_data, ignore_index=True)
full_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_data.csv'), index=False)

print("处理完成！结果已保存至output目录")