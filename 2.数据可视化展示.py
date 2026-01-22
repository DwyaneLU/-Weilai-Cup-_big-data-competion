import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取合并数据
df = pd.read_csv('output/combined_data.csv', parse_dates=['timestamp'])
df['month'] = df['timestamp'].dt.month  # 提取月份
df['hour'] = df['timestamp'].dt.hour    # 提取小时

# --------------------------
# 1. 月度故障报警分析（双重坐标轴）
# --------------------------
plt.figure(figsize=(14, 7))

# 统计故障次数和严重等级
fault_data = df[df['alarm_level'] > 0].groupby('month').agg(
    total_faults=('alarm_level', 'size'),
    avg_severity=('alarm_level', 'mean')
).reset_index()

# 主坐标轴：故障次数柱状图
ax1 = sns.barplot(x='month', y='total_faults', data=fault_data,
                 palette='Blues_d', alpha=0.8)
plt.title('月度故障报警趋势分析（2022年1-9月）', fontsize=14)
plt.xlabel('月份', fontsize=12)
plt.ylabel('故障次数', fontsize=12)

# 次坐标轴：故障等级折线图
ax2 = plt.twinx()
sns.lineplot(x='month', y='avg_severity', data=fault_data,
            marker='o', color='#e74c3c', linewidth=2.5, ax=ax2)
ax2.set_ylabel('平均故障等级', fontsize=12)
plt.grid(False)
plt.savefig('output/月度故障分析.jpg', dpi=300, bbox_inches='tight')

# --------------------------
# 2. 充电行为分析（交互式热力图）
# --------------------------
# 筛选充电时段数据
charging_data = df[df['charging_status'] == 3].copy()
charging_data['weekday'] = df['timestamp'].dt.dayofweek  # 0=周一

# 创建热力图数据
heatmap_data = charging_data.pivot_table(
    index='weekday',
    columns='hour',
    values='total_current',
    aggfunc='count'
)

# 可视化
plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False,
           cbar_kws={'label': '充电次数'})
plt.title('充电行为热力图（周 vs 小时）', fontsize=14)
plt.xlabel('小时', fontsize=12)
plt.ylabel('星期', fontsize=12)
plt.yticks(ticks=range(7),
          labels=['周一','周二','周三','周四','周五','周六','周日'])
plt.savefig('output/充电热力图.jpg', dpi=300)

# --------------------------
# 3. 用车规律分析（动态箱线图）
# --------------------------
# 按小时分析车速分布
plt.figure(figsize=(14, 7))
sns.boxplot(x='hour', y='speed', data=df,
           palette='Set2', showfliers=False)
plt.title('不同时段车速分布箱线图', fontsize=14)
plt.xlabel('小时', fontsize=12)
plt.ylabel('车速 (km/h)', fontsize=12)
plt.savefig('output/车速分布.jpg', dpi=300)

# --------------------------
# 4. 电池健康趋势（双指标折线图）
# --------------------------
battery_health = df.groupby('month').agg(
    max_voltage=('max_cell_voltage', 'mean'),
    min_voltage=('min_cell_voltage', 'mean')
).reset_index()

plt.figure(figsize=(14,7))
sns.lineplot(x='month', y='max_voltage', data=battery_health,
            marker='o', label='最高单体电压', linewidth=2.5)
sns.lineplot(x='month', y='min_voltage', data=battery_health,
            marker='s', label='最低单体电压', linewidth=2.5)
plt.title('电池电压衰减趋势', fontsize=14)
plt.xlabel('月份', fontsize=12)
plt.ylabel('电压值 (V)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('output/电池健康趋势.jpg', dpi=300)

# --------------------------
# 5. 交互式三维散点图（电压-温度-故障）
# --------------------------
fig = px.scatter_3d(df.sample(2000),  # 抽样部分数据
                   x='total_voltage',
                   y='motor_temp',
                   z='soc',
                   color='alarm_level',
                   title='核心参数三维分布',
                   labels={
                       'total_voltage': '总电压 (V)',
                       'motor_temp': '电机温度 (°C)',
                       'soc': 'SOC (%)',
                       'alarm_level': '故障等级'
                   })
fig.write_html("output/三维参数分布.html")