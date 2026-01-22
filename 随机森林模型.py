import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 引入随机森林分类器
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv(".\output\combined_data.csv")

# 处理时间戳，提取 hour 作为特征
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

# 选择特征和目标变量
features = ["soc", "hour"]
target = "combined_anomaly"

# 去除缺失值
df = df.dropna(subset=features + [target])

# 重置索引
df = df.reset_index(drop=True)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 将目标变量转换为 NumPy 数组
y_train = np.array(y_train)
y_test = np.array(y_test)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 设置类权重，解决类别不平衡
class_weights = {0: 1, 1: 15}  # 通过调整权重帮助模型更重视少数类

# 创建随机森林模型
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weights)

# 训练模型
rf_model.fit(X_train_scaled, y_train)

# 预测
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]  # 获取预测的概率

# 计算评估指标
auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")