import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

# 读取数据
df = pd.read_csv("./dataset/combined_data.csv")

# 处理时间戳，提取 hour 作为特征
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

# 选择特征和目标变量
features = ["soc", "hour"]
target = "combined_anomaly"

# 去除缺失值
df = df.dropna(subset=features + [target])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 计算类别权重
class_weight_dict = {0: 1, 1: (len(y_train) / (2 * np.bincount(y_train)[1]))}
print(f"Calculated class weights: {class_weight_dict}")

# 训练 LightGBM 模型，增加 n_estimators
lgb_model = lgb.LGBMClassifier(n_estimators=150, num_leaves=35, max_depth=6, class_weight=class_weight_dict, random_state=42)
lgb_model.fit(X_train, y_train)

# 预测
y_prob = lgb_model.predict_proba(X_test)[:, 1]

# 调整阈值
threshold = 0.28
predictions_adjusted = (y_prob >= threshold).astype(int)

# 计算评估指标
auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, predictions_adjusted)

print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")