import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier  # 使用 RidgeClassifier 替换 RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 训练 Ridge 分类模型
# 网格搜索调参
param_grid = {'alpha': [0.1, 1, 10, 100]}
ridge_model = RidgeClassifier(class_weight='balanced', random_state=42)   # 使用岭回归分类器
ridge_grid = GridSearchCV(ridge_model, param_grid, scoring='roc_auc')
ridge_model.fit(X_train, y_train)

# 预测
y_pred = ridge_model.predict(X_test)
y_prob = ridge_model.decision_function(X_test)  # 获取决策函数的值，作为概率预测

# 计算评估指标
auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")