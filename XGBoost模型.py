import pandas as pd
import numpy as np
from xgboost import XGBClassifier  # 修改1：导入XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

# 读取数据（保持原样）
df = pd.read_csv(".\output\combined_data.csv")
# 处理时间戳，提取 hour 作为特征（保持原样）
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

# 选择特征和目标变量（保持原样）
features = ["soc", "hour"]
target = "combined_anomaly"

# 去除缺失值（保持原样）
df = df.dropna(subset=features + [target])

# 划分训练集和测试集（保持原样）
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# 修改2：使用XGBoost模型
xgb_model = XGBClassifier(
    n_estimators=100,         # 保持树数量一致
    random_state=42,          # 保持随机种子一致
    eval_metric='logloss',    # 添加XGBoost专用参数
    use_label_encoder=False,  # 禁用旧版标签编码
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # 处理类别不平衡
)

# 模型训练（保持原样）
xgb_model.fit(X_train, y_train)

# 预测（保持原样）
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]  # 获取正类概率

# 计算评估指标（保持原样）
auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")

