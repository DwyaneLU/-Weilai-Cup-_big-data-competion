import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

# 读取训练数据
df = pd.read_csv("./output/combined_data.csv")

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

# 训练 LightGBM 模型
lgb_model = lgb.LGBMClassifier(n_estimators=150, num_leaves=35, max_depth=6, class_weight=class_weight_dict, random_state=42)
lgb_model.fit(X_train, y_train)

# ---------- 新增敏感度分析部分 ----------
# 在测试集上预测概率
y_prob_test = lgb_model.predict_proba(X_test)[:, 1]

# 生成阈值列表并计算 Recall 和 FPR
thresholds = np.linspace(0, 1, 100)
recalls = []
fprs = []

for thresh in thresholds:
    y_pred = (y_prob_test >= thresh).astype(int)
    recalls.append(recall_score(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fprs.append(fpr)

# 绘制 Recall 和 FPR 曲线
plt.figure(figsize=(10, 6))
plt.plot(thresholds, recalls, 'b-', label='Recall')
plt.plot(thresholds, fprs, 'r-', label='FPR')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.legend(loc='best')
plt.title('Threshold vs. Recall & FPR')
plt.grid(True)
plt.show()

# 选择最优阈值（Recall高且FPR适中）
# 方法1: Youden指数（Recall - FPR）最大化
youden = np.array(recalls) - np.array(fprs)
best_idx = np.argmax(youden)
best_threshold = thresholds[best_idx]

# 方法2: 确保Recall不低于80%时FPR最低（根据需要调整阈值）
# min_recall = 0.8
# valid_indices = np.where(np.array(recalls) >= min_recall)[0]
# if valid_indices.size > 0:
#     best_idx = valid_indices[np.argmin(np.array(fprs)[valid_indices])]
# else:
#     best_idx = np.argmax(recalls)
# best_threshold = thresholds[best_idx]

print(f"最优阈值: {best_threshold:.2f}")
print(f"对应 Recall: {recalls[best_idx]:.2f}, FPR: {fprs[best_idx]:.2f}")

# ---------- 使用最优阈值预测10月数据 ----------
# 读取10月数据
df_month_10 = pd.read_excel("month_10.xlsx")

# 处理时间列
df_month_10["数据采集时间"] = pd.to_datetime(df_month_10["数据采集时间"])
df_month_10["hour"] = df_month_10["数据采集时间"].dt.hour

# 特征选择
features_month_10 = ["SOC", "hour"]
X_month_10 = df_month_10[features_month_10]

# 预测概率
y_prob_month_10 = lgb_model.predict_proba(X_month_10)[:, 1]

# 应用最优阈值生成预测标签
predictions_month_10 = (y_prob_month_10 >= best_threshold).astype(int)

# 映射报警等级
df_month_10["报警_等级"] = np.clip(predictions_month_10 * 3, 1, 3)

