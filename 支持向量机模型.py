import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

# 读取数据
df = pd.read_csv("./dataset/combined_data.csv")

# 处理时间戳
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

# 选择特征和目标变量
features = ["soc", "hour"]
target = "combined_anomaly"

# 去除缺失值
df = df.dropna(subset=features + [target])
print(f"Data shape after dropping missing values: {df.shape}")

# 随机抽样 10% 的数据
df_sample = df.sample(frac=0.1, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(df_sample[features], df_sample[target], test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练线性SVM
svm_model = LinearSVC(C=0.1, max_iter=10000, class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 预测
y_pred = svm_model.predict(X_test_scaled)
y_prob = svm_model.decision_function(X_test_scaled)

# 计算评估指标
auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")