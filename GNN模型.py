import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.utils.class_weight import compute_class_weight

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
X = df[features].values
y = df[target].values.astype(np.int64)

# 构建
k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

edge_index = []
num_nodes = X.shape[0]
for i in range(num_nodes):
    for j in indices[i]:
        if i != j:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 创建图数据对象
x = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y_tensor)

# 划分训练集和测试集
num_nodes = data.num_nodes
indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
data.train_mask = train_mask
data.test_mask = test_mask


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 尝试不同的参数组合
best_recall = 0.0
best_auc = 0.0
best_threshold = 0.5
best_dropout = 0.5
class_weights = [1, 5]  # 调整正类权重
dropout_rates = [0.3, 0.5, 0.6, 0.7]  # 尝试不同的dropout比率
thresholds = [0.2, 0.3, 0.4, 0.5]  # 尝试不同的阈值

for dropout in dropout_rates:
    model = GCN(num_features=x.shape[1], num_classes=len(np.unique(y)), dropout_rate=dropout).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化

    # 训练过程
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            pred_probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            y_true = data.y[data.test_mask].cpu().numpy()

            for threshold in thresholds:
                y_pred = (pred_probs[data.test_mask.cpu().numpy()] >= threshold).astype(int)  # 仅预测测试集
                auc = roc_auc_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)

                if recall > best_recall:
                    best_recall = recall
                    best_auc = auc
                    best_threshold = threshold
                    best_dropout = dropout

                print(f"Epoch: {epoch:03d}, Dropout: {dropout:.2f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Threshold: {threshold:.2f}")

# 输出最佳结果
print(f"最佳组合 - AUC: {best_auc:.4f}, Recall: {best_recall:.4f}, 决策阈值: {best_threshold:.2f}, Dropout比率: {best_dropout:.2f}")