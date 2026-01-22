import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import rcParams
from sklearn.inspection import permutation_importance

# 配置全局绘图参数
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False



def load_data(file_path):
    """数据加载与初步处理"""
    df = pd.read_csv(file_path, parse_dates=['timestamp'])

    # 时间特征工程
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    df['is_night'] = df['hour'].between(22, 5)

    # 动态特征计算
    df['voltage_change'] = df['total_voltage'].pct_change() * 100
    df['temp_gradient'] = df['motor_temp'].diff(60)  # 每小时温度变化

    # 充电状态增强编码
    charging_map = {
        0: {'mode': 'off', 'power_level': 0},
        1: {'mode': 'slow', 'power_level': 1},
        2: {'mode': 'fast', 'power_level': 2},
        3: {'mode': 'fault', 'power_level': np.nan}
    }
    return df.join(pd.json_normalize(df['charging_status'].map(charging_map)))


def feature_engineering(df):
    """特征工程处理"""
    # 选择关键特征
    features = [
        'total_voltage', 'total_current', 'soc',
        'motor_temp', 'voltage_change', 'temp_gradient',
        'hour', 'is_weekend', 'is_night', 'power_level'
    ]

    # 处理分类特征
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[['mode']]).toarray()
    feature_names = encoder.get_feature_names_out(['mode'])

    # 构建特征矩阵
    X = pd.concat([
        df[features],
        pd.DataFrame(encoded_features, columns=feature_names)
    ], axis=1)

    return X.fillna(0), df['alarm_level']


def train_model(X, y):
    """模型训练与评估"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 随机森林参数（经贝叶斯优化）
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 模型评估
    print("\n模型评估报告：")
    print(classification_report(y_test, model.predict(X_test)))

    return model


def visualize_analysis(model, X, y):
    """可视化分析模块"""

    # 特征重要性分析
    importance = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(10, 6))
    importance.nlargest(10).plot(kind='barh', color='#2ecc71')
    plt.title('Top10故障预测特征重要性')
    plt.xlabel("重要性得分")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig("feature_importance.jpg")  # 保存图片
    plt.show()

    # SHAP 值分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP 值分析")
    plt.savefig("shap_analysis.jpg")  # 保存图片
    plt.show()

    # 选择最重要的 2 个特征进行部分依赖分析
    top_features = importance.nlargest(2).index.tolist()

    # 确保 y 是多分类，并选择第一个类别作为 target
    unique_classes = np.unique(y)
    target_class = unique_classes[0] if len(unique_classes) > 1 else None

    if target_class is not None:
        PartialDependenceDisplay.from_estimator(model, X, features=top_features, target=target_class)
    else:
        PartialDependenceDisplay.from_estimator(model, X, features=top_features)

    plt.suptitle("部分依赖分析 (PDP)")
    plt.savefig("pdp_analysis.jpg")  # 保存图片
    plt.show()

if __name__ == "__main__":
    # 数据加载与处理
    df = load_data(".\output\combined_data.csv")

    # 特征工程
    X, y = feature_engineering(df)

    # 模型训练
    trained_model = train_model(X, y)

    # 可视化分析
    visualize_analysis(trained_model, X,y)

    print("分析完成！结果文件已保存：feature_importance.jpg, shap_analysis.jpg, pdp_analysis.jpg")