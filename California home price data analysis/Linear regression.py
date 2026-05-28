import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 1. 加载加州房价数据集
# ======================
housing = fetch_california_housing()
X = housing.data
y = housing.target

print("数据集特征名：", housing.feature_names)
print("数据形状：", X.shape, y.shape)

# ======================
# 2. 划分训练集、测试集
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 打印模型系数（可选，了解每个特征对房价的影响）
print("\n线性回归模型系数：", lr_model.coef_)
print("模型截距：", lr_model.intercept_)

# ======================
# 4. 在测试集上预测
# ======================
y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== 线性回归 模型评估 =====")
print(f"平均绝对误差 MAE: {mae:.4f}")
print(f"均方误差 MSE: {mse:.4f}")
print(f"均方根误差 RMSE: {rmse:.4f}")
print(f"决定系数 R²: {r2:.4f}")

# ======================
# 6. 可视化：真实房价 vs 预测房价
# ======================
plt.figure(figsize=(10, 6))
# 取前200个样本画对比图，避免点太多看不清
n_samples = 200
plt.plot(y_test[:n_samples], label='真实房价', color='red', linewidth=1)
plt.plot(y_pred[:n_samples], label='线性回归预测房价', color='blue', linestyle='--', linewidth=1)
plt.xlabel("样本序号")
plt.ylabel("房价（单位：10万美元）")
plt.title("加州房价预测：线性回归模型 真实值 vs 预测值")
plt.legend()
plt.show()
# 平均绝对误差 MAE: 0.5272
# 均方误差 MSE: 0.5306
# 均方根误差 RMSE: 0.7284
# 决定系数 R²: 0.5958