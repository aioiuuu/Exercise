import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 替换为随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. 预测与评估
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("===== 随机森林回归 模型评估 =====")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 可视化对比
plt.figure(figsize=(10, 6))
n_samples = 200
plt.plot(y_test[:n_samples], label='真实房价', color='red', linewidth=1)
plt.plot(y_pred[:n_samples], label='随机森林预测房价', color='green', linestyle='--', linewidth=1)
plt.xlabel("样本序号")
plt.ylabel("房价（单位：10万美元）")
plt.title("加州房价预测：随机森林回归模型 真实值 vs 预测值")
plt.legend()
plt.show()

# MAE: 0.3323
# MSE: 0.2565
# RMSE: 0.5065
# R²: 0.8046
