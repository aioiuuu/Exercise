# ======================
# 1. 导入库
# ======================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 加载数据
# ======================
housing = fetch_california_housing()
X = housing.data
y = housing.target

# ======================
# 3. 划分训练集 / 测试集
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================
# 4. XGBoost 回归模型
# ======================
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# 训练
xgb_model.fit(X_train, y_train)

# 预测
y_pred = xgb_model.predict(X_test)

# ======================
# 5. 模型评估
# ======================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("===== XGBoost 回归评估 =====")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ======================
# 6. 可视化对比
# ======================
plt.figure(figsize=(10, 6))
plt.plot(y_test[:200], label="真实房价", color="red")
plt.plot(y_pred[:200], label="XGBoost 预测", color="green")
plt.title("加州房价预测 - XGBoost")
plt.xlabel("样本")
plt.ylabel("房价（10万美元）")
plt.legend()
plt.show()

# MAE  : 0.3090
# MSE  : 0.2155
# RMSE : 0.4642
# R²   : 0.8358
