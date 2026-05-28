# ======================
# 1. 导入库
# ======================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 2. 加载加州房价数据
# ======================
housing = fetch_california_housing()
X = housing.data
y = housing.target

# ======================
# 3. 划分数据集
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ======================
# 4. XGBoost + 网格搜索（自动找最优参数）
# ======================

# 定义模型
xgb = XGBRegressor(random_state=42)

# 定义要搜索的参数组合
param_grid = {
    'n_estimators': [100, 200],       # 树数量
    'max_depth': [3, 5, 7],           # 树深度
    'learning_rate': [0.05, 0.1]      # 学习率
}

# 网格搜索（5折交叉验证）
grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,            # 5折交叉验证
    n_jobs=-1,       # 用满CPU加速
    verbose=1        # 显示进度
)

# 开始训练 + 自动调参
grid.fit(X_train, y_train)

# ======================
# 5. 输出最优结果
# ======================
print("\n===== 网格搜索最优结果 =====")
print("最优参数：", grid.best_params_)
print("最优交叉验证分数：", grid.best_score_)

# 最优模型
best_model = grid.best_estimator_

# ======================
# 6. 在测试集上预测
# ======================
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== XGBoost 最优模型测试集评估 =====")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ======================
# 7. 画图对比
# ======================
plt.figure(figsize=(10, 6))
plt.plot(y_test[:200], label='真实房价', color='red')
plt.plot(y_pred[:200], label='XGBoost最优模型预测', color='green')
plt.title('加州房价预测 - XGBoost 网格搜索优化版')
plt.xlabel('样本序号')
plt.ylabel('房价（单位：10万美元）')
plt.legend()
plt.show()
# MAE  : 0.3005
# MSE  : 0.2096
# RMSE : 0.4578
# R²   : 0.8403