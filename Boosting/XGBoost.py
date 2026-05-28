from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#加载加州房价数据集
data = fetch_california_housing()
x,y  = data.data,data.target
feature_names = data.feature_names

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
import time

# 随机森林模型
start = time.time()
rf = RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1)
rf.fit(x_train,y_train)
rf_time = time.time() - start
rf_pred = rf.predict(x_test)

# XGBoost模型
start = time.time()
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(x_train,y_train)
xgb_time = time.time() - start
xgb_pred = xgb.predict(x_test)

#--结果评测--
print(f"{'指标':<15}{'随机森林':>12}{'XGBoost':>12}")
print('*'*50)
print(f"{'MSE':<15}{mean_squared_error(y_test,rf_pred):>12.4f} {mean_squared_error(y_test,xgb_pred):.4f}")
print(f"{'R^2':<15}{r2_score(y_test,rf_pred):>12.4f} {r2_score(y_test,xgb_pred):12.4f}")
print(f"{'训练时间':<15}{rf_time:.4f} {xgb_time:.4f}")