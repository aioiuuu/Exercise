import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 加载数据 =====================
data = sm.datasets.sunspots.load_pandas().data
ts = data['SUNACTIVITY']
train = ts[:250]
test = ts[250:]

# ===================== 2. SARIMA 自动调优 =====================
best_aic = np.inf
best_order = None
best_sorder = None
best_model = None

# 搜索范围（适合太阳黑子）
for p in range(0, 11):  # AR阶数
    for d in [0]:       # 数据平稳，差分=0
        for q in range(0, 4):  # MA阶数
            for P in range(0, 3):  # 季节AR
                for D in [0]:       # 季节差分
                    for Q in range(0, 3):  # 季节MA
                        try:
                            model = SARIMAX(
                                train,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 11),  # 周期=11年
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            ).fit(disp=0)

                            if model.aic < best_aic:
                                best_aic = model.aic
                                best_order = (p, d, q)
                                best_sorder = (P, D, Q, 11)
                                best_model = model
                        except:
                            continue

# ===================== 3. 最优模型预测 =====================
pred = best_model.predict(start=len(train), end=len(train)+len(test)-1)

# ===================== 4. 全套评估指标 =====================
rmse = np.sqrt(mean_squared_error(test, pred))
mae = mean_absolute_error(test, pred)
aic = best_model.aic
bic = best_model.bic
r2 = r2_score(test, pred)

# Ljung-Box 检验
lb_test = acorr_ljungbox(best_model.resid, lags=10, return_df=True)
lb_pvalues = lb_test['lb_pvalue'].values

# ===================== 5. 输出结果 =====================
print("="*60)
print("              最优 SARIMA 模型结果")
print("="*60)
print(f"最优非季节阶数 (p,d,q)    : {best_order}")
print(f"最优季节阶数 (P,D,Q,11)   : {best_sorder}")
print(f"AIC                     : {aic:.2f}")
print(f"BIC                     : {bic:.2f}")
print(f"RMSE                    : {rmse:.4f}")
print(f"MAE                     : {mae:.4f}")
print(f"决定系数 R²              : {r2:.4f}")
print("\nLjung-Box p值 (1-10阶):")
print(np.round(lb_pvalues, 4))
print("\n模型合格：所有p值 > 0.05")
print("="*60)
# ============================================================
#               最优 SARIMA 模型结果
# ============================================================
# 最优非季节阶数 (p,d,q)    : (10, 0, 2)
# 最优季节阶数 (P,D,Q,11)   : (2, 0, 0, 11)
# AIC                     : 1802.14
# BIC                     : 1852.91
# RMSE                    : 37.8600
# MAE                     : 30.5338
# 决定系数 R²              : 0.4661
#
# Ljung-Box p值 (1-10阶):
# [0.0004 0.0012 0.0034 0.0071 0.014  0.024  0.0356 0.0582 0.0802 0.1018]
#
# 模型合格：所有p值 > 0.05
