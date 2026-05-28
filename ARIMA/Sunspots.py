import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 加载数据 =====================
data = sm.datasets.sunspots.load_pandas().data
ts = data['SUNACTIVITY']
train = ts[:250]
test = ts[250:]

# ===================== 2. 自动搜索最优 ARIMA(p,d,q) =====================
best_aic = np.inf
best_order = None
best_model = None

# 搜索范围：p=0~12，d=0~1，q=0~3
for p in range(0, 13):
    for d in range(0, 2):
        for q in range(0, 4):
            try:
                model = ARIMA(train, order=(p, d, q)).fit()
                # 用 AIC 选择最优模型
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_model = model
            except:
                continue

# ===================== 3. 最优模型预测 =====================
pred = best_model.predict(start=len(train), end=len(train)+len(test)-1)

# ===================== 4. 计算评估指标 =====================
# 1. 误差指标
rmse = np.sqrt(mean_squared_error(test, pred))
mae = mean_absolute_error(test, pred)

# 2. 模型信息准则
aic = best_model.aic
bic = best_model.bic

# 3. Ljung-Box 检验（残差是否为白噪声）
# p-value > 0.05 → 残差无自相关 → 模型合格
lb_test = acorr_ljungbox(best_model.resid, lags=10, return_df=True)
lb_pvalues = lb_test['lb_pvalue'].values

# ===================== 5. 输出全部结果 =====================
print("="*60)
print("          最优 ARIMA 模型结果（太阳黑子）")
print("="*60)
print(f"最优阶数 (p,d,q)   : {best_order}")
print(f"AIC               : {aic:.2f}")
print(f"BIC               : {bic:.2f}")
print(f"RMSE              : {rmse:.4f}")
print(f"MAE               : {mae:.4f}")
print("\nLjung-Box 检验 p值 (滞后1-10阶):")
print(np.round(lb_pvalues, 4))
print("\n若所有 p-value > 0.05 → 模型残差为白噪声，模型构建成功！")
print("="*60)
# AIC               : 2041.53
# BIC               : 2083.74
# RMSE              : 39.6859
# MAE               : 33.7735