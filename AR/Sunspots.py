import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf
from sklearn.metrics import mean_squared_error, r2_score

data = sm.datasets.sunspots.load_pandas().data
ts = data['SUNACTIVITY']
year = data['YEAR']

#可视化原始序列
# plt.figure(figsize=(12,4))
# plt.plot(year,ts,color='#0000FF')
# plt.title('Sunspots Time Series (1700–2008)')
# plt.xlabel('Year')
# plt.ylabel('Sunspot Number')
# plt.show()

#划分训练集和测试集
train = ts[:250]
test = ts[250:]

#选AR阶数
# pacf_vals = pacf(train,nlags=15)
# plt.figure(figsize=(8,3))
# plt.bar(range(len(pacf_vals)), pacf_vals)
# plt.axhline(0,color='#0000FF',linestyle='--')
# plt.title('PACF-choose AR(p)')
# plt.show()

#todo:挑选最佳参数
# best_aic = np.inf
# best_p = 1
# for p in range(1,16):
#     model = AutoReg(train,lags=p,trend='c').fit()
#     if model.aic < best_aic:
#         best_aic = model.aic
#         best_p = p
# print(f'最优阶数p={best_p}')
# p=15

#训练模型
p=13
model = AutoReg(train,lags=p,trend='c').fit()
print(model.summary())

#预测
pred = model.predict(start=len(train),
                     end=len(train)+len(test)-1,
                     dynamic=False)

#评估
rmse  = np.sqrt(mean_squared_error(test,pred))
print('RMSE:',rmse)
r2 = r2_score(test,pred)
print('决定系数 R2=',r2)
# RMSE: 38.58441711996442
# 决定系数 R2= 0.44544634220635027
