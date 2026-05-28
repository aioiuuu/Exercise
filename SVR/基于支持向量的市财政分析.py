import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

data = pd.read_csv('../data/data.csv')

np.round(data.corr(method = 'pearson'),2).to_csv('../tmp/data_cor.csv')
print('相关系数矩阵为：\n',np.round(data.corr(method='pearson'),2))

lasso = Lasso(1000)
lasso.fit(data.iloc[:,0:13],data['y'])
print('相关系数为：',np.round(lasso.coef_,5))

print('相关系数非0的个数为：',np.sum(lasso.coef_!=0))

mask = lasso.coef_!=0
print('相关系数是否为0：',mask)

data_filtered = data.iloc[:,0:13]
new_reg_data = data_filtered.iloc[:,mask]
new_reg_data.to_csv('../tmp/new_reg_data.csv')
print('输出数据的维度为：',new_reg_data.shape)

def GM11(x0):
    x1 = x0.cumsum()
    z1 = (x1[:len(x1)-1]+x1[1:])/2.0
    z1 = z1.reshape((len(z1),1))
    B = np.append(-z1,np.ones_like(z1),1)
    yn = x0[1:].reshape((len(x0)-1,1))

    [[a],[b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), yn)
    f = lambda k:(x0[0]-b/a)*np.exp(-a*(k-1)) - (x0[0])
    delta = np.abs(x0-np.array([f(i) for i in range(1,len(x0)+1)]))
    C = delta.std()/x0.std()
    p = 1.0*(np.abs(delta - delta.mean())<0.6745*x0.std()).sum()/len(x0)

    return f,a,b,x0[0],C,p


import pandas as pd
import numpy as np





# 读取数据
new_reg_data = pd.read_csv('../tmp/new_reg_data.csv', index_col=0)
data = pd.read_csv('../data/data.csv')

# 设置年份索引
new_reg_data.index = list(range(1994, 2014))
new_reg_data.loc[2014] = None
new_reg_data.loc[2015] = None

Accuracy = []
features = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']

for col in features:
    # 提取当前特征的原始序列（1994-2013）
    raw_series = new_reg_data.loc[list(range(1994, 2014)), col].values

    # 对当前特征做GM(1,1)建模，获取预测函数、C值、P值
    gm_result = GM11(raw_series)
    f = gm_result[0]
    C = gm_result[4]
    P = gm_result[5]

    # 预测2014、2015年的值
    n = len(raw_series)
    new_reg_data.loc[2014, col] = f(n + 1)
    new_reg_data.loc[2015, col] = f(n + 2)
    new_reg_data[col] = new_reg_data[col].round(2)

    # 模型精度评级
    if P > 0.95 and C < 0.35:
        Accuracy.append('好')
    elif 0.8 <= P <= 0.95 and 0.35 <= C < 0.5:
        Accuracy.append('合格')
    elif 0.7 <= P <= 0.8 and 0.5 <= C < 0.65:
        Accuracy.append('勉强合格')
    else:
        Accuracy.append('不合格')

# 结果整合 - 不要删除第一列，注释掉这行
# new_reg_data = new_reg_data.iloc[:, 1:]
new_reg_data.loc['模型精度'] = Accuracy

# 合并y列数据
y = list(data['y'].values)
y.extend([np.nan, np.nan])
new_reg_data.loc[list(range(1994, 2016)), 'y'] = y

# 保存结果
outputfile = '../tmp/new_reg_data_GM11.xlsx'  # 推荐用xlsx格式
new_reg_data.to_excel(outputfile, index=True)

# 打印预测结果
print('预测结果为：\n', new_reg_data.loc[[2014, 2015, '模型精度'], :])

import pandas as pd
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取数据
data = pd.read_excel('../tmp/new_reg_data_GM11.xlsx')

# 2. 强制把所有数据转成数字，无法转换的变成 NaN（关键！）
data = data.apply(pd.to_numeric, errors='coerce')

# 3. 去掉全是空/无效的行
data = data.dropna(how='all')

# 4. 只保留纯数字行（彻底去掉“模型精度”等文字）
data = data[data.select_dtypes(include=[np.number]).all(axis=1)]

feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']

# 5. 取前20行作为训练集（1994-2013）
data_train = data.head(20).copy()

# 6. 标准化
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean) / data_std

# 7. 训练模型
x_train = data_train[feature].values
y_train = data_train['y'].values

linearsvr = LinearSVR(random_state=123, max_iter=100000)
linearsvr.fit(x_train, y_train)

# 8. 预测
x = ((data[feature] - data_mean[feature]) / data_std[feature]).values
data['y_pred'] = linearsvr.predict(x) * data_std['y'] + data_mean['y']

# 9. 保存结果
data.to_excel('../tmp/new_reg_data_GM11_revenue.xlsx', index=False)

# 10. 输出 + 画图
print('真实值 vs 预测值：')
print(data[['y', 'y_pred']])

plt.figure(figsize=(10, 6))
plt.plot(data['y'], 'b-o', label='真实值')
plt.plot(data['y_pred'], 'r-*', label='预测值')
plt.xlabel('样本序号')
plt.ylabel('财政收入')
plt.legend()
plt.title('SVR 财政收入预测')
plt.show()