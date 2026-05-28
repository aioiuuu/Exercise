import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

#生成带噪声的正弦曲线数据集
np.random.seed(42)
x = np.sort(np.random.rand(200,1)*10,axis=0)
y = np.sin(x).ravel() + np.random.randn(200)*0.2

n_trees = 25
learning_rate = 0.3
trees = []
F  = np.full_like(y,y.mean())

fig,axes = plt.subplots(5,5,figsize=(16,9))

for i in range(n_trees):
    residual = y - F

    #用一颗树来预测残差
    tree = DecisionTreeRegressor(max_depth=2).fit(x,residual)
    pred = tree.predict(x)
    trees.append(tree)

    #更新模型
    F += learning_rate * pred

    #可视化
    mse = np.mean((y - F)**2)
    ax = axes.flat[i]
    ax.scatter(x,y,s=10,alpha=0.4,color='steelblue',label='data')
    ax.plot(x,F,color='red',label='F')
    ax.set_title(f'第{i+1}棵树后 | MSE={mse:.2f}',fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle('手搓GBDT:逐步逼近真实函数',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.show()

