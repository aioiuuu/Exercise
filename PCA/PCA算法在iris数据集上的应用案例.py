import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition,datasets
iris = datasets.load_iris()
x = iris['data']
#todo:初始化PCA模型参数
model = decomposition.PCA(n_components=2)
#todo:训练PCA模型
model.fit(x)
x_new = model.fit_transform(x)
Maxcomponent = model.components_
ratio = model.explained_variance_ratio_
score = model.score(x)
print('降维后的数据：',x_new)
print('返回后具有最大方差的成分：',Maxcomponent)
print('保留主成分的方差贡献率：',ratio)

g1 = plt.figure(1,figsize=(8,6))
plt.scatter(x_new[:,0],x_new[:,1],c='r',cmap=plt.cm.Set1,edgecolors='k',s=40)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('After the dimension reduction')
plt.show()