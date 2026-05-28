import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as dct
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn import datasets
#todo:加载数据集
iris = datasets.load_iris()
# print(iris.keys())
# 'data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'
#todo:划分特征和标签
x = pd.DataFrame(iris.data,columns=iris.feature_names)
# x.head()
y = pd.DataFrame(iris.target,columns=['target'])
# print(y.head())
#todo:划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=12)
for i in [x_train,x_test,y_train,y_test]:
    i.reset_index(inplace=True,drop=True)
#todo:创建模型
tree_clf = dct()
tree_clf = tree_clf.fit(x_train,y_train)
train_score = tree_clf.score(x_train,y_train)
test_score = tree_clf.score(x_test,y_test)
print("训练集准确率：",train_score)
print("测试集准确率：",test_score)

#todo:绘制决策树
# plt.figure(figsize=(15,15))
# tree.plot_tree(tree_clf,
#                node_ids=True,
#                filled=True,
#                fontsize=12,
#                )
# plt.show()

y_pred = tree_clf.predict(x_test)
print("\n===== 分类报告 =====")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=iris.target_names,
#             yticklabels=iris.target_names)
# plt.xlabel("预测标签")
# plt.ylabel("真实标签")
# plt.title("决策树 混淆矩阵")
# plt.show()

overall_acc = tree_clf.score(x_test, y_test)
print(f"模型整体准确率：{overall_acc:.2%}")