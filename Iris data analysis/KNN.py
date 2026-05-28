import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#获取数据
iris = load_iris()
data = pd.DataFrame(data=iris.data,columns=iris.feature_names)
data['species']=iris.target_names[iris.target]

#描述性统计
# print(data.head(10))
# data.info()
# data.describe()

#数据可视化
# sns.pairplot(data,hue='species')
# plt.show()

#统计每个类别的数量
# category_count = data['species'].value_counts()
# print(category_count)
# setosa        50
# versicolor    50
# virginica     50

#分析数据之间的相关性
# corr = data.drop(columns=['species']).corr(method='pearson')
# plt.figure(figsize=(10,8))
# sns.heatmap(corr,annot=True)
# plt.show()

#标准化
from sklearn.preprocessing import StandardScaler
x = data.iloc[:,0:4]
y = data.iloc[:,4]
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

#数据拆分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=22
)

#创建KNN分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

knn = KNeighborsClassifier(n_neighbors=3,metric='euclidean')

#训练模型
knn.fit(x_train,y_train)
#预测测试集
y_pred = knn.predict(x_test)
#计算评估指标
# report = classification_report(y_test,y_pred,target_names=iris.target_names)
# print(f"{report}")
#计算混淆矩阵
# confusion_matrix = confusion_matrix(y_test,y_pred)
#绘制混淆矩阵
# sns.heatmap(confusion_matrix,annot=True,cmap="Blues",fmt="d",xticklabels=iris.feature_names,yticklabels=iris.target_names)
# plt.xlabel("predicted Label")
# plt.ylabel("True Label")
# plt.title("KNN:Confusion Matrix")
# plt.show()

from sklearn.metrics import accuracy_score
print("准确率：", accuracy_score(y_test, y_pred))

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, knn.predict_proba(x_test), multi_class="ovr")
print("AUC：", auc)

from sklearn.metrics import f1_score
print("F1：", f1_score(y_test, y_pred, average="macro"))

from sklearn.model_selection import LearningCurveDisplay
LearningCurveDisplay.from_estimator(knn, x, y)
plt.show()