from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 创建SVM分类器
svm_model = SVC(kernel='linear',random_state=42)
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

#计算评估指标
train_acc = svm_model.score(x_train, y_train)
test_acc = svm_model.score(x_test, y_test)
overall_acc = accuracy_score(y_test, y_pred)
print(f"训练集准确率：{train_acc:.2%}")
print(f"测试集准确率：{test_acc:.2%}")
print(f"模型整体准确率：{overall_acc:.2%}")

#分类报告
print("\n===== 分类报告 =====")
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

#混淆矩阵
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("SVM 混淆矩阵")
plt.show()