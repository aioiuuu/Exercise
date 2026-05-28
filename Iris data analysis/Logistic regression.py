from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data
y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#构建逻辑回顾模型
lr_model = LogisticRegression(max_iter=200)   # max_iter加大，保证收敛
lr_model.fit(Xtrain, Ytrain)

# 4. 预测
y_pred = lr_model.predict(Xtest)

# 5. 计算准确率
train_acc = lr_model.score(Xtrain, Ytrain)
test_acc = lr_model.score(Xtest, Ytest)
overall_acc = accuracy_score(Ytest, y_pred)

print("===== 逻辑回归 模型评估 =====")
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"整体准确率: {overall_acc:.4f}")

# 6. 分类报告
print("\n===== 分类报告 =====")
print(classification_report(Ytest, y_pred, target_names=iris.target_names))

# 7. 混淆矩阵绘图
# cm = confusion_matrix(Ytest, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=iris.target_names,
#             yticklabels=iris.target_names)
# plt.xlabel("预测标签")
# plt.ylabel("真实标签")
# plt.title("逻辑回归 混淆矩阵")
# plt.show()

from sklearn.model_selection import cross_val_score

# 用逻辑回归模型做 5 折交叉验证
scores = cross_val_score(lr_model, X, y, cv=5)

print("5次验证分数：", scores)
print("交叉验证平均分数：", scores.mean())
print("标准差：", scores.std())