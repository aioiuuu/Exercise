# 导入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 划分训练集、测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 构建 MLP 神经网络模型
mlp_model = MLPClassifier(
    hidden_layer_sizes=(10, 5),  # 2层隐藏层：第一层10个神经元，第二层5个
    max_iter=1000,               # 最大迭代次数，保证收敛
    random_state=42,
    activation='relu'           # 激活函数
)
mlp_model.fit(Xtrain, Ytrain)

# 4. 预测
y_pred = mlp_model.predict(Xtest)

# 5. 各类准确率
train_acc = mlp_model.score(Xtrain, Ytrain)
test_acc = mlp_model.score(Xtest, Ytest)

print("===== MLP 多层感知器 模型评估 =====")
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"整体准确率: {accuracy_score(Ytest, y_pred):.4f}")

# 6. 5折交叉验证
scores = cross_val_score(mlp_model, X, y, cv=5)
print("\n===== 5折交叉验证 =====")
print(f"5次验证分数: {scores}")
print(f"平均准确率: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")

# 7. 分类报告
print("\n===== 分类报告 =====")
print(classification_report(Ytest, y_pred, target_names=iris.target_names))

# 8. 混淆矩阵绘图
cm = confusion_matrix(Ytest, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("MLP 神经网络 混淆矩阵")
plt.show()