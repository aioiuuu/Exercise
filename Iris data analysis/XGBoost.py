# 导入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# 解决中文显示
# ======================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 划分训练集测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 创建 XGBoost 模型
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
xgb_model.fit(Xtrain, Ytrain)

# 4. 预测
y_pred = xgb_model.predict(Xtest)

# ======================
# 5. 准确率
# ======================
train_acc = xgb_model.score(Xtrain, Ytrain)
test_acc = xgb_model.score(Xtest, Ytest)

print("===== XGBoost 模型评估 =====")
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")
print(f"整体准确率: {accuracy_score(Ytest, y_pred):.4f}")

# ======================
# 6. 5折交叉验证
# ======================
scores = cross_val_score(xgb_model, X, y, cv=5)
print("\n===== 5折交叉验证 =====")
print(f"5次分数: {scores}")
print(f"平均分数: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")

# ======================
# 7. 分类报告
# ======================
print("\n===== 分类报告 =====")
print(classification_report(Ytest, y_pred, target_names=iris.target_names))

# ======================
# 8. 混淆矩阵
# ======================
cm = confusion_matrix(Ytest, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("XGBoost 混淆矩阵")
plt.show()