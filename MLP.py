from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
%matplotlib inline

# 引入数据
X,y = make_moons(n_samples=100, noise=0.25, random_state=3)
# 分为训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# 训练得到模型，solver指定算法为lbfgs，hidden_layer_sizes指定2层，每层10个神经元。使用tanh激活函数
mlp = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train, y_train)
