import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
%matplotlib inline

# 1.1 引入数据
cancer = load_breast_cancer()
# 1.2 拆分测试集、训练集
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)

# 未剪枝决策树模型
# tree = DecisionTreeClassifier(random_state=0)
# 预剪枝，指定最大深度max_depth
tree = DecisionTreeClassifier(max_depth=4, random_state=0)

# 训练模型
tree.fit(X_train,y_train)
# 模型评分
print('训练数据得分：{:.3f}'.format(tree.score(X_train,y_train)))
print('测试数据得分：{:.3f}'.format(tree.score(X_test,y_test)))

# 决策树可视化，生成了tree.dot文件
# export_graphviz(tree, out_file='tree.dot', class_names=['malignant','benign'],feature_names=cancer.feature_names,impurity=False, filled=True)


# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —

# 随机森林
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10, random_state=0) #n_estimators=5即设置为5个模型树

# 训练模型
forest.fit(X_train,y_train)

print('随机森林训练数据得分：{:.3f}'.format(forest.score(X_train,y_train)))
print('随机森林训练数据得分：{:.3f}'.format(forest.score(X_test,y_test)))
