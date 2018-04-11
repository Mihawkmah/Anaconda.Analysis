import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import train_test_split
%matplotlib inline

# 1.1 引入数据
cancer = load_breast_cancer()
# 1.2 拆分测试集、训练集
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)

# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —

# 线性支持向量机
svc = LinearSVC(C=0.1,random_state=0).fit(X_train,y_train) # 默认C=1,越大模型越灵活
print('线性训练数据得分：{:.3f}'.format(svc.score(X_train,y_train)))
print('线性测试数据得分：{:.3f}'.format(svc.score(X_test,y_test)))

# 引入核函数的支持向量机
svm = SVC(kernel='rbf', random_state=0, gamma=2, C=10) #gamma越大越容易导致过拟合
svm.fit(X_train,y_train)
print('引入核函数训练数据得分：{:.3f}'.format(svm.score(X_train,y_train)))
print('引入核函数测试数据得分：{:.3f}'.format(svm.score(X_test,y_test)))
