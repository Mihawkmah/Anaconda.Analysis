import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.datasets import load_boston
%matplotlib inline

# 引入数据
boston = load_boston()

# 1.1 单变量线性回归
# 将数据转化为DataFrame
# df=pd.DataFrame(boston.data,columns=boston.feature_names)
# df['MEDV']=boston.target

# 定义X, y
# X = df[['RM']].values
# y = df['MEDV'].values

# 数据探索，观察图形注意到RM和MEDV有明显线性关系
# cols=['LSTAT','INDUS','NOX','RM','MEDV']
# sns.pairplot(df[cols],size=2.5)


# 1.2 多变量线性回归
# 定义X, y
# X,y = boston.data,boston.target


# 1.3 多项式回归
y = boston.target
# 通过MinMaxScaler标准化
X = MinMaxScaler().fit_transform(boston.data)
# 利用多项式回归，degree=2，得到新的X
X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)


# — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —

# 将数据分为训练集、测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# 使用训练数据对模型进行训练，得到lr
lr = LinearRegression().fit(X_train,y_train)
# 模型正则化 Ridge回归，alpha的值越大就越严格，值越小越灵活
# ridge10 = Ridge(alpha=10).fit(X_train,y_train)
# 模型正则化 Lasso回归，alpha的值越大使用的特征数量越少，值越小惩罚越轻特征越多模型越复杂
# lasso01 = Lasso(alpha=0.1).fit(X_train,y_train)
# 打印出Lasso使用了几个特征
# print(format(np.sum(lasso01.coef_!=0)))

# 通过模型lr函数做预测
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# 通过图形直观看模型质量
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='red')

# 通过模型得分看模型质量
print('训练数据得分：{:.3f}'.format(lr.score(X_train,y_train)))
print('测试数据得分：{:.3f}'.format(lr.score(X_test,y_test)))
