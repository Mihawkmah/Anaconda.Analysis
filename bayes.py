import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

# 导入康奈尔大学网站的2M影评数据集，tokens下的文件夹neg、pos被识别是target
movie_reviews = load_files('./tokens')
movie_data = movie_reviews.data
movie_target = movie_reviews.target

# 有多少条样本数据
len(movie_data)

# 将文本型的数据转化为TF-IDF矩阵
count_vec = TfidfVectorizer(binary = False, decode_error='ignore', stop_words='english')
# 分为训练集、测试集
x_train,x_test,y_train,y_test = train_test_split(movie_data, movie_target, test_size=0.2)
# 将训练数据转化为词频
x_train = count_vec.fit_transform(x_train)
# 利用上一步得到的count_vec，将测试数据也转化为词频
x_test= count_vec.transform(x_test)

# MultinomialNB，这个分类器以出现次数作为特征值，我们使用的TF-IDF也能符合这类分布。GaussianNB适用于高斯分布（正态分布）的特征，而BernoulliNB适用于伯努利分布（二值分布）的特征
clf = MultinomialNB().fit(x_train,y_train)
doc_class_predicted = clf.predict(x_test)

# 查看有多少比率预测正确了
print(np.mean(doc_class_predicted == y_test))

# 准确率与召回率(模型评估)
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
answer = clf.predict_proba(x_test)[:,1]
report = answer > 0.5
print(classification_report(y_test, report, target_names = ['neg','pos']))
