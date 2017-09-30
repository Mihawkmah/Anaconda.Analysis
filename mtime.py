# -*- coding: utf-8 -*-
%matplotlib inline
import pymongo
import pandas as pd
import numpy as np
from plotly import offline
import plotly.graph_objs as go

client = pymongo.MongoClient(host='localhost',port=27017)
db = client.mtime
movies = db.movietop100

# 从数据表中取出数据转化为DataFrame
arr = []
for x in movies.find():
    arr.append(x)
movietop100 = pd.DataFrame(arr)

# 清洗DataFrame数据
for row in movietop100.iterrows():
    row[1]['year'] = row[1]['year'].replace('下','2005')
    row[1]['pointnum'] = row[1]['pointnum'].replace('人评分','')
    if type(row[1]['point']) == list:
        if len(row[1]['point']) > 0:
            row[1]['point'] = row[1]['point'][0] + row[1]['point'][1]
        else:
            row[1]['point'] = '9.0'


# 将电影类别列表拆分成3个字段，然后堆叠，然后补全形成新表，计算每个类型出现了多少次
types = movietop100.type.apply(pd.Series)
types.columns = ['type1', 'type2', 'type3']
newtypes = pd.concat([movietop100[:], types[:]], axis=1)[['year', 'type1', 'type2', 'type3']]
datatypes = newtypes.set_index(['year']).stack().reset_index().groupby(0).agg('count')

# 电影年份分布柱状图
yeardata = movietop100.groupby('year').agg('count')
offline.init_notebook_mode()
data = [go.Bar(
    x = yeardata.index,
    y = yeardata._id,
)]
offline.iplot(data)

# 电影类型分布饼状图
data = [go.Pie(
    labels=datatypes.index,
    values=datatypes.year
)]
offline.iplot(data)

# 电影的评分年份分布散点图
data = [go.Scatter(
    x = movietop100.year,
    y = movietop100.point,
    mode = 'markers'
)]
offline.iplot(data)
