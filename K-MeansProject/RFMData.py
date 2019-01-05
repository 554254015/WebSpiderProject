# -*- coding: utf-8 -*-
'''
基于RFM模型使用K-Means算法聚类航空消费行为特征数据
'''
import pandas as pd

## 第一步：数据清洗
# 读取数据
data = pd.read_excel(r'F:\PythonTest\BigData\i_nuc.xls', index_col = 'Id', sheetname = 'Sheet2')
# 保存结果的文件名
outputfile = r'F:\PythonTest\BigData\2018book\data_type.xls'
# 聚类的类别
k = 3
# 聚类最大循环次数
iteration = 500


## 第二步：标准化处理
# 标准化后的数据存储路径文件
zscoredfile = r'F:\PythonTest\BigData\2018book\zscoreddata.xls'
# 数据标准化
data_zs = 1.0 * (data - data.mean())/data.std()
# 数据写入，备用
data_zs.to_excel(zscoredfile, index = False)


## 第三步：使用K-Means算法聚类消费行为特征数据，并导出各自类别的概率密度图
from sklearn.cluster import KMeans
# 分为k类，并发数4
model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration)
# 开始聚类
model.fit(data_zs)

# 简单打印结果
# 统计各个类别的数目
r1 = pd.Series(model.labels_).value_counts()
# 找到聚类中心
r2 = pd.DataFrame(model.cluster_centers_)
# 横向连接（0是纵向），得到聚类中心对应类别下的数目
r = pd.concat([r2, r1], axis = 1)
# 重命名表头
r.columns = list(data.columns) + [u'类别数目']
print(r)

# 详细输出原始数据及其类别
r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)
# 详细输出每个样本对应的类别
# 重命名表头
r.columns = list(data.columns) + [u'聚类类别']
# 保存结果
r.to_excel(outputfile)

# 自定义作图函数
def density_plot(data):
    import matplotlib.pyplot as plt
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    #用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    p = data.plot(kind = 'kde', linewidth = 2, subplots = True, sharex = False)
    [p[i].set_ylabel(u'密度') for i in range(k)]
    plt.xlabel('分群%s'%(i + 1))
    plt.legend()
    plt.show()
    return plt

# 概率密度图文件名前缀
pic_output = r'F:\PythonTest\BigData\2018book\pd_'
for i in range(k):
    density_plot(data[r[u'聚类类别'] == i]).savefig(u'%s%s.png'%(pic_output, i))
