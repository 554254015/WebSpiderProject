# jupyter下加载所需要包
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import jieba
from pandas import read_csv
from wordcloud import WordCloud,ImageColorGenerator
from ggplot import *
from pylab import *
from pandas.tslib import Timestamp
from pandas.lib import Timestamp
from pandas.core import datetools, generic
from pandas import DataFrame
from scipy.ndimage import imread
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import pdist,squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.cluster.util import cosine_distance
from nltk.cluster.kmeans import KMeansClusterer
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# 设置字体
font = FontProperties(fname = 'C:/Windows/Fonts/simsun.ttc', size = 14)
# 设置pandas显示方式
pd.set_option("display.max_rows", 8)
pd.options.mode.chained_assignment = None    # default = 'warn'
# 在jupyter中设置显示图像的方式
%matplotlib inline
%config InlineBackend.figure_format = "retina"

"""
# 读取数据
try:
    with open(r'F:\PythonTest\BigData\2018book\my_stop_words.txt', 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    for line in lines:
        print(line)
except UnicodeDecodeError as e:
    with open(r'F:\PythonTest\BigData\2018book\my_stop_words.txt', 'r', encoding = 'gbk') as f:
        lines = f.readlines()
    for line in lines:
        print(line)
"""        
# 读取停用词和需要的词典
stopword = read_csv(r'F:\PythonTest\BigData\2018book\my_stop_words.txt', header = None, names = ["Stopwords"])
mydict = read_csv(r'F:\PythonTest\BigData\2018book\red_dictionary.txt', header = None, names = ["Dictionary"])
print(stopword)
print("----------------------------------------------")
print(mydict)

RedDream = read_csv(r'F:\PythonTest\BigData\2018book\red_UTF82.txt', header = None, names = ["Reddream"])
print(RedDream)


# 数据预处理
# 查看数据是否存在空白的行，如有则删除
np.sum(pd.isnull(RedDream))

# 删除卷数据，使用正则表达式
# 包含相应关键字的索引
indexjuan = RedDream.Reddream.str.contains("^第+.+卷")
# 删除不需要的段，并重新设置索引，~：取反操作
RedDream = RedDream[~indexjuan].reset_index(drop = True)
print(RedDream)

# 找出每一章节的头部索引和尾部索引
# 每一章节的标题
indexhui = RedDream.Reddream.str.contains("^第+.+回")
chapnames = RedDream.Reddream[indexhui].reset_index(drop = True)
print(chapnames)
print("--------------------------------------------")
# 处理章节名，按照空格分隔字符串
chapnamesplit = chapnames.str.split(" ")
print(chapnamesplit)

# 建立保存数据的数据表
Red_df = pd.DataFrame(list(chapnamesplit),
                     columns = ["Chapter", "Leftname", "Rightname"])
#print(Red_df)


# 添加新的变量
Red_df["Chapter2"] = np.arange(1, 121)
Red_df["ChapName"] = Red_df.Leftname + "," + Red_df.Rightname
# 每章的开始行（段）索引
Red_df["StartCid"] = indexhui[indexhui == True].index
# 每章的结束行数
Red_df["EndCid"] = Red_df["StartCid"][1:len(Red_df["StartCid"])].reset_index(drop = True) - 1
Red_df["EndCid"][len(Red_df["EndCid"]) - 1] = RedDream.index[-1]
# 每章的段落长度
Red_df["Lengthchaps"] = Red_df.EndCid - Red_df.StartCid
Red_df["Artical"] = "Artical"
Red_df

# 每章节的内容
for ii in Red_df.index:
    # 将内容使用""连接
    chapid = np.arange(Red_df.StartCid[ii] + 1, int(Red_df.EndCid[ii]))
    # 每章节的内容替换掉空格
    Red_df["Artical"][ii] = "".join(list(RedDream.Reddream[chapid])).replace("\u3000", "")
# 计算某章有多少个字
Red_df["lenzi"] = Red_df.Artical.apply(len)
Red_df
    

# 散点图分析
#mpl.rcParams['font.sans-serif'] = ['SimHei']   # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

# 字长和段落数的散点图一
plt.figure(figsize = (8, 6))
plt.scatter(Red_df.Lengthchaps, Red_df.lenzi)
for ii in Red_df.index:
    plt.text(Red_df.Lengthchaps[ii] + 1, Red_df.lenzi[ii], Red_df.Chapter2[ii])
plt.xlabel("章节段数")
plt.ylabel("章节字数")
plt.title("《红楼梦》120回")
plt.show()

# 字长和段落数的散点图二
plt.figure(figsize = (8, 6))
plt.scatter(Red_df.Lengthchaps, Red_df.lenzi)
for ii in Red_df.index:
    plt.text(Red_df.Lengthchaps[ii] - 2, Red_df.lenzi[ii] + 100, Red_df.Chapter[ii], size = 7)
plt.xlabel("章节段数")
plt.ylabel("章节字数")
plt.title("《红楼梦》120回")
plt.show()

# 章节段数、字数与情节发展趋势
plt.figure(figsize = (12, 10))
plt.subplot(2, 1, 1)
plt.plot(Red_df.Chapter2, Red_df.Lengthchaps, "ro-", label = "段落")
plt.ylabel("章节段数", Fontproperties = font)
plt.title("《红楼梦》120回", Fontproperties = font)
# 添加平均值
plt.hlines(np.mean(Red_df.Lengthchaps), -5, 125, "g")
plt.xlim((-5, 125))
plt.subplot(2, 1, 2)
plt.plot(Red_df.Chapter2, Red_df.lenzi, "ro-", label = "段落")
plt.xlabel("章节", Fontproperties = font)
plt.ylabel("章节字数", Fontproperties = font)
# 添加平均值
plt.hlines(np.mean(Red_df.lenzi), -5, 125, "b")
plt.xlim((-5, 125))
plt.show()
