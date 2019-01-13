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
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

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


# 对红楼梦进行分词
# 加载包
import jieba
# 对《红楼梦》全文进行分词
# 数据表的行数
row, col = Red_df.shape
# 预定义列表
Red_df["cutword"] = "cutword"
for ii in np.arange(row):
    # 分词
    cutwords = list(jieba.cut(Red_df.Artical[ii], cut_all = True))
    # 去除长度为1的词
    cutwords = pd.Series(cutwords)[pd.Series(cutwords).apply(len) > 1]
    # 去除停用词
    cutwords = cutwords[~cutwords.isin(stopword)]
    Red_df.cutword[ii] = cutwords.values
# 查看最后一段的分词结果
print(cutwords)
print(cutwords.values)

# 查看全书的分词结果
Red_df.cutword


# 连接list
words = np.concatenate(Red_df.cutword)
# 统计词频
word_df = pd.DataFrame({"Word": words})
word_stat = word_df.groupby(by = ["Word"])["Word"].agg({"number": np.size})
word_stat = word_stat.reset_index().sort_values(by = "number", ascending = False)
word_stat["wordlen"] = word_stat.Word.apply(len)
print(word_stat)
# 去除长度大于5的词
print(np.where(word_stat.Word.apply(len) < 5))
word_stat = word_stat.loc[word_stat.Word.apply(len) < 5,:]
word_stat = word_stat.sort_values(by = "number", ascending = False)
word_stat


### 词云
from wordcloud import WordCloud

# 采用“/”将词分开的形式
# 连接全书的词
"/".join(np.concatenate(Red_df.cutword))
# width = 1800, height = 800, 设置图片的清晰程度
wlred = WordCloud(font_path = "C:/Windows/Fonts/simsun.ttc", 
                  margin = 5, width = 1800, height = 800
                 ).generate("/".join(np.concatenate(Red_df.cutword)))
plt.imshow(wlred)
plt.axis("off")
plt.show()


# 采用指定{词语：频率}字典的形式
# 数据准备
worddict = {}
# 构造{词语：频率}字典
for key, value in zip(word_stat.Word, word_stat.number):
    worddict[key] = value
# 生成词云
# 查看其中的10个元素
for ii, myword in zip(range(10), worddict.items()):
    print(ii, end=" ")
    print(myword)

redcold = WordCloud(font_path = "C:/Windows/Fonts/simsun.ttc",
                   margin = 5, width = 1800, height = 1800)
redcold.generate_from_frequencies(frequencies = worddict)

plt.figure(figsize = (10, 10))
plt.imshow(redcold)
plt.axis("off")
plt.show()


# 利用图片生成有背景图片的词云
from scipy.ndimage import imread
back_image = imread(r'F:\PythonTest\BigData\2018book\带土n.jpg')
# 生成词云可以用计算好的词频，再使用generate_from_frequencies函数
red_wc = WordCloud(font_path = "C:/Windows/Fonts/simsun.ttc",   # 设置字体
                  margin = 5, width = 1800, height = 1800,      # 设置字体的清晰度
                  background_color = "black",                   # 设置背景颜色
                  max_words = 2000,                             # 设置词云显示的最大词数
                  mask = back_image,                            # 设置背景图片
                  #max_font_size = 100,                         # 设置字体最大值
                  random_state = 42
                  ).generate("/".join(np.concatenate(Red_df.cutword)))
# 从背景图生成颜色值
image_colors = ImageColorGenerator(back_image)
plt.figure()
plt.imshow(red_wc.recolor(color_func = image_colors))
plt.axis("off")
plt.show()


##绘制词语出现词数（频数）的直方图
# 筛选数据
newdata = word_stat.loc[word_stat.number > 500]
# 绘制频数大于500次的词语的直方图
newdata.plot(kind = "bar", x = "Word", y = "number", figsize = (10, 7))
plt.xticks(FontProperties = font, size = 10)     # 设置x轴刻度上的文本
plt.xlabel("关键词", FontProperties = font)      # 设置x轴上的标签
plt.ylabel("频数", FontProperties = font)
plt.title("《红楼梦》", FontProperties = font)
plt.show()

# 筛选数据
newdata = word_stat.loc[word_stat.number > 250]
# 绘制频数大于250次的词语的直方图
newdata.plot(kind = "bar", x = "Word", y = "number", figsize = (10, 7))
plt.xticks(FontProperties = font, size = 10)     # 设置x轴刻度上的文本
plt.xlabel("关键词", FontProperties = font)      # 设置x轴上的标签
plt.ylabel("频数", FontProperties = font)
plt.title("《红楼梦》", FontProperties = font)
plt.show()

print("************************************************")
print(Red_df)
# 保存数据
Red_df.to_json(r'F:\PythonTest\BigData\2018book\Red_dream_data.json')


##编写一个函数
def plotwordcloud(wordlist, title, figsize = (6, 6)):
    """
    该函数用来控制一个list的词云
    wordlist:词组成的一个列表
    title:图的名字
    """
    # 统计词频
    words = wordlist
    name = title
    word_df = pd.DataFrame({"Word":words})
    word_stat = word_df.groupby(by = ["Word"])["Word"].agg({"number":np.size})
    word_stat = word_stat.reset_index().sort_values(by = "number", ascending = False)
    word_stat["wordlen"] = word_stat.Word.apply(len)
    print(word_stat)
    # 将词和词频组成字典数据准备
    worddict = {}
    for key, value in zip(word_stat.Word, word_stat.number):
        worddict[key] = value
    # 生成词云，可以用generate_from_frequencies函数计算词频
    red_wc = WordCloud(font_path = "C:/Windows/Fonts/simsun.ttc",   # 设置字体
                      margin = 5, width = 1800, height = 1800,      # 设置字体的清晰度
                      background_color = "black",                  # 设置背景颜色
                      max_words = 800,                              # 设置词云显示的最大词数
                      max_font_size = 400,                          # 设置字体最大值
                      random_state = 42                             # 设置随机状态种子数
                      ).generate_from_frequencies(frequencies = worddict)
    # 绘制词云
    plt.figure(figsize = figsize)
    plt.imshow(red_wc)
    plt.axis("off")
    plt.title(name, FontProperties = font, size = 12)
    plt.show()

# 调用函数
import time
print("plot all red dream wordcloud")
t0 = time.time()
for ii in np.arange(12):
    ii = ii * 10
    name = Red_df.Chapter[ii] + ":" + Red_df.Leftname[ii] + "," + Red_df.Rightname[ii]
    words = Red_df.cutword[ii]
    plotwordcloud(words, name, figsize = (6, 6))
print("Plot all wordcloud use %.2fs"%(time.time() - t0))


def plotredmanfre(wordlist, title, figsize = (12, 6)):
    """
    该函数用来统计一个列表中人物出现的频率
    wordlist:词组成的一个列表
    title:图的名字
    """
    # 统计词频
    words = wordlist
    name = title
    word_df = pd.DataFrame({"Word": words})
    word_stat = word_df.groupby(by = ["Word"])["Word"].agg({"number": np.size})
    word_stat = word_stat.reset_index().sort_values(by = "number", ascending = False)
    wordname = word_stat.loc[word_stat.Word.isin(word_stat.iloc[:, 0].values)].reset_index(drop = True)
    # 直方图
    # 绘制直方图
    size = np.min([np.max([6, np.ceil(300 / wordname.shape[0])]), 12])
    wordname.plot(kind = "bar", x = "Word", y = "number", figsize = (10, 6))
    plt.xticks(FontProperties = font, size = size)
    plt.xlabel("人名",  FontProperties = font)
    plt.ylabel("频数",  FontProperties = font)
    plt.title(name, FontProperties = font)
    plt.show()
    
# 调用函数，为每一章出现次数较多的人物绘制直方图
import time
print("plot 所有章节的人物词频")
t0 = time.time()
for ii in np.arange(120):
    name = Red_df.Chapter[ii] + ":" + Red_df.Leftname[ii] + ":" + Red_df.Rightname[ii]
    words = Red_df.cutword[ii]
    plotredmanfre(words, name, figsize = (12, 6))
print("Plot 所有章节的人物词频 use %.2fs"%(time.time() - t0))
    
