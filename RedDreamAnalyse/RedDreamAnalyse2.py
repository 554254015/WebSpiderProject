import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
from nltk.cluster.kmeans import KMeansClusterer
import matplotlib.pyplot as plt

# 设置字体
font = FontProperties(fname = 'C:/Windows/Fonts/simsun.ttc', size = 14)
# 设置pandas显示方式
pd.set_option("display.max_rows", 8)
pd.options.mode.chained_assignment = None    # default = 'warn'
# 在jupyter中设置显示图像的方式
%matplotlib inline
%config InlineBackend.figure_format = "retina"


# 读取数据
Red_df = pd.read_json("F:/PythonTest/BigData/2018book/Red_dream_data.json")

# 准备工作，将分词后的结果整理成CountVectorizer()可应用的形式
# 将所有分词后的结果使用空格连接为字符串，并组成列表，每一段为列表中的一个元素
articals = []
for cutword in Red_df.cutword:
    articals.append(" ".join(cutword))
# 构建语料库，并计算“文档一词”的TF-IDF矩阵
vectorizer = CountVectorizer()
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(articals)
# tfidf 以稀疏矩阵的形式存储
print(tfidf)
# 将tfidf转化为数组的形式，“文档一词”矩阵
dtm = tfidf.toarray()
dtm

# 使用夹角余弦距离进行k均值聚类
kmeans = KMeansClusterer(num_means = 3,                   # 聚类数目
                        distance = nltk.cluster.util.cosine_distance
                                                          # 夹角余弦距离
                        )
kmeans.cluster(dtm)
# 聚类得到的类别
labpre = [kmeans.classify(i) for i in dtm]
kmeanlab = Red_df[["ChapName", "Chapter"]]
kmeanlab["cosd_pre"] = labpre

# 查看每类有多少个分组
count = kmeanlab.groupby("cosd_pre").count()
print(kmeanlab)
# 将分类可视化
count.plot(kind = "barh", figsize = (6, 5))
for xx, yy, s in zip(count.index, count.ChapName, count.ChapName):
    plt.text(y = xx, x = yy, s = s)
plt.ylabel("cluster label")
plt.xlabel("number")
plt.show()


# 聚类结果可视化
# 使用MDS对数据进行降维
from sklearn.manifold import MDS
mds = MDS(n_components = 2, random_state = 123)
coord = mds.fit_transform(dtm)
print(coord.shape)
# 绘制降维后的结果
plt.figure(figsize = (8, 8))
plt.scatter(coord[:, 0], coord[:, 1], c = kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii, 0], coord[ii, 1], s = Red_df.Chapter2[ii])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means MDS")
plt.show()


# 聚类结果可视化
# 使用PCA对数据进行降维
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(dtm)
print(pca.explained_variance_ratio_)
# 对数据进行降维
coord = pca.fit_transform(dtm)
print(coord.shape)
# 绘制降维后的结果
plt.figure(figsize = (8, 8))
plt.scatter(coord[:, 0], coord[:, 1], c = kmeanlab.cosd_pre)
for ii in np.arange(120):
    plt.text(coord[ii, 0], coord[ii, 1], s = Red_df.Chapter2[ii])
plt.xlabel("主成分1", FontProperties = font)
plt.ylabel("主成分2", FontProperties = font)
plt.title("K-means PCA")
plt.show()


# 层次聚类
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import pdist, squareform
# 标签，每个章节的标题
labels = Red_df.Chapter.values
cosin_matrix = squareform(pdist(dtm, 'cosine'))   # 计算每章的距离矩阵
ling = ward(cosin_matrix)                         # 根据距离聚类
# 聚类结果可视化
fig, ax = plt.subplots(figsize = (10, 15))   # 设置大小
ax = dendrogram(ling, orientation = 'right', labels = labels)
plt.yticks(FontProperties = font, size = 8)
plt.title("《红楼梦》各章节层次聚类", FontProperties = font)
plt.tight_layout()      # 展示紧凑的绘图布局
plt.show()


# t-SNE高维数据可视化
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
# 准备工作，将分词后的结果整理成CountVectorizer()可应用的形式
# 将所有分词后的结果使用空格连接为字符串并组成列表，每一段为其中一个元素
articals = []
for cutword in Red_df.cutword:
    cutword = [s for s in cutword if len(s) < 5]
    cutword = " ".join(cutword)
    articals.append(cutword)
# max_features 参数根据出现的频率排序，只取指定的数目
vectorizer = CountVectorizer(max_features = 10000)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(articals))
# 降维为三维
X = tfidf.toarray()
tsne = TSNE(n_components = 3, init = 'random', random_state = 1233)
X_tsne = tsne.fit_transform(X)
# 可视化
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1, 1, 1, projection = "3d")
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c = "red")
ax.view_init(30, 45)
plt.xlabel("章节段数", FontProperties = font)
plt.ylabel("章节字数", FontProperties = font)
plt.title("《红楼梦》——t-SNE", FontProperties = font)
plt.show()


# LDA主题模型
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# 准备工作，将分词后的结果整理成CountVectorizer()可应用的形式
# 将所有分词后的结果使用空格连接为字符串，并组成列表，每一段为列表中的一个元素
articals = []
for cutword in Red_df.cutword:
    cutword = [s for s in cutword if len(s) < 5]
    cutword = " ".join(cutword)
    articals.append(cutword)
# max_features参数根据出现的频率排序，只取指定的数目
tf_vectorizer = CountVectorizer(max_features = 10000)
tf = tf_vectorizer.fit_transform(articals)

# 查看结果
print(tf_vectorizer.get_feature_names()[400:420])
tf.toarray()[20: 50, 200: 800]

# 主题数目
n_topics = 3
lda = LatentDirichletAllocation(n_topics = n_topics,
                              max_iter = 25,
                              learning_method = 'online',
                              learning_offset = 50,
                              random_state = 0)
# 模型应用于数据
lda.fit(tf)
# 等到每个章节属于某个主题的可能性
chapter_top = pd.DataFrame(lda.transform(tf),
                          index = Red_df.Chapter,
                          columns = np.arange(n_topics) + 1)
chapter_top
# 每一行的和
chapter_top.apply(sum, axis = 1).values
# 查看每一行的最大值
chapter_top.apply(max, axis = 1).values
# 找到大于相应值的索引
np.where(chapter_top >= np.min(chapter_top.apply(max, axis = 1).values))

# 可视化主题，主成分分析可视化LDA
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']     # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False      # 解决保存图像负号‘-’显示为方块的问题

n_top_words = 40
tf_feature_names = tf_vectorizer.get_feature_names()
for topic_id, topic in enumerate(lda.components_):
    topword = pd.DataFrame(
        {"word": [tf_feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]],
        "componets": topic[topic.argsort()[: -n_top_words - 1 : -1]]})
    topword.sort_values(by = "componets").plot(kind = "barh",
                                              x = "word",
                                              y = "componets",
                                              figsize = (6, 8),
                                              legend = False)
    plt.yticks(FontProperties = font, size = 10)
    plt.ylabel("")
    plt.xlabel("")
    plt.title("Topic %d"%(topic_id + 1))
    plt.show()
    
# 查看每个主题的关键词
def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic Nr.%d:' % int(topic_id + 1))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
                      + '|' for i in topic.argsort()[: -n_top_words - 1 : -1]]))
n_top_words = 10
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# 人物社交网络分析
# 两种方式：1、如果两个人名同时出现在同一段落，则其联系+1
#           2、如果两个人名同时出现在同一章节，则其联系+1
# 加载绘制社交网络图的包
import networkx as nx
# 读取数据
Red_df = pd.read_csv(r'F:\PythonTest\BigData\2018book\red_social_net_weight.csv')
Red_df.head()

# 计算其中的一种权重
Red_df["weight"] = Red_df.chapweight / 120
Red_df2 = Red_df[Red_df.weight > 0.025].reset_index(drop = True)
plt.figure(figsize = (12, 12))
# 生成社交网络图
G = nx.Graph()
# 添加边
for ii in Red_df2.index:
    G.add_edge(Red_df2.First[ii], Red_df2.Second[ii], weight = Red_df2.weight[ii])
# 定义3种边
elarge = [(u, v) for (u, v, d) in G.edges(data = True) if d['weight'] > 0.2]
emidle = [(u, v) for (u, v, d) in G.edges(data = True) if (d['weight'] > 0.1) & (d['weight'] <= 0.2)]
esmall = [(u, v) for (u, v, d) in G.edges(data = True) if d['weight'] < 0.1]
# 图的布局
pos = nx.spring_layout(G)  #positions for all nodes
# nodes
nx.draw_networkx_nodes(G, pos, alpha = 0.6, node_size = 350)
# edges
nx.draw_networkx_edges(G, pos, edgelist = elarge,
                      width = 2, alpha = 0.9, edge_color = 'g')
nx.draw_networkx_edges(G, pos, edgelist = emidle,
                      width = 1.5, alpha = 0.6, edge_color = 'y')
nx.draw_networkx_edges(G, pos, edgelist = esmall,
                      width = 1, alpha = 0.3, edge_color = 'b', style = 'dashed')
# labels
nx.draw_networkx_labels(G, pos, font_size = 10)
plt.axis('off')
plt.title("《红楼梦》社交网络")
plt.show()

# 计算每个节点的度
Gdegree = nx.degree(G)
Gdegree = dict(Gdegree)
Gdegree = pd.DataFrame({"name": list(Gdegree.keys()),
                       "degree": list(Gdegree.values())})
Gdegree.sort_values(by = "degree", ascending = False).plot(x = "name",
                                                          y = "degree",
                                                          kind = "bar",
                                                          figsize = (12, 6),
                                                          legend = False)
plt.xticks(FontProperties = font, size = 5)
plt.ylabel("degree")
plt.show()

plt.figure(figsize = (12, 12))
# 生成社交网络图
G = nx.Graph()
# 添加边
for ii in Red_df2.index:
    G.add_edge(Red_df2.First[ii], Red_df2.Second[ii], weight = Red_df2.weight[ii])
# 定义两种边
elarge = [(u, v) for (u, v, d) in G.edges(data = True) if d['weight'] > 0.3]
emidle = [(u, v) for (u, v, d) in G.edges(data = True) if (d['weight'] > 0.2) & (d['weight'] <= 0.3)]
esmall = [(u, v) for (u, v, d) in G.edges(data = True) if d['weight'] <= 0.2]
# 图的布局
pos = nx.circular_layout(G) # positions for all nodes
# nodes 根据节点的入度和出度来设置节点的大小
nx.draw_networkx_nodes(G, pos, alpha = 0.6, node_size = 20 + Gdegree.degree * 15)
# edges
nx.draw_networkx_edges(G, pos, edgelist = elarge,
                      width = 2, alpha = 0.9, edge_color = 'g')
nx.draw_networkx_edges(G, pos, edgelist = emidle,
                      width = 1.5, alpha = 0.6, edge_color = 'y')
nx.draw_networkx_edges(G, pos, edgelist = esmall,
                      width = 1, alpha = 0.3, edge_color = 'b', style = 'dashed')
# labels
nx.draw_networkx_labels(G, pos, font_size = 10) # font_size = 10, 即设置图中字体的大小
plt.axis('off')
plt.title("《红楼梦》社交网络")
plt.show()
