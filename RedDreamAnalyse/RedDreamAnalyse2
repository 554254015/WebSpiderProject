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
