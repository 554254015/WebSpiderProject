import jieba
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
from collections import Counter

content = open(r'F:\PythonTest\BigData\pachong.txt', encoding = 'utf-8')
mylist = list(content)

word_list = [" ".join(jieba.cut(sentence)) for sentence in mylist]
new_text = ' '.join(word_list)
con_words = [x for x in jieba.cut(new_text) if len(x) >= 2]
frequencies = Counter(con_words).most_common()
frequencies = dict(frequencies)
pac_mask = imread("apchong.png")
wc = WordCloud(font_path = 'simhei.ttf',
               background_color = 'white',
               max_words = 2000,
               mask = pac_mask).fit_words(frequencies)
plt.imshow(wc)
plt.axis("off")
plt.show()

wc.to_file(r'F:\PythonTest\BigData\我要的_fre.png')    # 保存词云图
