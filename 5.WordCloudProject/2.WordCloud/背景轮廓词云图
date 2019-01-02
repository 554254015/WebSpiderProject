import jieba
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import matplotlib.pyplot as plt

content = open(r'F:\PythonTest\BigData\pachong.txt', encoding = 'utf-8')
mylist = list(content)
word_list = [" ".join(jieba.cut(sentence)) for sentence in mylist]
new_text = ' '.join(word_list)

pac_mask = imread("apchong.png")
wc = WordCloud(font_path = 'simhei.ttf', background_color = 'white', max_words = 2000, 
               mask = pac_mask).generate(new_text)
plt.imshow(wc)
plt.axis("off")
plt.show()
wc.to_file(r'F:\PythonTest\BigData\我要的.png')
