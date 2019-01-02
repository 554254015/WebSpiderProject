import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = open(r'F:\PythonTest\BigData\pachong.txt', encoding='utf-8')
mylist = list(text)

# 将每个元素进行分词
word_list = [" ".join(jieba.cut(sentence)) for sentence in mylist]
new_text = ' '.join(word_list)
wordcloud = WordCloud(font_path = 'simhei.ttf', background_color = 'black').generate(new_text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
