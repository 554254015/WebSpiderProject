1、cut用法
jieba.cut方法的命令格式如下：
    jieba.cut(s, cut_all=True)
该方法接收两个输入参数：
（1）第一个参数s为需要分词的字符串；
（2）cut_all参数用来控制是否采用全模式

jieba.cut_for_search(s)方法的命令如下：
    jieba.cut_for_search(s)
jieba.cut_for_search方法接收一个参数s：需要被分词的字符串。该方法适用于搜索引擎构建倒排序索引的分词，粒度比较细

注意：待分词的字符串可以是gbk字符串、utf-8字符串或unicode。


2、词频与分词字典
词频（Term Frequency, TF）指的是某一个给定的词语在该文件中出现的次数。
这个次数通常会被正规化，以防止它偏向长文件（同一个词语在长文件里可能会比短文件有更高的词频，而不管词语重要与否）

