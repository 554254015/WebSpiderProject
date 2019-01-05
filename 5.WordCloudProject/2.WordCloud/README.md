# wordcloud
wordcloud包的基本用法如下：</br>
<p>wordcloud.WordCloud(font_path = None,
                    width = 400,
                    height = 200,
                    margin = 2,
                    ranks_only = None,
                    prefer_horizontal = 0.9,
                    mask = None,
                    scale = 1,
                    color_func = None,
                    max_words = 200,
                    min_font_size = 4,
                    stopwords = None,
                    random_state = None,
                    background_color = 'black',
                    max_font_size = None,
                    font_step = 1,
                    mode = 'RGB',
                    relative_scaling = 0.5,
                    regexp = None,
                    collocations = True,
                    colormap = None,
                    normalize_plurals = True)
</p>
这是wordcloud的所有参数格式，下面具体介绍各个参数。</br>
<b>normalize_plurals</b>：是否溢出单词末尾的's'，布尔型，默认为TRUE。</br>
<b>margin</b>：画布偏移，词语边缘距离，默认2像素。</br>
<b>ranks_only</b>：是否只用词频排序而不是实际词频统计值，默认None。</br>
<b>font_path</b>：数据类型为string，字体路径，需要展现什么字体就把该字体路径+后缀名写上，如font_path = '黑体.ttf'。</br>
<b>width</b>：数据类型为int（default = 400），输出的画布宽度，默认为400像素。</br>
<b>height</b>:数据类型为int（default = 200），输出的画布高度，默认为200像素。</br>
<b>prefer_horizontal</b>：数据类型为float（default = 0.90），词语水平方向排版出现的频率，默认为0.9（所以词语垂直方向排版出现频率为0.1）。</br>
<b>mask</b>：数据类型为nd_array or None（default = None）、如果mask参数为空，则使用二维遮罩绘制词云；如果mask为非空，设置的宽高值将被忽略，
遮罩形状被mask取代。除全白（#FFFFFF）的部分将不会绘制，其余部分会用于绘制词云。如bg_pic = imread('读取一张图片.png')，背景图片的画布
一定要设置为白色（#FFFFFF），然后显示的形状为不是白色的其他颜色。可以用Photoshop将自己要显示的形状复制到一个纯白的画布上再保存，就可以
了。</br>
<b>scale</b>：数据类型为float（default = 1），按照比例进行放大画布，如设置1.5，则长和宽都是原来画布的1.5倍。</br>
<b>min_font_size</b>：数据类型为int（default = 4），显示的最小的字体大小。</br>
<b>font_step</b>：数据类型为int（default = 1），字体步长。如果步长大于1，会加快运算但是可能导致结果出现较大的误差。</br>
<b>max_words</b>：数据类型为number（default = 200），要显示的词的最大个数。</br>
<b>stopwords</b>：数据类型为strings or None，设置需要屏蔽的词，如果为空，则使用内置的STOPWORDS。</br>
<b>background_color</b>：数据类型为color value（default = "black"），背景颜色，如：background_color = 'white'，背景颜色为白色。</br>
<b>max_font_size</b>：数据类型为int or None（default = None），显示的最大的字体大小。</br>
<b>mode</b>：数据类型为string（default = "RGB"），当参数为RGBA并且background_color不为空时，背景为透明。</br>
<b>relative_scaling</b>：数据类型为float（fault = .5），词频和字体大小的关联性。</br>
<b>color_func</b>：数据类型为callable，default = None，生成新颜色的函数，如果为空，则使用self.color_func。</br>
<b>regexp</b>：数据类型为string or None（optional），使用正则表达式分隔输入的文本。</br>
<b>collocations</b>：数据类型为bool，default=True，是否包括两个词的搭配。</br>
<b>colormap</b>：数据类型为string or matplotlib colormap，default = "viridis"，给每个单词随机分配颜色吗，若指定color_func,则忽略该方法。</br>
</br>
关于词云的方法有：</br>
generate（text）  //根据文本生成词云</br>
fit_words(frequencies)  //根据词频生成词云</br>
generate_from_frequencies(frequencies[,...])    // 根据词频生成词云</br>
generate_from_text(text)    //根据文本生成词云</br>
to_file(filename)     //输出到文件</br>
to_array()            //转化为 numpy array</br>
