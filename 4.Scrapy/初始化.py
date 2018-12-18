
import os 
pname = input('项目名：')
os.system("scrapy startproject " + pname)
os.chdir(pname)
wname = input('爬虫名：')
sit = input('网址：')
os.system('scrapy genspider ' + wname + ' ' + sit)
runc = """
from scrapy.srawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from %s.souders.%s import %s

# 获取settings.py模块的设置
settings = get_project_settings()
process = CrawlerProcess(settings = settings)

# 可以添加多个spider
# process.crawl(Spider1)
# process.crawl(Spider2)
process.crawl(%s)

# 启动爬虫，会阻塞，直到爬取完成
process.start()
""" % (pname, wname, wname[0].upper() + wname[1:] + 'Spider', wname[0].upper() + wname[1:] + 'Spider')

with open('main.py', 'w', encoding = 'utf-8') as f:
    f.write(runc)
input('end')
