# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 22:58:32 2018

@author: LINQUN
"""

# Scrapy是Python开发的一个快速、高层次的屏幕抓取和Web抓取框架，用于抓取Web站点并从页面中提取结构化的数据。
# Scrapy用途广泛，可以用于数据挖掘、监测和自动化测试。
# Scrapy依赖于pywin32,Twisted,lxml,最后安装Scrapy

import os 
pname = input('项目名：')
os.system("scrapy startproject " + pname)
os.chdir(pname)
wname = input('爬虫名：')
sit = input('网址：')
os.system('scrapy genspider ' + wname + ' ' + sit)
runc = """
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from %s.spiders.%s import %s

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