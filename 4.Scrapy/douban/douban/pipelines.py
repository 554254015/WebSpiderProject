# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import csv

class DoubanPipeline(object):
    def __init__(self):
        self.fp = open('TOP250.csv', 'w', encoding = 'utf-8')
        self.wrt = csv.DictWriter(self.fp, ['name', 'fen', 'words'])
        self.wrt.writeheader()
        
    def __del__(self):
        self.fp.close()
        
    def process_item(self, item, spider):
        self.wrt.writerow(item)
        return item
