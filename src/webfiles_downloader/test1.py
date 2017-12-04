#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
import xlwt
import xlrd
import logging
import os.path
import sys
from bs4 import BeautifulSoup
import threading
import time
from xml.dom.minidom import Document
from xml.etree.ElementTree import Element, SubElement, ElementTree
import random
from haishoku.haishoku import Haishoku
import math
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# http://pdfzj.cn/2016/cpbook_0221/165.html

class item(object):
    def __init__(self, bookID, summary, status, product, component, version, importance, assignedto, reporttime, modifiedtime, description, comments):
        self.bookID = bookID
        self.summary = summary
        self.status = status
        self.reporttime = reporttime
        self.modifiedtime = modifiedtime
        self.description = description
        self.importance = importance
        self.assignedto = assignedto
        self.version = version
        self.comments = comments
        self.product = product
        self.component = component

def getdetailinfo(url, agentlist):
    # 具体方法可以参见官方文档 https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html
    # rand = random.randint(0, len(agentlist) - 1)
    # headers = {}
    # headers['User_Agent'] = agentlist[rand]
    p = re.compile('<[^>]+>')
    bs = BeautifulSoup(requests.get(url).text, "html.parser")
    buginfo = bs.find('td', attrs={'id': 'error_msg'})
    bugid = "Bug " + url.split("=")[1]
    status = " "
    product = " "
    component = " "
    version = " "
    assignedto = " "
    reporttime = " "
    modifiedtime = " "
    summary = " "
    importance = " "
    if(buginfo == None):
        basicalinfo = bs.find('div', attrs={'class': 'bz_alias_short_desc_container edit_form'})
        detailinfo = bs.find('table', attrs={'class': 'edit_form'})
        commentinfo = bs.find('table', attrs={'class': 'bz_comment_table'})
        detailinfotable1 = detailinfo.find('td', attrs={'id': 'bz_show_bug_column_1'})
        detailinfotable2 = detailinfo.find('td', attrs={'id': 'bz_show_bug_column_2'})

        # basical info
        # bugid = basicalinfo.find('a').string
        if(basicalinfo.find('span', attrs={'id': 'short_desc_nonedit_display'}) != None):
            summary = basicalinfo.find('span', attrs={'id': 'short_desc_nonedit_display'}).string
        list1 = detailinfotable1.find_all('tr')
        if(list1[0].find('span', attrs={'id': 'static_bug_status'}) != None):
            status = list1[0].find('span', attrs={'id': 'static_bug_status'}).string
        if(list1[2].find('td', attrs={'id': 'field_container_product'}) != None):
            product = list1[2].find('td', attrs={'id': 'field_container_product'}).string
        if(list1[4].find('td', attrs={'id': 'field_container_component'}) != None):
            component = list1[4].find('td', attrs={'id': 'field_container_component'}).string
        if(list1[5].find('td') != None):
            version = list1[5].find('td').string

        # assignment info
        importancelist = re.findall('<td>.*?(.*?)<span', str(list1[8]), re.S)
        if(len(importancelist) != 0):
            importance = importancelist[0]
        if (list1[10].find('span', attrs={'class': 'fn'}) != None):
            assignedto = list1[10].find('span', attrs={'class': 'fn'}).string

        # time info
        list2 = detailinfotable2.find_all('tr')
        reporttimelist = re.findall('<td>.*?(.*?)<span', str(list2[0]), re.S)
        if(len(reporttimelist) != 0):
            reporttime = reporttimelist[0].strip()[0:20]
        modifiedtimelist = re.findall('<td>.*?(.*?)<a', str(list2[1]), re.S)
        if (len(modifiedtimelist) != 0):
            modifiedtime = modifiedtimelist[0].strip()[0:20]

        # author's description
        description = {}
        desinfo = commentinfo.find('div', attrs={'class': 'bz_comment bz_first_comment'})
        if(desinfo.find('span', attrs={'class': 'fn'}) != None):
            description['author'] = desinfo.find('span', attrs={'class': 'fn'}).string
        else:
            description['author'] = " "
        if(desinfo.find('span', attrs={'class': 'bz_comment_time'}) != None):
            description['time'] = desinfo.find('span', attrs={'class': 'bz_comment_time'}).string.strip()[0:23]
        else:
            description['time'] = " "
        deslist = re.findall('<pre class="bz_comment_text">(.*?)</pre>', str(desinfo), re.S)
        if(len(deslist) != 0):
            description['detail'] = p.sub(" ", deslist[0])
        else:
            description['detail'] = " "

        # comments info list
        comments = []
        commentlist = commentinfo.find_all('div', attrs={'class': 'bz_comment'})
        if(len(commentlist) > 1):
            for i in range(1, len(commentlist)):
                comment = commentlist[i]
                com = {}
                if(comment.find('span', attrs={'class': 'fn'}) != None):
                    com['author'] = comment.find('span', attrs={'class': 'fn'}).string
                else:
                    com['author'] = " "
                if(comment.find('span' , attrs={'class': 'bz_comment_time'}) != None):
                    com['time'] = comment.find('span' , attrs={'class': 'bz_comment_time'}).string.strip()[0:23]
                else:
                    com['time'] = " "
                comlist = re.findall('<pre class="bz_comment_text">(.*?)</pre>', str(comment), re.S)
                if (len(comlist) != 0):
                    com['detail'] = p.sub(" ", comlist[0])
                else:
                    com['detail'] = " "
                comments.append(com)
    else:
        logger.info(buginfo.string.strip())
        status = "This bug not exist"
        description = {}
        comments = []
    bug = item(bugid, summary, status, product, component, version, importance, assignedto, reporttime, modifiedtime, description, comments)
    return bug


class PythonOrgSearch():
    def setUp(self):
        self.driver = webdriver.Chrome()

    def test_search_in_python_org(self):
        driver = self.driver
        driver.get("http://www.python.org")
        self.assertIn("Python", driver.title)
        elem = driver.find_element_by_name("q")
        elem.send_keys("pycon")
        elem.send_keys(Keys.RETURN)
        assert "No results found." not in driver.page_source

    def tearDown(self):
        self.driver.close()

if __name__ == "__main__":

    # log the precessing
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    #  http://pdfzj.cn/2016/cpbook_0221/165.html
