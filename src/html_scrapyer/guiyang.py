#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import requests
import xlwt
import xlrd
import logging
import os.path
import sys

class item(object):
    def __init__(self, programnum, date, district, programname, organization, money, url):
        self.programnum = programnum
        self.date = date
        self.district = district
        self.programname = programname
        self.organization = organization
        self.money = money
        self.url = url

def geturllist(url):
    html = requests.get(url)
    urllist = re.findall('<span class="span1" style="width:660px;font-size:14px">.*?<a href=(.*?)>',
                       html.text, re.S)
    return urllist

def getallurl():
    targeturl = []
    resulturl = []
    str1 = "http://www.gygp.gov.cn/list-13-"
    for i in range(298):
        if i != 0:
            targeturl.append(str1 + str(i) + ".html")
    print(len(targeturl))
    print(targeturl)
    for url in targeturl:
        list = geturllist(url)
        for j in list:
            resulturl.append(j[2:-3])
    print(len(resulturl))
    print(resulturl)

    return resulturl

def getdetailinfo(url):
    programnum = ""
    date = ""
    district = ""
    programname = ""
    organization = ""
    money = ""

    #url = "http://www.gygp.gov.cn/content-13-18464-1.html"
    html = requests.get(url)

    district1 = re.findall('根据法律法规、部门规章和招标文件的规定，.*?<u>(.*?)</u>.*?的', html.text, re.S)
    district2 = re.findall('受.*?<u>(.*?)</u>.*?委托', html.text, re.S)
    if(len(district1) != 0):
        district = district1[0]
    elif(len(district2) != 0):
        district = district2[0]
    else:
        district = ""

    programnum1 = re.findall('交易编号：(.*?)<br/>', html.text, re.S)
    if(len(programnum1) != 0 and len(programnum1[0]) > 14):
        programnum1 = re.findall('(.*?)<br />', programnum1[0], re.S)
    if (len(programnum1) != 0 and len(programnum1[0]) > 14):
        programnum1 = re.findall('(.*?)</span>', programnum1[0], re.S)
    if(len(programnum1) == 0):
        programnum1 = re.findall('交易编号：(.*?)</span>', html.text, re.S)
    if (len(programnum1) == 0):
        programnum1 = re.findall('项目编号：(.*?)</span>', html.text, re.S)
    if (len(programnum1) != 0):
        programnum = programnum1[0]

    date1 = re.findall('公告日期：.*?公告日期：(.*?)</p>', html.text, re.S)
    if(len(date1) != 0 and len(date1[0]) > 11):
        date1 = re.findall('(.*?)&nbsp;', date1[0], re.S)
    if (len(date1) != 0 and len(date1[0]) > 11):
        date1 = re.findall('(.*?)</span>', date1[0], re.S)
    date2 = re.findall('评标日期：(.*?)</p>', html.text, re.S)
    if(len(date1) != 0):
        date = date1[0]
    elif(len(date2) != 0):
        date = date2[0]
    else:
        date = ""

    table1 = re.findall('<tr>(.*?)</tr>', html.text, re.S)
    if(len(table1) != 0):
        for i in range(len(table1)):
            if(i != 0):
                organization1 = re.findall('&nbsp;(.*?)</td>', table1[i], re.S)
                if(len(organization1) > 3):
                    organization = organization + ";" + organization1[1]
                    programname = programname + ";" + organization1[0]
                    money = money + ";" + organization1[2]
                else:
                    organization = ""
                    programname = ""
                    money = ""
    else:
        organization = ""
        programname = ""
        money = ""

    file = item(programnum, date, district, programname, organization, money, url)
    return file

# 根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引
def read_excel_to_list(filepath):
    file = xlrd.open_workbook(filepath)
    table = file.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    list = []
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        if row:
            list.append(row[1])
    return list


def write_list_to_excel(list, filepath):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    sheet.write(0, 0, '编号')
    sheet.write(0, 1, '日期')
    sheet.write(0, 2, '区域')
    sheet.write(0, 3, '项目名称')
    sheet.write(0, 4, '中标机构名称')
    sheet.write(0, 5, '金额')
    sheet.write(0, 6, '网址')
    m = 1
    for i in list:
        sheet.write(m, 0, i.programnum)
        sheet.write(m, 1, i.date)
        sheet.write(m, 2, extract_distrcit(i.district))
        sheet.write(m, 3, i.programname[1:])
        sheet.write(m, 4, i.organization[1:])
        sheet.write(m, 5, i.money[1:])
        sheet.write(m, 6, i.url)
        m = m + 1
        wbk.save(filepath)

def write_list_to_excel1(list, filepath):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    sheet.write(0, 0, '编号')
    sheet.write(0, 1, 'url')
    m = 1
    for i in list:
        sheet.write(m, 0, m)
        sheet.write(m, 1, i)
        m = m + 1
    wbk.save(filepath)

def extract_distrcit(str):
    dict = ['南明', '云岩', '花溪', '乌当', '白云', '小河', '清镇', '红枫湖', '开阳', '城关镇', '修文',
            '龙场镇', '息烽', '永靖']
    for i in dict:
        if i in str:
            return i
    return str

if __name__ == "__main__":

    # log the precessing
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    #urllist = getallurl()
    #write_list_to_excel1(urllist, 'C:\\Users\\admin\\Desktop\\2.xls')
    result = []
    urllist = read_excel_to_list('C:\\Users\\admin\\Desktop\\2.xls')
    print("Read urls successfully!")
    print(len(urllist))
    print(urllist)

    for i in range(len(urllist)):
        print(i + 1, urllist[i])
        if(i %200 == 0):
            print("Processed: " + str(i) + " urls")
        result.append(getdetailinfo(urllist[i]))


    print(len(result))
    write_list_to_excel(result, 'C:\\Users\\admin\\Desktop\\1.xls')
    print("Write to excel successfully!")
