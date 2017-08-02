#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import sys
import xlrd
import xlwt
import datetime


class cuidan(object):
    def __init__(self, name, numbers, people, date1, date2, status, beizhu, days):
        self.name = name
        self.numbers =numbers
        self.people = people
        self.date1 = date1
        self.date2 = date2
        self.status = status
        self.beizhu = beizhu
        self.days = days


def readluti(excel):
    # Open the workbook
    xl_workbook = xlrd.open_workbook(excel)
    # List sheet names, and pull a sheet by name
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])
    num_cols = xl_sheet.ncols  # Number of columns
    num_rows = xl_sheet.nrows
    lutilist = []
    for i in range(2, num_rows - 1):
        row = xl_sheet.row_values(i)
        if(row[1] != '序号' and (row[16] == "审核通过") and row[19] == "已下发"):
            # 录题信息
            date1 = xlrd.xldate.xldate_as_datetime(row[21], 0)
            date2 = xlrd.xldate.xldate_as_datetime(row[22], 0)
            beizhu = row[23] + "；" + row[25]
            lutilist.append(cuidan(row[3], row[18], row[20], date1, date2, row[23], beizhu, 0))
    return lutilist

def readjietu(excel):
    xl_workbook = xlrd.open_workbook(excel)
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])
    num_rows = xl_sheet.nrows
    jietulist = []
    for i in range(2, num_rows - 1):
        row = xl_sheet.row_values(i)
        if(row[1] != '序号' and type(row[8]) is float and row[9] != "已备份" and row[14] != "试做"):
            # 截图信息
            date1 = xlrd.xldate.xldate_as_datetime(row[8], 0)
            date2 = date1 + datetime.timedelta(days = 3)
            beizhu = row[12] + "；" + row[13]
            jietulist.append(cuidan(row[3], row[5], row[7], date1, date2, "已下发", beizhu, 0))
    return jietulist

def gencuidanbiao(lutilist, jietulist):
    resultlist = []
    now_time = datetime.datetime.now()
    print (now_time)
    if(len(lutilist) != 0):
        resultlist.append(cuidan("教辅名", "题数", "录题人", "下发时间", "预计完成时间", "状态", "备注", "超时天数"))
        for i in lutilist:
            if((now_time - i.date2).days > 2 and i.status != "已完成"):
                i.status = "超时"
                i.days = (now_time - i.date2).days
                resultlist.append(i)
    if(len(jietulist) != 0):
        resultlist.append(cuidan("教辅名", "题数", "截图人", "下发时间", "预计完成时间", "状态", "备注", "超时天数"))
        for i in jietulist:
            if ((now_time - i.date2).days > 1 and i.status != "已备份"):
                i.status = "超时"
                i.days = (now_time - i.date2).days
                resultlist.append(i)
    return resultlist

def writetoexcel(resultlist, path):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    m = 0
    for i in resultlist:
        sheet.write(m, 0, i.name)
        sheet.write(m, 1, i.numbers)
        sheet.write(m, 2, i.people)
        sheet.write(m, 3, str(i.date1))
        sheet.write(m, 4, str(i.date2))
        sheet.write(m, 5, i.status)
        sheet.write(m, 6, i.beizhu)
        sheet.write(m, 7, i.days)
        m += 1
    wbk.save(path)

if __name__ == "__main__":

    # log the precessing
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if (len(sys.argv) < 3):
        print ("Usage: 分配表 + 催单表输出地址")
    else:
        fenpeibiao = sys.argv[1]
        cuidanbiao = sys.argv[2]

    lutilist = readluti(fenpeibiao)
    jietulist = readjietu(fenpeibiao)
    cuidanlist = gencuidanbiao(lutilist, jietulist)
    writetoexcel(cuidanlist, cuidanbiao)