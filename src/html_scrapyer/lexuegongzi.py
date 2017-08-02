#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import sys
import xlrd
import xlwt


class people(object):
    def __init__(self, name, idnum, cardnum, province, banksite, dailing, realname):
        self.name = name
        self.idnum = idnum
        self.cardnum = cardnum
        self.province = province
        self.banksite = banksite
        self.dailing = dailing
        self.realname = realname

def readpeopleinfo(excel):
    # Open the workbook
    xl_workbook = xlrd.open_workbook(excel)
    # List sheet names, and pull a sheet by name
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])
    num_cols = xl_sheet.ncols  # Number of columns
    num_rows = xl_sheet.nrows
    resultinfo = []
    for i in range(1, num_rows - 1):
        row = xl_sheet.row_values(i)
        item = people(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
        resultinfo.append(item)
    return resultinfo

def readjietu(excel):
    # Open the workbook
    xl_workbook = xlrd.open_workbook(excel)
    # List sheet names, and pull a sheet by name
    sheet_names = xl_workbook.sheet_names()
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])
    num_cols = xl_sheet.ncols  # Number of columns
    num_rows = xl_sheet.nrows
    jietulist = []
    lutilist = []
    pptlist = []
    for i in range(2 , num_rows - 1):
        row = xl_sheet.row_values(i)
        jietudict = {}
        lutidict = {}
        pptdict = {}
        if(row[1] != '序号'):
            # 截图信息
            jietudict["教辅名"] = row[3]
            jietudict["链接"] = row[4]
            jietudict["页数"] = row[5]
            jietudict["截图人"] = row[7]
            jietudict["已完成"] = row[9]
            jietudict["工资"] = row[11]
            jietudict["有答案"] = row[12]
            jietudict["备注"] = row[13]
            jietudict["试做"] = row[14]
            # 录题信息
            lutidict["教辅名"] = row[3]
            lutidict["题数"] = row[18]
            lutidict["录题人"] = row[20]
            lutidict["已完成"] = row[23]
            lutidict["工资"] = row[24]
            lutidict["备注"] = row[25]
            # 做ppt信息
            if(type(row[26]) is float):
                pptdict["教辅名"] = row[3]
                pptdict["页数"] = row[5]
                pptdict["做题人"] = row[28]
                pptdict["已完成"] = row[30]
                pptdict["工资"] = row[31]
                pptdict["备注"] = row[32]
        if(len(jietudict) != 0):
            jietulist.append(jietudict)
        if(len(lutidict) != 0):
            lutilist.append(lutidict)
        if(len(pptdict) != 0):
            pptlist.append(pptdict)
    return jietulist, lutilist, pptlist

def maketheform(peoplelist, jietulist, lutilist, pptlist):
    resultlist = []

    dyyformdict = {}
    for hh in peoplelist:
        if (str(hh.name) == "刁艳玉"):
            dyyformdict["people"] = hh
            print(hh.name)
    dyyjieturesult = []

    numsjietu = 0
    for jietu in jietulist:
        if (jietu["有答案"] == "有答案"):
            jietu["单件工资"] = 65.0
        elif (type(jietu["页数"]) is float):
            if (jietu["页数"] >= 180.0):
                jietu["单件工资"] = 50.0
            elif (jietu["页数"] < 180.0):
                jietu["单件工资"] = 40.0
        else:
            jietu["单件工资"] = 0.0
        if (jietu["试做"] == "试做" and jietu["已完成"] == "已备份" and jietu["工资"] != "已结"):
            dyyjieturesult.append(jietu)

    for luti in lutilist:
        if (type(luti["题数"]) is float):
            luti["单件工资"] = luti["题数"] / 2.0
        else:
            luti["单件工资"] = 0.0
    for ppt in pptlist:
        ppt["单件工资"] = 25.0

    jietunumbers = len(dyyjieturesult)
    lutinumbers = 0
    pptnumbers = 0
    for people in peoplelist:
        formdict = {}
        formdict["people"] = people
        if(people.dailing != 1.0):
            peoplename = people.name
        else:
            peoplename = people.realname
        jieturesult = []
        lutiresult = []
        pptresult = []
        for jietu in jietulist:
            if(jietu["截图人"] == peoplename and jietu["已完成"] == "已备份" and jietu["工资"] != "已结"):
                jieturesult.append(jietu)
                jietunumbers += 1
        for luti in lutilist:
            if(luti["录题人"] == peoplename and luti["已完成"] == "已完成" and luti["工资"] != "已结"):
                lutiresult.append(luti)
                lutinumbers += 1
        for ppt in pptlist:
            if(ppt["做题人"] == peoplename and ppt["已完成"] == "已完成" and ppt["工资"] != "已结"):
                pptresult.append(ppt)
                pptnumbers += 1
        if(len(jieturesult) != 0):
            formdict["jietu"] = jieturesult
        if(len(lutiresult) != 0):
            formdict["luti"] = lutiresult
        if(len(pptresult) != 0):
            formdict["ppt"] = pptresult
        resultlist.append(formdict)
    print ("本次截图: " + str(jietunumbers) + " 本次录题: " + str(lutinumbers) + " 本次ppt: " + str(pptnumbers))
    if(len(dyyjieturesult) != 0):
        dyyformdict["jietu"] = dyyjieturesult
    resultlist.append(dyyformdict)
    return resultlist

def writeformtoexcel(formlist, excelpath):
    wbk = xlwt.Workbook()
    peoplenumbers = len(formlist)
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    peoplenum = 1
    lastline = 0
    totalmoney = 0
    mymoney = 0
    for form in formlist:
        if("jietu" in form or "luti" in form or "ppt" in form):
            peopleline = 0
            peoplemoney = 0
            sheet.write(lastline, 0, "序号")
            sheet.write(lastline, 1, peoplenum)
            peoplenum = peoplenum + 1
            peopleline = peopleline + 1

            sheet.write(lastline + peopleline, 0, "姓名")
            sheet.write(lastline + peopleline, 1, str(form["people"].name))
            sheet.write(lastline + peopleline, 2, str(form["people"].idnum))
            sheet.write(lastline + peopleline, 3, str(form["people"].cardnum))
            sheet.write(lastline + peopleline, 4, str(form["people"].province))
            sheet.write(lastline + peopleline, 5, str(form["people"].banksite))
            peopleline = peopleline + 1

            sheet.write(lastline + peopleline, 0, "总计")
            moneyline = lastline + peopleline
            peopleline = peopleline + 1

            sheet.write(lastline + peopleline, 0, "备注")
            peopleline = peopleline + 1

            sheet.write(lastline + peopleline, 0, "详细信息")
            sheet.write(lastline + peopleline, 1, "序号")
            sheet.write(lastline + peopleline, 2, "项目")
            sheet.write(lastline + peopleline, 3, "备注")
            sheet.write(lastline + peopleline, 4, "计数")
            sheet.write(lastline + peopleline, 5, "工资")
            peopleline = peopleline + 1

            if("jietu" in form):
                sheet.write(lastline + peopleline, 2, "截图")
                peopleline = peopleline + 1
                jietulist = form["jietu"]
                for i in range(len(jietulist)):
                    sheet.write(lastline + peopleline, 1, i + 1)
                    sheet.write(lastline + peopleline, 2, str(jietulist[i]["教辅名"]))
                    sheet.write(lastline + peopleline, 3, str(jietulist[i]["备注"]))
                    sheet.write(lastline + peopleline, 4, str(jietulist[i]["页数"]))
                    sheet.write(lastline + peopleline, 5, str(jietulist[i]["单件工资"]))
                    peoplemoney = peoplemoney + jietulist[i]["单件工资"]
                    peopleline = peopleline + 1
                totalmoney = totalmoney + len(jietulist) * 10.0
                if(form["people"].name == "刁艳玉"):
                    mymoney += peoplemoney
                else:
                    mymoney += len(jietulist) * 10.0
            if("luti" in form):
                sheet.write(lastline + peopleline, 2, "录题")
                peopleline = peopleline + 1
                lutilist = form["luti"]
                for i in range(len(lutilist)):
                    sheet.write(lastline + peopleline, 1, i + 1)
                    sheet.write(lastline + peopleline, 2, str(lutilist[i]["教辅名"]))
                    sheet.write(lastline + peopleline, 3, str(lutilist[i]["备注"]))
                    sheet.write(lastline + peopleline, 4, str(lutilist[i]["题数"]))
                    sheet.write(lastline + peopleline, 5, str(lutilist[i]["单件工资"]))
                    peoplemoney = peoplemoney + lutilist[i]["单件工资"]
                    peopleline = peopleline + 1
            if("ppt" in form):
                sheet.write(lastline + peopleline, 2, "做ppt")
                peopleline = peopleline + 1
                pptlist = form["ppt"]
                for i in range(len(pptlist)):
                    sheet.write(lastline + peopleline, 1, i + 1)
                    sheet.write(lastline + peopleline, 2, str(pptlist[i]["教辅名"]))
                    sheet.write(lastline + peopleline, 3, str(pptlist[i]["备注"]))
                    sheet.write(lastline + peopleline, 4, str(pptlist[i]["页数"]))
                    sheet.write(lastline + peopleline, 5, str(pptlist[i]["单件工资"]))
                    peoplemoney = peoplemoney + pptlist[i]["单件工资"]
                    peopleline = peopleline + 1
            totalmoney += peoplemoney
            sheet.write(moneyline, 1, peoplemoney)
            lastline = lastline + peopleline + 5
    print ("本次总钱数  本次我的钱数")
    print (totalmoney, mymoney)
    wbk.save(excelpath)

if __name__ == "__main__":

    # log the precessing
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if(len(sys.argv) < 4):
        print ("Usage: 银行信息表 + 分配表 + 工资表输出地址")
    else:
        cardinfo = sys.argv[1]
        fenpeibiao = sys.argv[2]
        gongzibiao = sys.argv[3]

    peoplelist = readpeopleinfo(cardinfo)
    jietulist, lutilist, pptlist = readjietu(fenpeibiao)
    print ("截图完成总数/截图总数  录题完成总数/录题总数  ppt完成总数/ppt总数")
    print (len(jietulist), len(lutilist), len(pptlist))
    resultlist = maketheform(peoplelist, jietulist, lutilist, pptlist)
    writeformtoexcel(resultlist, gongzibiao)