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

# https://bugs.eclipse.org/bugs/show_bug.cgi?id=520375 2017-08-01 00:00:00
class item(object):
    def __init__(self, bugid, summary, status, product, component, version, importance, assignedto, reporttime, modifiedtime, description, comments):
        self.bugid = bugid
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

def get_domain_color(image):

    print()


def readexceltolist(excel):
    # Open the workbook
    xl_workbook = xlrd.open_workbook(excel)
    # List sheet names, and pull a sheet by name
    sheet_names = xl_workbook.sheet_names()
    print('Sheet Names', sheet_names)
    xl_sheet = xl_workbook.sheet_by_name(sheet_names[0])
    num_cols = xl_sheet.ncols  # Number of columns
    num_rows = xl_sheet.nrows
    resultlist = []
    for i in range(1 , num_rows - 1):
        row = xl_sheet.row(i)
        rowlist = []
        for j in range (0, num_cols - 1):
            rowlist.append(row[j])
        resultlist.append(rowlist)
    return resultlist

def getdetailinfo(url, agentlist):
    # 具体方法可以参见官方文档 https://www.crummy.com/software/BeautifulSoup/bs4/doc/index.zh.html
    rand = random.randint(0, len(agentlist) - 1)
    headers = {}
    headers['User_Agent'] = agentlist[rand]
    p = re.compile('<[^>]+>')
    bs = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
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

def write_list_to_excel1(list, filepath):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('Sheet1', cell_overwrite_ok=True)
    sheet.write(0, 0, 'bugid')
    sheet.write(0, 1, 'reporttime')
    sheet.write(0, 2, 'modifiedtime')
    sheet.write(0, 3, 'description')
    sheet.write(0, 4, 'importance')
    sheet.write(0, 5, 'assignedto')
    sheet.write(0, 6, 'version')
    sheet.write(0, 7, 'summary')
    sheet.write(0, 8, 'status')
    sheet.write(0, 9, 'product')
    sheet.write(0, 10, 'component')
    sheet.write(0, 11, 'comments')
    m = 1
    p = re.compile('<[^>]+>')
    for bug in list:
        sheet.write(m, 0, "" + str(bug.bugid))
        sheet.write(m, 1, "" + str(bug.reporttime))
        sheet.write(m, 2, "" + str(bug.modifiedtime))
        des = " "
        if(len(bug.description) != 0):
            des = des + bug.description['author'] + "@@" + bug.description['time'] + "@@" + bug.description['detail']
            #des = des.replace('\n', '')
            if (len(des) > 32000):
                des = "!!!Too many characters!!!@@@" + bug.description['author'] + "@@" + bug.description['time']
        sheet.write(m, 3, p.sub(" ", des))
        sheet.write(m, 4, "" + str(bug.importance))
        sheet.write(m, 5, "" + str(bug.assignedto))
        sheet.write(m, 6, "" + str(bug.version))
        sheet.write(m, 7, "" + str(bug.summary))
        sheet.write(m, 8, "" + str(bug.status))
        sheet.write(m, 9, "" + str(bug.product))
        sheet.write(m, 10, "" + str(bug.component))
        comm = " "
        if(len(bug.comments) != 0):
            for com in bug.comments:
                comm = comm + com['author'] + "@@" + com['time'] + "@@" + com['detail'] + "@@@"
                #comm = comm.replace('\n', ' ')
            if (len(comm) > 32000):
                comm = "!!!Too many characters!!!@@@"
                for com in bug.comments:
                    comm = comm + com['author'] + "@@" + com['time'] + "@@@"
        sheet.write(m, 11, p.sub(" ", comm))
        m = m + 1
    wbk.save(filepath)
    #print("Saved to " + filepath + " successfully!")

def gentheurl(start, step):
    url = "https://bugs.eclipse.org/bugs/show_bug.cgi?id="
    resultlist = []
    for i in range(start * step, (start + 1) * step):
        resultlist.append(url + str(i + 1))
    return resultlist

def getinfolist(urllist):
    result = []
    User_Agent = []
    User_Agent.append("Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19")
    User_Agent.append("Mozilla/5.0 (Linux; U; Android 4.0.4; en-gb; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30")
    User_Agent.append("Mozilla/5.0 (Linux; U; Android 2.2; en-gb; GT-P1000 Build/FROYO) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1")
    User_Agent.append("Mozilla/5.0 (Windows NT 6.2; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0")
    User_Agent.append("Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0")
    User_Agent.append("Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19")
    User_Agent.append("Mozilla/5.0 (iPad; CPU OS 5_0 like Mac OS X) AppleWebKit/534.46 (KHTML, like Gecko) Version/5.1 Mobile/9A334 Safari/7534.48.3")
    User_Agent.append("Mozilla/5.0 (iPod; U; CPU like Mac OS X; en) AppleWebKit/420.1 (KHTML, like Gecko) Version/3.0 Mobile/3A101a Safari/419.3")
    User_Agent.append("Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36")
    i = 0
    for url in urllist:
        detailinfo = getdetailinfo(url, User_Agent)
        if(detailinfo != None):
            result.append(detailinfo)
            time.sleep(10)
        i += 1
        if(int(url.split('=')[-1]) % 50 == 0):
            logger.info("Prosecced: " + url.split('=')[-1] + " urls.")
    return result

def spliturls(urllist, cores):
    result =[]
    partsize = int(len(urllist) / cores)
    for i in range(0, cores - 1):
        parturl = []
        for u in range(i * partsize, (i + 1) * partsize):
            parturl.append(urllist[u])
        result.append(parturl)
    parturl = []
    for u in range((cores - 1) * partsize, len(urllist)):
        parturl.append(urllist[u])
    result.append(parturl)
    return result

class myThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, urllist, resultlist):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.urllist = urllist
        self.resultlist = resultlist

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        logger.info("Starting thread: " + str(self.threadID))
        self.resultlist = getinfolist(self.urllist)
        logger.info("Exiting thread: " + str(self.threadID))

    def getresult(self):
        return self.resultlist

def multithread(urllist, cores):
    parturllist = spliturls(urllist, cores)
    resultlist = []
    result = []
    threads = []
    for i in range(cores):
        thread = myThread(i + 1, parturllist[i], resultlist)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
        list = t.getresult()
        for text in list:
            result.append(text)
    return result

def sleeptime(hour, min, sec):
    return hour * 3600 + min * 60 + sec

def writeInfoToXml1(list, filename):
    # 生成根节点
    root = Element('root')
    # 生成第一个子节点 head
    head = SubElement(root, 'head')
    # head 节点的子节点
    title = SubElement(head, 'title')
    title.text = 'Well Dola!'
    # 生成 root 的第二个子节点 body
    body = SubElement(root, 'body')
    # body 的内容
    body.text = 'I love Dola!'
    tree = ElementTree(root)
    tree.write(filename, encoding='utf-8')

    print()

# 将list中的信息写入本地xml文件，参数filename是xml文件名
def writeInfoToXml(list, filename):
    # 创建dom文档
    doc = Document()
    # 创建根节点
    buglist = doc.createElement('buglist')
    # 根节点插入dom树
    doc.appendChild(buglist)
    # 依次将list中的每一组元素提取出来，创建对应节点并插入dom树
    for bug in list:
        # 分离出信息
        (bugid, summary, status, product, component, version, importance, assignedto, reporttime, modifiedtime, description, comments) = \
            (bug.bugid, bug.summary, bug.status, bug.product, bug.component, bug.version, bug.importance, bug.assignedto, bug.reporttime, bug.modifiedtime, bug.description, bug.comments)
        # 每一组信息先创建节点<bugitem>，然后插入到父节点<buglist>下
        bug_item = doc.createElement('bug')
        buglist.appendChild(bug_item)

        # 将bugid插入<bug_item>中
        # 创建节点<bugid_item>
        bugid_item = doc.createElement('bugid')
        # 创建<bugid_item>下的文本节点
        bugid_text = doc.createTextNode(bugid)
        # 将文本节点插入到<bugid_item>下
        bugid_item.appendChild(bugid_text)
        # 将<bugid_item>插入到父节点<bug_item>下
        bug_item.appendChild(bugid_item)

        # summary
        summary_item = doc.createElement('summary')
        summary_text = doc.createTextNode(str(summary))
        summary_item.appendChild(summary_text)
        bug_item.appendChild(summary_item)

        # status
        status_item = doc.createElement('status')
        status_text = doc.createTextNode(str(status))
        status_item.appendChild(status_text)
        bug_item.appendChild(status_item)

        # component
        component_item = doc.createElement('component')
        component_text = doc.createTextNode(str(component))
        component_item.appendChild(component_text)
        bug_item.appendChild(component_item)

        # version
        version_item = doc.createElement('version')
        version_text = doc.createTextNode(str(version))
        version_item.appendChild(version_text)
        bug_item.appendChild(version_item)

        # importance
        importance_item = doc.createElement('importance')
        importance_text = doc.createTextNode(str(importance))
        importance_item.appendChild(importance_text)
        bug_item.appendChild(importance_item)

        # assignedto
        assignedto_item = doc.createElement('assignedto')
        assignedto_text = doc.createTextNode(str(assignedto))
        assignedto_item.appendChild(assignedto_text)
        bug_item.appendChild(assignedto_item)

        # component
        component_item = doc.createElement('component')
        component_text = doc.createTextNode(str(component))
        component_item.appendChild(component_text)
        bug_item.appendChild(component_item)

        # reporttime
        reporttime_item = doc.createElement('reporttime')
        reporttime_text = doc.createTextNode(str(reporttime))
        reporttime_item.appendChild(reporttime_text)
        bug_item.appendChild(reporttime_item)

        # modifiedtime
        modifiedtime_item = doc.createElement('modifiedtime')
        modifiedtime_text = doc.createTextNode(str(modifiedtime))
        modifiedtime_item.appendChild(modifiedtime_text)
        bug_item.appendChild(modifiedtime_item)

        # description
        description_item = doc.createElement('description')
        if (len(description) != 0):
            description_author = doc.createElement('author').appendChild(doc.createTextNode(str(description['author'])))
            description_time = doc.createElement('time').appendChild(doc.createTextNode(str(description['time']))  )
            description_detail = doc.createElement('detail').appendChild(doc.createTextNode(str(description['detail'])))
            description_item.appendChild(description_author)
            description_item.appendChild(description_time)
            description_item.appendChild(description_detail)
        else:
            description_item.appendChild(doc.createTextNode("no description"))
        bug_item.appendChild(description_item)

        # comments
        comments_item = doc.createElement('comments')
        if (len(comments) != 0):
            comments_numbers_item = doc.createElement('commentsnumbers')
            comments_numbers_text = doc.createTextNode(str(len(comments)))
            comments_numbers_item.appendChild(comments_numbers_text)
            comments_item.appendChild(comments_numbers_item)
            for comment in comments:
                comment_item = doc.createElement('comment')
                comment_item_author = doc.createElement('author').appendChild(doc.createTextNode(str(comment['author'])))
                comment_item_time = doc.createElement('time').appendChild(doc.createTextNode(str(comment['time'])))
                comment_item_detail = doc.createElement('detail').appendChild(doc.createTextNode(str(comment['detail'])))
                comment_item.appendChild(comment_item_author)
                comment_item.appendChild(comment_item_time)
                comment_item.appendChild(comment_item_detail)
                comments_item.appendChild(comment_item)
        else:
            comments_item.appendChild(doc.createTextNode("no comment"))
        bug_item.appendChild(comments_item)

    # 将dom对象写入本地xml文件
    with open(filename, 'wb+') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
        f.close()

def get_domain_color(image):
    # returns: (R, G, B) tuple
    dominant = Haishoku.getDominant(image)
    color = "red"
    redsimi = math.sqrt((dominant[0] - 255)*(dominant[0] - 255) + dominant[1]*dominant[1] + dominant[2]*dominant[2])
    greensimi = math.sqrt(dominant[0]*dominant[0] + (dominant[1] - 255)*(dominant[1] - 255) + dominant[2]*dominant[2])
    if(greensimi < redsimi):
        color = "green"
    return color

if __name__ == "__main__":

    # log the precessing
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # if(len(sys.argv) < 3):
    #     logger.info("Usage: (int)Start number; (Str)Save dir")
    # else:
    #     begin = sys.argv[1]
    #     pathdir = sys.argv[2]

    for i in range(61, 2000):
        urllist = gentheurl(i, 500)
        resultlist = multithread(urllist, 10)
        path = "D:\\data\\eclipse\\bugreports\\bugs-" + str(i * 500 + 1) + "~" + str((i + 1) * 500) + ".xml"
        writeInfoToXml(resultlist, path)
        logger.info("Saved " + str(i * 500 + 1) + "~" + str((i + 1) * 500) + " to " + path)
        logger.info("Wait 3 minutes.")
        second = sleeptime(0, 3, 0);
        time.sleep(second)
        logger.info("Restart.")