#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xlwt
import os,sys
import datetime
from pip._vendor.requests.packages.urllib3.util import url
from pip._vendor.packaging.requirements import URL
import urllib
import re
from html.parser import HTMLParser
from html.entities import name2codepoint
import src.Test1.app

class Log(object):
    def __init__(self,filepath):
        self.filepath = filepath
    def Logoutput(self):
        file = open(self.filepath,'r',encoding = 'utf-8')
        i = 0
        list = []
        line = file.readline()
        while line:    
            if i == 0:
                dict = {}
                dict['CommitID'] = line[7:-1]
                i += 1
            elif i == 1:
                if line[0:6] == 'Author':
                    dict['Author'] = line[8:-1]
                    i += 1
            elif i == 2:
                s = line[12:-7]
                c = datetime.datetime.strptime(s, '%b %d %H:%M:%S %Y')
                dict['Date'] = c.strftime('%Y-%m-%d %H:%M:%S')
                i += 1
            elif i == 3:
                i += 1
                s = '' 
            elif i == 4:
                if line[0:1] == ' ':
                    s += line
                    if line[4:7] == 'Bug':
                        dict['BugID'] = line[8:14]
                else:
                    i += 1
                    dict['Information'] = s
                    MfileList,AfileList,DfileList = [],[],[]
                    #fileList.append(line)
            elif i == 5:
                if line == '\n':    
                    dict['ChangedFile-M'] = MfileList
                    dict['ChangedFile-A'] = AfileList
                    dict['ChangedFile-D'] = DfileList
                    list.append(dict)
                    fileList = []
                    i = 0
                elif line[0:6] == 'commit':
                    dict['ChangedFile-M'] = MfileList
                    dict['ChangedFile-A'] = AfileList
                    dict['ChangedFile-D'] = DfileList
                    list.append(dict)
                    dict = {}
                    dict['CommitID'] = line[7:]
                    i = 1
                else:
                    if line[0:1] == 'M':
                        MfileList.append(line[2:-1])
                    elif line[0:1] == 'A':
                        AfileList.append(line[2:-1])
                    elif line[0:1] == 'D':
                        DfileList.append(line[2:-1])
                    
            line = file.readline()
        else:
            dict['ChangedFile-M'] = MfileList
            dict['ChangedFile-A'] = AfileList
            dict['ChangedFile-D'] = DfileList
            list.append(dict)
        
        file.close()
        return list
    
    def AuthorStatictics(self):
        AuthorData = {}
        list = self.Logoutput()
        for i in range(len(list)):
            dict = list[i]
            Authorname = dict.get('Author')
            if Authorname in AuthorData:
                j = AuthorData.get(Authorname)
                AuthorData [Authorname] = j + 1
            else:
                AuthorData [Authorname] = 1
        return(AuthorData)
    
    def ModifiedFile(self):
        FileChangeData = {}
        list = self.Logoutput()
        for i in range(len(list)):
            dict = list[i]
            FileList = dict.get('ChangedFile-M')
            for j in range(len(FileList)):
                FileName = FileList[j]
                if FileName in FileChangeData:
                    k = FileChangeData.get(FileName)
                    FileChangeData [FileName] = k + 1
                else:
                    FileChangeData [FileName] = 1
        return(FileChangeData)
    
    
    def BugList(self):
        BugData = {}
        BugData ['other'] = 0
        list = self.Logoutput()
        for i in range(len(list)):
            dict = list[i]
            BugNumber = dict.get('BugID')
            if BugNumber == None:
                j = BugData.get('other')
                BugData['other'] = j + 1
            elif BugNumber in BugData:
                j = BugData.get(BugNumber)
                BugData [BugNumber] = j + 1
            else:
                BugData [BugNumber] = 1
        return(BugData)
                
    def write_dict_to_excel(self,dict,path,name): 
        wbk = xlwt.Workbook()
        sheet = wbk.add_sheet('Sheet1',cell_overwrite_ok=True)
        sheet.write(0,0,name)
        sheet.write(0,1,'Number')
        m = 1
        for i in dict:
            sheet.write(m,0,i)
            sheet.write(m,1,dict[i])
            m = m + 1
        wbk.save('C:\\Users\\admin\\Desktop\\' + path + '\\' + name +'.xls')    
         
    def excels(self,path):
        os.makedirs('C:\\Users\\admin\\Desktop\\' + path + '\\')
        List = self.Logoutput()
        AuthorDict = self.AuthorStatictics()
        self.write_dict_to_excel(AuthorDict,path,'Author')
        FileDict = self.ModifiedFile()
        self.write_dict_to_excel(FileDict,path,'FileChange-M')
        BugDict = self.BugList()
        self.write_dict_to_excel(BugDict,path,'BugModify')               
            
    def FileDetail(self):
        list = self.Logoutput()
        ChangedFiledict = {}
        for dict in list:
            if dict.get('BugID'):
                ModifyFileList = dict['ChangedFile-M']
                for ModifyFile in ModifyFileList:
                    if ChangedFiledict.get(ModifyFile):
                        ChangedFiledict[ModifyFile]['commits'].append(dict['CommitID'])
                        ChangedFiledict[ModifyFile]['authors'].append(dict['Author'])
                        ChangedFiledict[ModifyFile]['dates'].append(dict['Date'])
                    else:
                        fileditc = {}
                        commit_list = [dict['CommitID'],]
                        author_list = [dict['Author'],]
                        date_list = [dict['Date'],]
                        fileDict = {'commits': commit_list, 'authors': author_list, 'dates': date_list}
                        ChangedFiledict[ModifyFile] = fileDict
        return(ChangedFiledict)
    
    def excels1(self,path):
        realpath = os.path.join('C:\\Users\\admin\\Desktop\\' + path +'\\')
        os.mkdir(realpath)
        dict = self.FileDetail()
        wbk = xlwt.Workbook()
        sheet = wbk.add_sheet('Sheet1',cell_overwrite_ok=True)
        sheet.write(0,0,'FileName')
        sheet.write(0,1,'commits')
        sheet.write(0,2,'authors')
        sheet.write(0,3,'dates')
        m = 1
        for dict1 in dict:
            sheet.write(m,0,dict1)
            dict2 = dict[dict1]
            list = dict2['commits']
            l = len(list)
            alignment = xlwt.Alignment()
            alignment.vert = xlwt.Alignment.VERT_TOP
            style = xlwt.XFStyle() # Create Style
            style.alignment = alignment # Add Alignment to Style
            sheet.write_merge(m,m+l-1,0,0,dict1,style)
            for j in range(l):
                sheet.write(m,1,dict2['commits'][j])
                sheet.write(m,2,dict2['authors'][j])
                sheet.write(m,3,dict2['dates'][j])
                m += 1
        wbk.save(realpath + 'jdt' +'.xls')


class MyHTMLParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        print('<%s>' % tag)

    def handle_endtag(self, tag):
        print('</%s>' % tag)

    def handle_startendtag(self, tag, attrs):
        print('<%s/>' % tag)

    def handle_data(self, data):
        print(data)

    def handle_comment(self, data):
        print('<!--', data, '-->')

    def handle_entityref(self, name):
        print('&%s;' % name)

    def handle_charref(self, name):
        print('&#%s;' % name)


class Githubissues(object):
    def __init__(self,url):
        self.url = url
    def geturl(self):
        i = 1
        s = set()
        html = self.url
        while i < 40:
            url = html + '?page=' + str(i) + '&q=is%3Aissue+is%3Aopen'
            print('正在抓取' + url)
            data = urllib.request.urlopen(url).read()
            data = data.decode('UTF-8')
            s1 = html[18:-1] + 's/'
            reg = r'href="' + s1 + r'(.+?)"'
            urlchoose = re.compile(reg)
            urllist = re.findall(urlchoose, data)
            for l in range(len(urllist)):
                j = urllist[l]
                urllist[l] = html + j
                s.add(urllist[l])
            i += 1
        return(s)
    
