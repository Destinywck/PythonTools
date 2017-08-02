'''
Created on 2017年2月23日

@author: admin
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''from PIL import Image
im = Image.open('test.png')
print(im.format, im.size, im.mode)
im.thumbnail((200, 100))
im.save('thumb.jpg', 'JEPG')'''
import os
import shutil
import urllib.request
from urllib import request
import requests
from bs4 import BeautifulSoup
import re
from collections import deque

'''try:
    print('try...')
    r = 10 / 0
    print('result:', r)
except ZeroDivisionError as e:
    print('except:', e)
finally:
    print('finally...')
print('END')

try:
    print('try...')
    r = 10 / 0
    print('result:', r)
except ValueError as e:
    print('ValueError:', e)
except ZeroDivisionError as e:
    print('ZeroDivisionError:', e)
else:
    print('no error!')
finally:
    print('finally...')
print('END')

print(os.name)
print(os.environ)
print(os.environ.get('PATH'))


url = "http://www.baidu.com"
data = urllib.request.urlopen(url).read()
data = data.decode('UTF-8')
print(data)

def getHtml(url):
    html = urllib.request.urlopen(url).read().decode('utf-8')
    return html

def getImg(html):
    reg = r'src="(.+?\.jpg)"'
    imgre = re.compile(reg)
    imglist = re.findall(imgre,html)
    x = 0
    for imgurl in imglist:
        path = 'C:\\Users\\admin\\Desktop\\jdt2\\' + str(x) + '.jpg'
        urllib.request.urlretrieve(imgurl,path)
        x += 1 
        print(imgurl)
html = getHtml('http://www.ivsky.com/tupian/xiaohuangren_t21343/')
print(html)
getImg(html)
'''
'''
queue = deque()
visited = set()
 
url = 'https://github.com/docker/docker/issues?page=2&q=is%3Aissue+is%3Aopen'  # 入口页面, 可以换成别的
 
queue.append(url)
cnt = 0
 
while queue:
    url = queue.popleft()  # 队首元素出队
    visited |= {url}  # 标记为已访问
 
    print('已经抓取: ' + str(cnt) + '   正在抓取 <---  ' + url)
    cnt += 1
    urlop = urllib.request.urlopen(url)
    if 'html' not in urlop.getheader('Content-Type'):
        continue
 
    # 避免程序异常中止, 用try..catch处理异常
    try:
        data = urlop.read().decode('utf-8')
    except:
        continue
 
    # 正则表达式提取页面中所有队列, 并判断是否已经访问过, 然后加入待爬队列
    linkre = re.compile('href=\"(.+?)\"')
    for x in linkre.findall(data):
        if 'http' in x and x not in visited:
            queue.append(x)
            print('加入队列 --->  ' + x)

'''
# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
from hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()



