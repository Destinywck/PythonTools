#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _dummy_thread import error
'''print ('hello python1!')
from numpy import *
a=array([1,2,3])
b=array([4,5,6])
print(a+b)
print('Hello world!')

print("{name} wants to eat {food}".format(name="Bob", food="lasagna"))
print(0==False)

some_var=5
some_other_var=5

print("yahoo!" if 3<2 else "next")

li=[]
other_li=[4,5,6]
print(
    other_li)
li.append(1)
li.append(2)
li.append(3)
print(li)
li.append(5)
li.append(4)
print(li)
li.pop()
print(li)

print(li+other_li)
print(li,other_li
      )

li.extend(other_li)
print(li)

print(1 in li,len(li))

li[1]=8
print(li)

filled_dict={"one":1,"two":2,"three":3}
print(filled_dict["three"])

#name=input('please input you name:')
#print('hello',name)

print(0==False)  #=>True
print(''==False)  #=>False

print(10/3)
print(10//3)

print(ord('A'))    #对于单个字符的编码，Python提供了ord()函数获取字符的整数表示，chr()函数把编码转换为对应的字符：
print(ord('中'),'国')
print(chr(66))
print(chr(25991))

t1=(1)         #只有1个元素的tuple定义时必须加一个逗号,，来消除歧义
t2=(1,)
print(t1,t2)

def fact(n):
    if n == 1:
        return 1
    return n * fact(n - 1)
print(fact(1000))
'''
def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    body = '<h1>Hello, %s!<h1>' % (environ['PATH_INFO'][1:] or 'web')
    return [body.encode(encoding='utf_8')]



