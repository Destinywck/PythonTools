#!/usr/bin/env python
# -*- coding: utf-8 -*-
# process_wiki_data.py 用于解析XML，将XML的wiki数据转换为text格式

import logging
import os.path
import sys
import warnings
import re
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import WikiCorpus
from langconv import *


# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line

# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("usage:arg1:the target file, arg2: the result")
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    # process the dataset
    space = ""
    j = 0
    output = open(outp, 'w', encoding='utf-8')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        for i in text:
            #print(i)
            #i = cht_to_chs(i)
            output.write(i.decode('utf-8'))
            output.write(" ")
        output.write("\n")
        j = j + 1
        if (j % 10000 == 0):
            logger.info("Saved " + str(j) + " articles")
    output.close()
    logger.info("Finished Saved " + str(j) + " articles")
