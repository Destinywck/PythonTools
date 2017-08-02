#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import os, time, random
import sys
import logging
from multiprocessing import Process, Pool

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("usage:arg1:the target file, arg2: the result")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    k = 0

    # process the segment
    output = open(outp, 'w', encoding='utf-8')
    input = open(inp, 'r', encoding='utf-8')
    lines = len(input.readline())

    while(input.readline()):
        sentence_list = input.readline().split(' ')
        for i in sentence_list:
            seg_list = jieba.cut(i, cut_all=False)
            for j in seg_list:
                output.write(j)
                output.write(' ')
        output.write("\n")
        k = k + 1
        if (k % 10000 == 0):
            logger.info("Seged " + str(k) + " rows")
    output.close()
