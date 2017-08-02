#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import sys
import os
import logging

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print("usage:arg1:the model file, arg2: the target dict")
        sys.exit(1)
    inp_m, inp_w = sys.argv[1:3]

    # load the model to find the result
    model = gensim.models.Word2Vec.load(inp_m)
    result = model.most_similar(inp_w)
    for e in result:
        print(e[0], e[1])

