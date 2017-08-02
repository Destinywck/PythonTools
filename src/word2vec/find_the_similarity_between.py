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
    if len(sys.argv) < 4:
        print("usage:arg1:the model file, arg2: word1, arg3: word2")
        sys.exit(1)
    inp_m, inp_w1, inp_w2 = sys.argv[1:4]

    # load the model to find the result
    model = gensim.models.Word2Vec.load(inp_m)
    result = model.similarity(inp_w1, inp_w2)
    print("the similarity between ", inp_w1, "and ", inp_w2, "is: ", result)

