#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from haishoku.haishoku import Haishoku
import math
import requests

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

    print(get_domain_color(requests.get("https://dev.eclipse.org/site_login/web-api/cla_decorator.php?email=fortbild@streber24.de")))