#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7
import nltk
from nltk.book import *

with open("E:\\03_tools\\python\\nltk\\daqindiguo\\dqdg.txt", "r") as f:
    str = f.read()

    print(len(str))
    print(len(set(str)))

    print(str.count("秦"))
    print(str.count("大秦"))
    print(str.count("国"))
    fdist = FreqDist(str)
    fdist.plot()
