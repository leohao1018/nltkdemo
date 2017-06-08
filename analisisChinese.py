#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author leo hao
# os windows 7

import pymysql
import jieba
import nltk
from nltk.tokenize.stanford_segmenter import StanfordSegmenter


# conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='1qaz@WSX', db='world')
#
# cur = conn.cursor()
# query = cur.execute("SELECT * FROM world.city")
#
# print("1")

def segment(sent):
    seg_list = jieba.cut(sent)  # 默认是精确模式
    tokens = list(seg_list)
    print("Full Mode: " + " ".join(tokens))
    return tokens


# 中文词性标注
def nerTagger(tokens):
    chi_tagger = nltk.StanfordNERTagger(
        model_filename=r'E:\03_tools\machine learning\stanfordnlp\3.7\stanford-chinese-corenlp-2016-10-31-models\edu\stanford\nlp\models\ner\chinese.misc.distsim.crf.ser.gz',
        path_to_jar=r'E:\03_tools\machine learning\stanfordnlp\3.7\stanford-ner-2016-10-31\stanford-ner.jar')
    for word, tag in chi_tagger.tag(tokens):
        print(word, tag)


# 中英文词性标注
def postTagger(tokens):
    chi_tagger = nltk.StanfordPOSTagger(
        model_filename=r'E:\03_tools\machine learning\stanfordnlp\3.7\stanford-chinese-corenlp-2016-10-31-models\edu\stanford\nlp\models\pos-tagger\chinese-distsim\chinese-distsim.tagger',
        path_to_jar=r'E:\03_tools\machine learning\stanfordnlp\3.7\stanford-postagger-full-2016-10-31\stanford-postagger.jar')

    print(chi_tagger.tag(tokens))


# 中英文句法分析
def parser(tokens):
    from nltk.parse.stanford import StanfordParser

    chi_parser = StanfordParser(
        r"E:\03_tools\machine learning\stanfordnlp\3.7\stanford-parser-full-2016-10-31\stanford-parser.jar",
        r"E:\03_tools\machine learning\stanfordnlp\3.7\stanford-parser-full-2016-10-31\stanford-parser-3.7.0-models.jar",
        r"E:\03_tools\machine learning\stanfordnlp\3.7\stanford-chinese-corenlp-2016-10-31-models\edu\stanford\nlp"
        r"\models\lexparser\chinesePCFG.ser.gz")
    print(list(chi_parser.parse(tokens)))


# 中文依存句法分析
def dependencyParser(tokens):
    from nltk.parse.stanford import StanfordDependencyParser
    chi_parser = StanfordDependencyParser(
        r"E:\03_tools\machine learning\stanfordnlp\3.7\stanford-parser-full-2016-10-31\stanford-parser.jar",
        r"E:\03_tools\machine learning\stanfordnlp\3.7\stanford-parser-full-2016-10-31\stanford-parser-3.7.0-models.jar",
        r"E:\03_tools\machine learning\stanfordnlp\3.7\stanford-chinese-corenlp-2016-10-31-models\edu\stanford\nlp"
        r"\models\lexparser\chinesePCFG.ser.gz")

    tree = chi_parser.parse(tokens)
    res = list(tree)
    for row in res[0].triples():
        print(row)


if __name__ == '__main__':
    sent = u'包邮德国米莱MILEL浓缩乳清蛋白粉wpc80健身健肌增肌粉保真'
    tokens = segment(sent)

    # nerTagger(tokens)

    postTagger(tokens)

    # parser(tokens)
    #
    dependencyParser(tokens)
