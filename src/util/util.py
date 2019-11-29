# encoding=utf-8

"""
计算一些统计信息
"""
import logging

import nltk
import numpy
from nltk.corpus import stopwords

stoplist = stopwords.words('english')

from stanfordcorenlp import StanfordCoreNLP

from src.config import STANFORDCORENLP_PATH

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from nltk.tree import *
import config


def remove_stop_word(data):
    """ remove stop word """

    result = [[token for token in sample if token not in stoplist] for sample in data]
    return result


def word_stemming(data):
    """ word stemming """

    stemmer = PorterStemmer()

    result = [[stemmer.stem(token) for token in sample] for sample in data]
    return result


def word_stemming_remove_stop(data):
    """ word stemming and remove stop word """

    stemmer = PorterStemmer()
    result = [[stemmer.stem(token) for token in sample if token not in stoplist] for sample in data]
    return result


def tokenizer(data):
    result = [word_tokenize(d) for d in data]
    return result


def vector_cosine_similarity(vector1, vector2):
    """ compute the cosine similarity """

    assert vector1 and vector2, u"vector_cosine_similarity中, vector1和vector2不能为none"

    result = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))

    return result


def matrix_cosine_similarity(matrix1, matrix2=None):
    """ 为矩阵的元素两两计算余弦相似度 """

    assert matrix1 is not None, u"matrix_cosine_similarity中, matrix1不能为none"

    if matrix2 is None:
        matrix2 = matrix1

    cosine = cosine_similarity(matrix1, matrix2)
    return cosine


def pos_tagging(train_data):
    """ 词性标注 input:tokens"""
    result = []
    for sample in train_data:
        pos_tag = nltk.pos_tag(sample)
        result.append([_tag[1] for _tag in pos_tag])
    return result


def ngram(train_data, n=2, join_char='_'):
    """ n-gram """
    result = []
    for sample in train_data:
        bi_sample = nltk.ngrams(sample, n)
        result.append([join_char.join(s) for s in bi_sample])

    return result


clause_num = 0
clause_length = 0


def constituency_tree(train_data):
    """ 计算mean clause使用
     input: sentences"""

    nlp = StanfordCoreNLP(STANFORDCORENLP_PATH)
    clause_nums = []
    clause_lengths = []
    sentences_num = []
    mean_depth = []
    mean_level = []
    for essay in train_data:

        global clause_num
        global clause_length

        clause_num = 0
        clause_length = 0

        depth_all = 0
        level_all = 0
        # 对每句话进行处理
        sentences = get_sentences(essay)
        sentences_num.append(len(sentences))

        for sentence in sentences:

            tokens = word_tokenize(sentence)
            if len(tokens) > 80:
                continue
            constituency_str = nlp.parse(sentence)
            tree = Tree.fromstring(constituency_str)

            # tree.draw()
            distance = []
            traverse_nltk_tree(tree, 'SBAR', 1, distance)

            distance = numpy.array(distance)
            depth = numpy.sum(distance)
            level = numpy.amax(distance)
            depth_all += depth
            level_all += level

        mean_depth.append(depth_all / len(sentences))
        mean_level.append(level_all / len(sentences))

        clause_nums.append(clause_num)
        clause_lengths.append(clause_length)

    nlp.close()

    clause_lengths = np.array(clause_lengths).reshape(-1, 1)
    clause_nums = np.array(clause_nums).reshape(-1, 1)
    sentences_num = np.array(sentences_num).reshape(-1, 1)
    ret_depth = numpy.array(mean_depth).reshape(-1, 1)
    ret_level = numpy.array(mean_level).reshape(-1, 1)

    return clause_lengths, clause_nums, sentences_num, ret_depth, ret_level


def traverse_nltk_tree(node, label, length, res):
    """ 遍历nltk树，找子树 """
    if not node:
        # 说明是叶子节点
        return

    if type(node) == str:
        res.append(length)
        return

    global clause_num
    global clause_length
    # 非叶子节点
    if node.label() == label:
        temp_sent = node.leaves()
        # print(temp_sent)
        clause_num += 1
        clause_length += len(temp_sent)

    # 遍历所有的孩子节点
    node_num = len(node)
    for i in range(node_num):
        if node[i]:
            traverse_nltk_tree(node[i], label, length + 1, res)
    return


def get_sentences(essay):
    """ 将文章分成多句话
    测试发现，必须得是tokenizer之后的sentences"""
    result = nltk.sent_tokenize(essay)
    return result
