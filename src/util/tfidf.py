# encoding=utf-8

"""
计算一些统计信息
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from src.util.util import tokenizer, word_stemming_remove_stop
import numpy as np


#
# class TFTF:
#     """
#     计算tf/TF
#     """
#
#     def __init__(self, train_data):
#         """ input: sequence"""
#         self.vectorizer = CountVectorizer()
#         self.train_tf = self.vectorizer.fit_transform(train_data).toarray()
#         self.TF = np.sum(self.train_tf, 0)
#
#     def get_train_TF(self):
#         return self.TF
#
#     def get_train_tfTF(self):
#         return self.train_tf / self.TF
#
#     def get_test_tfTF(self, test_data):
#         tf = self.vectorizer.transform(test_data)
#
#         return tf / (self.TF + tf)
#
#     def get_word(self):
#         return self.vectorizer.get_feature_names()
#
#
# class TFIDF:
#     """ 训练集初始化"""
#
#     def __init__(self, data):
#         self.vectorizer = CountVectorizer()
#         self.tf = self.vectorizer.fit_transform(data)  # 返回的是稀疏表示
#
#         self.transformer = TfidfTransformer()
#         tfidf = self.transformer.fit_transform(self.tf)
#         self.tfidf = tfidf.toarray()
#
#     def get_train_tfidf(self):
#         return self.tfidf
#
#     def get_idf(self):
#         return self.transformer.idf_
#
#     def get_train_tf(self):
#         return self.tf
#
#     def get_word(self):
#         return self.vectorizer.get_feature_names()
#
#     def get_test_tfidf(self, data):
#         """ 获取测试的tfidf """
#
#         # 不fit了，使用train data的
#         tf = self.vectorizer.transform(data)
#         tfidf = self.transformer.transform(tf)
#
#         return tfidf.toarray()
#
#     @staticmethod
#     def process_data(corpus):

#         corpus = word_stemming_remove_stop(corpus)
#         corpus = [' '.join(d) for d in corpus]
#         return corpus
#

def process_tfidf_data(corpus):
    """ 去除停用词，词形还原 input: tokens"""

    corpus = word_stemming_remove_stop(corpus)
    corpus = [' '.join(d) for d in corpus]
    return corpus


def tfidf_train(data):
    """ 计算tfidf input: sentences"""

    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(data)  # 返回的是稀疏表示

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tf)
    tfidf = tfidf.toarray()

    tf_vocab = vectorizer.get_feature_names()
    idf_diag = transformer.idf_

    return tfidf, tf_vocab, idf_diag


def tfidf_test(data, tf_vocab, idf_diag):
    """ input: sentences """
    vectorizer = CountVectorizer(vocabulary=tf_vocab)
    tf = vectorizer.transform(data)  # 返回的是稀疏表示

    transformer = TfidfTransformer()
    transformer.idf_ = idf_diag
    tfidf = transformer.transform(tf)
    tfidf = tfidf.toarray()

    return tfidf


def tfTF_train(data):
    """ 计算tfTF input: sentences"""
    vectorizer = CountVectorizer()
    train_tf = vectorizer.fit_transform(data).toarray()
    TF = np.sum(train_tf, 0)

    tf_vocab = vectorizer.get_feature_names()

    return train_tf / TF, TF, tf_vocab


def tfTF_test(data, TF, tf_vocab):
    """ 计算tfTF input: sentences"""
    vectorizer = CountVectorizer(vocabulary=tf_vocab)
    tf = vectorizer.transform(data).toarray()

    return tf / (TF + tf)
