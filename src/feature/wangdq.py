# encoding=utf-8

import numpy as np

from src.util.tfidf import process_tfidf_data, tfidf_train, tfidf_test, tfTF_train, tfTF_test
from src.util.util import matrix_cosine_similarity, pos_tagging, ngram, constituency_tree

"""
输入分为两种： sequence or  tokens
"""


def word_vector_similarity_train(train_data, scores):
    """ input: tokens (已经tokenizer的)"""

    sample_num = scores.shape[0]

    train_data = process_tfidf_data(train_data)

    tfidf, tf_vocab, idf_diag = tfidf_train(train_data)
    cosine = matrix_cosine_similarity(tfidf)

    attn = scores.T * cosine
    np.fill_diagonal(attn, 0)
    sum_attn = np.sum(attn, 1)
    result = sum_attn / (sample_num - 1)
    return result.reshape(sample_num, 1), tf_vocab, idf_diag

def word_vector_similarity_test(test_data, train_score_list, tf_vocab, idf_diag):
    """ input: tokens (已经tokenizer的)"""

    assert idf_diag is not None, u"测试阶段，idf_diag不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    sample_num = train_score_list.shape[0]

    test_data = process_tfidf_data(test_data)

    tfidf = tfidf_test(test_data, tf_vocab, idf_diag)
    cosine = matrix_cosine_similarity(tfidf)

    attn = train_score_list.T * cosine
    np.fill_diagonal(attn, 0)
    sum_attn = np.sum(attn, 1)
    result = sum_attn / (sample_num - 1)
    return result.reshape(sample_num, 1)


def pos_bigram_train(train_data):
    """ input: tokens"""

    # 1. 词性标注
    tagged_data = pos_tagging(train_data)

    # 2. 组成2-gram
    gramed_data = ngram(tagged_data, 2)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data)

    return train_tfTF, TF, tf_vocab


def pos_bigram_test(test_data, TF, tf_vocab):
    """ input: tokens (已经tokenizer的)"""

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"
    # 1. 词性标注
    tagged_data = pos_tagging(test_data)

    # 2. 组成2-gram
    gramed_data = ngram(tagged_data, 2)

    # 3. 计算tfTF
    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab)

    return test_tfTF


def mean_clause(data):
    """
    train test使用 input: sentences
    """
    assert data is not None, u"data不能为none"

    clause_lengths, clause_nums = constituency_tree(data)

    # 暂时for处理了
    for i in range(len(clause_nums)):
        if clause_nums[i] == 0:
            clause_nums[i] = 1
    mean_clause_length = clause_lengths / clause_nums

    # TODO 需要肖肖的序列长度处理一下
    sample_num = len(data)
    return mean_clause_length.reshape(sample_num, 1), clause_nums.reshape(sample_num, 1)
