# encoding=utf-8
import os
import pickle

import numpy as np

from src.util.tfidf import process_tfidf_data, tfidf_train, tfidf_test, tfTF_train, tfTF_test
from src.util.util import matrix_cosine_similarity, ngram, constituency_tree, remove_stop_word

"""
输入分为两种： sequence or  tokens
"""


def word_vector_similarity_train(train_data, scores):
    """ input: tokens (已经tokenizer的)"""

    print("word_vector_similarity_train")

    sample_num = scores.shape[0]

    train_data = process_tfidf_data(train_data)

    tfidf, tf_vocab, idf_diag = tfidf_train(train_data)
    cosine = matrix_cosine_similarity(tfidf)

    attn = scores.T * cosine
    np.fill_diagonal(attn, 0)
    sum_attn = np.sum(attn, 1)
    result = sum_attn / (sample_num - 1)
    return result.reshape(sample_num, 1), tf_vocab, idf_diag, tfidf


def word_vector_similarity_test(test_data, train_score_list, tf_vocab, idf_diag, train_tfidf):
    """ input: tokens (已经tokenizer的)"""

    assert idf_diag is not None, u"测试阶段，idf_diag不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    print("word_vector_similarity_test")

    train_sample_num = train_score_list.shape[0]
    test_sample_num = len(test_data)
    # print(test_sample_num)

    test_data = process_tfidf_data(test_data)

    tfidf = tfidf_test(test_data, tf_vocab, idf_diag)
    cosine = matrix_cosine_similarity(tfidf, train_tfidf)

    attn = train_score_list.T * cosine
    np.fill_diagonal(attn, 0)
    sum_attn = np.sum(attn, 1)
    result = sum_attn / (train_sample_num - 1)
    return result.reshape(test_sample_num, 1)


def pos_gram_train(tagged_data, gram):
    """ input: tokens"""

    print("pos_bigram_train")

    # 2. 组成2-gram
    gramed_data = ngram(tagged_data, gram)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data, word_ngram=False, gram_num=gram)

    # print("pos",gram,train_tfTF.shape)

    return train_tfTF, TF, tf_vocab


def pos_gram_test(tagged_data, TF, tf_vocab, gram):
    """ input: tokens (已经tokenizer的)"""

    print("pos_bigram_test")

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    # 2. 组成2-gram
    gramed_data = ngram(tagged_data, gram)

    # 3. 计算tfTF
    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab, word_ngram=False)

    return test_tfTF


def mean_clause(data):
    """
    train test使用 input: sentences
    """
    assert data is not None, u"data不能为none"

    print("mean_clause")

    clause_lengths, clause_nums, sentences_num, ret_depth, ret_level = constituency_tree(data)

    mean_clause_num = clause_nums / sentences_num
    # 暂时for处理了
    for i in range(len(clause_nums)):
        if clause_nums[i] == 0:
            clause_nums[i] = 1

    mean_clause_length = clause_lengths / clause_nums

    sample_num = len(data)

    mean_clause_length = mean_clause_length.reshape(sample_num, 1)
    mean_clause_num = mean_clause_num.reshape(sample_num, 1)
    return mean_clause_length, mean_clause_num, ret_depth, ret_level


NGRAM_PATH = "../../data/good_pos_ngrams.p"


def good_pos_ngrams(tagged_data, gram=2):
    """
    input: tokens
    """
    print("good_pos_ngrams")

    if (os.path.isfile(NGRAM_PATH)):
        good_pos_ngrams = pickle.load(open(NGRAM_PATH, 'rb'))
    else:
        good_pos_ngrams = ['NN PRP', 'NN PRP .', 'NN PRP . DT', 'PRP .', 'PRP . DT', 'PRP . DT NNP', '. DT',
                           '. DT NNP', '. DT NNP NNP', 'DT NNP', 'DT NNP NNP', 'DT NNP NNP NNP', 'NNP NNP',
                           'NNP NNP NNP', 'NNP NNP NNP NNP', 'NNP NNP NNP .', 'NNP NNP .', 'NNP NNP . TO',
                           'NNP .', 'NNP . TO', 'NNP . TO NNP', '. TO', '. TO NNP', '. TO NNP NNP',
                           'TO NNP', 'TO NNP NNP']

    # 2. 组成2-gram
    gramed_data = ngram(tagged_data, gram, join_char=' ')

    correct_result = []
    uncorrect_result = []
    for essay in gramed_data:
        correct = 0
        uncorrect = 0
        for gram in essay:
            if gram in good_pos_ngrams:
                correct += 1
            else:
                uncorrect += 1

        correct_result.append(correct)
        uncorrect_result.append(uncorrect)

    return np.array(correct_result).reshape(-1, 1), np.array(uncorrect_result).reshape(-1, 1)


def pos_tagger(tagged_data, label):
    """
    input: tokens
    """
    print("pos_tagger")
    result = []

    for i in tagged_data:
        temp = 0
        for j in i:
            if j == label:
                temp += 1
        result.append(temp)
    return np.array(result).reshape(-1, 1)


def pos_tagger2(tagged_data, label):
    """
    input: tokens
    """
    print("pos_tagger2")
    result = []
    gramed_data = ngram(tagged_data, n=2, join_char='_')
    for i in gramed_data:
        temp = 0
        for j in i:
            if j == label:
                temp += 1
        result.append(temp)

    return np.array(result).reshape(-1, 1)


def vocab_size(data):
    """ input: tokens"""

    print('vocab_size')
    unique_len = []
    essay_len = []
    data = remove_stop_word(data)

    for essay in data:
        unique_len.append(len(set(essay)))
        essay_len.append(len(essay))

    unique_len = np.array(unique_len).reshape(-1, 1)
    essay_len = np.array(essay_len).reshape(-1, 1)
    return unique_len, unique_len / essay_len
