import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from src.util.util import ngram
from src.util.tfidf import tfTF_train, tfTF_test


def word_length(data):
    """ input: tokens"""
    print("word_length")
    m_word_l = []
    m_word_v = []
    sample_num = len(data)

    for essay in data:
        token_len = []
        for token in essay:
            token_len.append(len(token))

        m_w_l = np.mean(token_len)
        m_w_v = np.var(token_len)
        m_word_l.append(m_w_l)
        m_word_v.append(m_w_v)

    m_word_l = np.array(m_word_l).reshape(sample_num, 1)
    m_word_v = np.array(m_word_v).reshape(sample_num, 1)

    return m_word_l, m_word_v


"""
def mean_word_length(tk_essay):
    m_word_l = []
    for essay in tk_essay:
        token_len = []
        tokens = word_tokenize(essay)
        for token in tokens:
            token_len.append(len(token))
        m_w_l = np.mean(token_len)
        m_word_l.append(m_w_l)
    m_word_l = np.array(m_word_l)
    m_word_l.reshape(m_word_l.shape[0], 1)
    return m_word_l


def variance_word_length(essay_data):
    m_word_l = []
    for essay in essay_data:
        token_len = []
        tokens = word_tokenize(essay)
        for token in tokens:
            token_len.append(len(token))
        m_w_l = np.var(token_len)
        m_word_l.append(m_w_l)
    m_word_l = np.array(m_word_l)
    m_word_l.reshape(m_word_l.shape[0], 1)
    return m_word_l
"""


def get_sentence_length(data):
    """ input: sentences(token过后用join起来的)"""

    print("get_sentence_length")

    m_sentence_l = []
    m_sentence_v = []
    sample_num = len(data)

    for essay in data:
        sent_len_list = []
        sentences = sent_tokenize(essay)

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            word_num = len(tokens)
            sent_len_list.append(word_num)
        m_sent_l = np.mean(sent_len_list)
        m_sent_v = np.var(sent_len_list)
        m_sentence_l.append(m_sent_l)
        m_sentence_v.append(m_sent_v)

    m_sentence_l = np.array(m_sentence_l).reshape(sample_num, 1)
    m_sentence_v = np.array(m_sentence_v).reshape(sample_num, 1)

    return m_sentence_l, m_sentence_v


#
# def sentence_length(essay_data):
#     print("sentence_length")
#     sent_len = []
#     for essay in essay_data:
#         sent_len_list = []
#         sentences = sent_tokenize(essay)
#         for sentence in sentences:
#             words = word_tokenize(sentence)
#             word_num = len(words)
#             sent_len_list.append(word_num)
#         sent_len.append(sent_len_list)
#     sent_len = np.array(sent_len)
#     return sent_len

"""
def mean_sentence_length(essay_data):
    m_sentence_l = []
    for essay in essay_data:
        sent_len_list = []
        sentences = sent_tokenize(essay)
        for sentence in sentences:
            words = word_tokenize(sentence)
            word_num = len(words)
            sent_len_list.append(word_num)
        m_sent_l = np.mean(sent_len_list)
        m_sentence_l.append(m_sent_l)
    m_sentence_l = np.array(m_sentence_l)
    m_sentence_l.reshape(m_sentence_l.shape[0], 1)
    return m_sentence_l


def variance_sentence_length(essay_data):
    m_sentence_l = []
    for essay in essay_data:
        sent_len_list = []
        sentences = sent_tokenize(essay)
        for sentence in sentences:
            words = word_tokenize(sentence)
            word_num = len(words)
            sent_len_list.append(word_num)
        m_sent_l = np.var(sent_len_list)
        m_sentence_l.append(m_sent_l)
    m_sentence_l = np.array(m_sentence_l)
    m_sentence_l.reshape(m_sentence_l.shape[0], 1)
    return m_sentence_l
"""


def word_bigram_train(train_data):
    """ input: tokens"""

    print("word_bigram_train")

    gramed_data = ngram(train_data, 2)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data, word_ngram=True, gram_num=2)

    return train_tfTF, TF, tf_vocab


def word_bigram_test(test_data, TF, tf_vocab):
    """ input: tokens (已经tokenizer的)"""

    print("word_bigram_test")

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    gramed_data = ngram(test_data, 2)

    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab, word_ngram=True)

    return test_tfTF


def word_trigram_train(train_data):
    """ input: tokens"""
    print("word_trigram_train")

    gramed_data = ngram(train_data, 3)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data, word_ngram=True, gram_num=3)

    return train_tfTF, TF, tf_vocab


def word_trigram_test(test_data, TF, tf_vocab):
    """ input: tokens (已经tokenizer的)"""

    print("word_trigram_test")

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    gramed_data = ngram(test_data, 3)

    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab, word_ngram=True)

    return test_tfTF


def bag_of_words_train(train_data):
    """ input: tokens"""
    print("bag_of_words_train")

    gramed_data = ngram(train_data, 1)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data, word_ngram=True)

    return train_tfTF, TF, tf_vocab


def bag_of_words_test(test_data, TF, tf_vocab):
    """ input: tokens (已经tokenizer的)"""

    print("bag_of_words_test")

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    gramed_data = ngram(test_data, 1)

    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab, word_ngram=True)

    return test_tfTF
