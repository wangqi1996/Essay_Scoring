import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
from src.util.util import ngram
from src.util.tfidf import tfTF_train, tfTF_test
def mean_word_length(tk_essay):
    m_word_l=[]
    for essay in tk_essay:
        token_len=[]
        tokens=word_tokenize(essay)
        for token in tokens:
            token_len.append(len(token))
        m_w_l=np.mean(token_len)
        m_word_l.append(m_w_l)
    m_word_l=np.array(m_word_l)
    m_word_l.reshape(m_word_l.shape[0],1)
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

def sentence_length(essay_data):
    sent_len=[]
    for essay in essay_data:
        sent_len_list = []
        sentences=sent_tokenize(essay)
        for sentence in sentences:
            words=word_tokenize(sentence)
            word_num=len(words)
            sent_len_list.append(word_num)
        sent_len.append(sent_len_list)
    sent_len=np.array(sent_len)
    return sent_len

def mean_sentence_length(essay_data):
    m_sentence_l=[]
    for essay in essay_data:
        sent_len_list = []
        sentences=sent_tokenize(essay)
        for sentence in sentences:
            words=word_tokenize(sentence)
            word_num=len(words)
            sent_len_list.append(word_num)
        m_sent_l=np.mean(sent_len_list)
        m_sentence_l.append(m_sent_l)
    m_sentence_l=np.array(m_sentence_l)
    m_sentence_l.reshape(m_sentence_l.shape[0],1)
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


def word_bigram_train(train_data):
    """ input: tokens"""

    gramed_data = ngram(train_data, 2)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data)

    return train_tfTF, TF, tf_vocab

def word_bigram_test(test_data,TF, tf_vocab):
    """ input: tokens (已经tokenizer的)"""

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    gramed_data = ngram(test_data, 2)

    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab)

    return test_tfTF


def word_trigram_train(train_data):
    """ input: tokens"""

    gramed_data = ngram(train_data, 3)

    join_data = [' '.join(d) for d in gramed_data]

    train_tfTF, TF, tf_vocab = tfTF_train(join_data)

    return train_tfTF, TF, tf_vocab


def word_bigram_test(test_data, TF, tf_vocab):
    """ input: tokens (已经tokenizer的)"""

    assert TF is not None, u"测试阶段，TF不能为None"
    assert tf_vocab is not None, u"测试阶段，tf_vocab不能为None"

    gramed_data = ngram(test_data, 3)

    join_data = [' '.join(d) for d in gramed_data]
    test_tfTF = tfTF_test(join_data, TF, tf_vocab)

    return test_tfTF