# encoding=utf-8
import unittest
import numpy as np
from src.feature.wangdq import word_vector_similarity_train
from src.feature.wangdq import word_vector_similarity_test
from util.util import tokenizer


class TestFeature(unittest.TestCase):

    def test_wv_similarity(self):
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?'
        ]

        corpus = tokenizer(corpus)
        score_list = np.array([2, 3, 4, 5]).reshape(4, 1)
        train_result, tf_vocab, idf_diag, tfidf = word_vector_similarity_train(corpus, score_list)

        test_corpus = [
            'This is the first document.',
            'hello And '
        ]
        test_corpus = tokenizer(test_corpus)
        word_vector_similarity_test(test_corpus, score_list, tf_vocab, idf_diag, tfidf)
