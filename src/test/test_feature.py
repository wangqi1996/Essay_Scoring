# encoding=utf-8
import unittest

from src.feature import word_vector_similarity


class TestFeature(unittest.TestCase):

    def test_wv_similarity(self):
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?'
        ]

        score_list = [2, 3, 4, 5]
        word_vector_similarity(corpus, score_list)
