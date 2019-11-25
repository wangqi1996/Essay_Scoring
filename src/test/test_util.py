# encoding=utf-8

import unittest

from src.util.tfidf import process_tfidf_data, tfidf_train, tfidf_test
from src.util.util import word_stemming_remove_stop, tokenizer, pos_tagging, ngram, constituency_tree


class TestUtil(unittest.TestCase):

    # def test_get_tfidf(self):
    #     corpus = [
    #         'This is the first document.',
    #         'This is the second second document.',
    #         'And the third one.',
    #         'Is this the first document?'
    #     ]
    #     corpus = TFIDF.process_data(corpus)
    #     print(corpus)
    #
    #     tfidf_class = TFIDF(corpus)
    #
    #     tfidf1 = tfidf_class.get_train_tfidf()
    #     word = tfidf_class.get_word()
    #     idf = tfidf_class.get_idf()
    #     tf = tfidf_class.get_train_tf()
    #     # print(tfidf1)
    #
    #     # print('---------------')
    #     # print(word)
    #     # print('----------------')
    #     # print(idf)
    #     # print('-----------------')
    #     # print(tf)
    #
    #     test = [
    #         'first first document',
    #         'This is the first document.',
    #         'second Is one'
    #     ]
    #
    #     test = TFIDF.process_data(test)
    #     print(test)
    #
    #     tfidf2 = tfidf_class.get_test_tfidf(test)
    #     print(tfidf2)

    def test_get_tfidf2(self):
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?'
        ]
        corpus = process_tfidf_data(corpus)
        print(corpus)

        tfidf3, tf_vocab, idf_diag = tfidf_train(corpus)
        print(tfidf3)

        test = [
            'first first document',
            'This is the first document.',
            'second Is one'
        ]

        test = process_tfidf_data(test)
        print(test)

        tfidf4 = tfidf_test(test, tf_vocab, idf_diag)
        print(tfidf4)

        # print(tfidf1 == tfidf3)
        # print(tfidf2 == tfidf4)
        # print(word == tf_vocab)
        # print(idf == idf_diag)

    def test_word_stemming_remove_stop(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?'
        ]

        corpus = tokenizer(corpus)
        print(corpus)

        data = word_stemming_remove_stop(corpus)
        print(data)

    def test_pos_tag(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?'
        ]

        corpus = tokenizer(corpus)
        # print(corpus)

        data = pos_tagging(corpus)
        return data

    def test_bigram(self):
        data = self.test_pos_tag()

        data = ngram(data)

        return data

    def test_get_tftf(self):
        data = self.test_pos_tag()

        data = ngram(data)
        data = [' '.join(d) for d in data]
        print(data)

        TFTF_class = TFTF(data)
        print(TFTF_class.get_word())
        print(TFTF_class.get_train_tfTF())
        print(TFTF_class.get_train_TF())

        test = [
            'first first document',
            'This is the first document.',
            'second Is one'
        ]
        test = tokenizer(test)
        # rint(test)

        test = pos_tagging(test)
        test = ngram(test)
        test = [' '.join(d) for d in test]
        print(test)

        print(TFTF_class.get_test_tfTF(test))
        print(TFTF_class.get_train_TF())

    def test_constituenty_tree(self):
        corpus = [
            'ABC cites the fact that chemical additives are banned in many countries and feels they may be banned in this state too ',
            'ABC cites the fact that chemical additives are banned in many countries and feels they may be banned in this state too ',
            'ABC cites the fact that chemical additives are banned in many countries and feels they may be banned in this state too and I hava and i i i'
        ]

        result = constituency_tree(corpus)
        print(result)

        return result
