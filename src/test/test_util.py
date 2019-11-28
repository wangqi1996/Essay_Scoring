# encoding=utf-8

import unittest

from feature.wangdq import good_pos_ngrams, vocab_size
from feature.xiaoyl import word_length, get_sentence_length
from src.util.tfidf import process_tfidf_data, tfidf_train, tfidf_test
from src.util.util import word_stemming_remove_stop, tokenizer, pos_tagging, ngram, constituency_tree, get_sentences


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
        corpus = tokenizer(corpus)
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
            'Computers do  have any affect on kids we just love going on cause we use it for help and this persuade the readers of the local newspaper cause we need to be able to communicate also do writing essays and doing social studies or science homework my ideas are let us go computers cause were not bothering u can just leave us alone and let us do what you need to do cause what computers are what give us information for we have to do and were to do wat we got ta do and u people can just leave us alone cause are nt addicting to me or anyone and if we were it still would it matter cause a computers a computer u do nt punish it because just punish us from the computer punish us because of it cause its the computer fault it can be addicting cause the computer is device that ']

        result = constituency_tree(corpus)
        print(result)

        return result

    def test_get_sentences(self):
        corpus = [
            'A . B.C'
        ]
        result = get_sentences(corpus[0])

        return result

    def test_word_length(self):
        corpus = [
            'Computers do  have any affect on kids we just love going on cause we use it for help and this persuade the readers of the local newspaper cause we need to be able to . communicate also do writing essays and doing social studies or science . homework my ideas are let us go computers cause were not bothering u can just leave us alone and let us do what you need to do cause what computers are what give us information for we have to do and were to do wat we got ta do and u people can just leave us alone cause are nt addicting to me or anyone and if we were it still would it matter cause a computers a computer u do nt punish it because just punish us from the computer punish us because of it cause its the computer fault it can be addicting cause the computer . is device that ']

        result = word_length(tokenizer(corpus))
        print(result)

        result = get_sentence_length(corpus)
        print(result)

    def test_good_pos_ngrams(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?',
            'a a a a'
        ]

        corpus = tokenizer(corpus)
        print(corpus)

        data = good_pos_ngrams(corpus)
        print(data)


    def test_vocab_size(self):
        corpus = [
            'This is the first document. the a',
            'This is the second seconds document.',
            'And the third one. fying',
            'Is this the first documents?',
            'a  a a a'
        ]

        corpus = tokenizer(corpus)
        print(corpus)

        result = vocab_size(corpus)
        print(result)
if __name__ == '__main__':
    TestUtil().test_pos_tag()