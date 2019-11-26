import unittest
from src.data import Dataset
from src.feature.xiaoyl import *
from src.feature.wangdq import pos_bigram_train


def test():
    train_dataset: Dataset = Dataset.load("../../data/train.pickle")
    train_data=train_dataset.data['1']
    essay_data,token_data,scores = Dataset.get_data_list(train_data,acquire_score=False)
    result1,result2=word_length(essay_data)
    result3,result4,result5=word_bigram_train(token_data)

    result6, result7, result8=pos_bigram_train(token_data)
    print(result1)
    print(result1.shape)
    print(result2)
    print(result2.shape)
    print(result3)
    print(result3.shape)
    print(result4)
    print(result4.shape)
    print(result6)
    print(result6.shape)
    print(result7)
    print(result7.shape)
    #print(result5)
'''
class Test(unittest.TestCase):
    def test_mean_word_leangth(self):
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?'
        ]
        result=mean_word_length(corpus)
        print(result)

    def test_var_word_leangth(self):
        corpus = [
            'This is the first document.',
            'This is the second second document.',
            'And the third one.',
            'Is this the first document?'
        ]
        result=variance_word_length(corpus)
        print(result)

   # def test_mean_sentence_length(self):
'''
if __name__ == '__main__':
    test()