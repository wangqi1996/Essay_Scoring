# encoding=utf-8
from src.data import Dataset
import numpy as np
#
from src.feature.feature1 import word_vector_similarity_train, pos_bigram_train, word_vector_similarity_test, \
    mean_clause, pos_bigram_test


class Feature:
    """
    我也不知道定义类还是函数！！！
    """

    def __init__(self):
        # 应该放在dataset中懒得放了
        self.wv_tf_vocab = None
        self.wv_idf_diag = None

        self.pos_tf_vocab = None
        self.pos_TF = None

    def append_feature(self, *feature):
        return np.concatenate(feature, axis=1)

    def get_feature(self, sentences_set, token_set):
        """ 不用区分训练测试的, 可以写在这类"""

        mean_clause_length, mean_clause_number = mean_clause(sentences_set)

        feature = self.append_feature(mean_clause_length, mean_clause_number)
        return feature

    def get_train_feature(self, sentences_list, tokens_list, scores):
        """ 获取training的feature
        sentences_array, tokens_array 均已经tokenizer处理过
        所有返回的特征的维度: ndarray类型 sample_num * feature_fim
        """
        feature = self.get_feature(sentences_list, tokens_list)

        wv_similarity, wv_tf_vocab, wv_idf_diag = word_vector_similarity_train(tokens_list, scores)
        self.wv_idf_diag = wv_idf_diag
        self.wv_tf_vocab = wv_tf_vocab

        # [*, *(TODO 是否需要裁剪)]
        pos_bigram, pos_TF, pos_tf_vocab = pos_bigram_train(tokens_list)
        self.pos_tf_vocab = pos_tf_vocab
        self.pos_TF = pos_TF

        feature = self.append_feature(feature, wv_similarity, pos_bigram)
        # print(feature)
        # print(feature.shape)
        return feature

    def get_test_feature(self, sentences_list, tokens_list, train_score):
        """ """
        feature = self.get_feature(sentences_list, tokens_list)

        wv_similarity = word_vector_similarity_test(tokens_list, train_score, self.wv_tf_vocab, self.wv_idf_diag)

        pos_bigram = pos_bigram_test(tokens_list, self.pos_TF, self.pos_tf_vocab)

        # print(feature)
        # print(feature.shape)
        feature = self.append_feature(feature, wv_similarity, pos_bigram)
        return feature
