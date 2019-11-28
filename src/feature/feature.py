# encoding=utf-8
from src.feature.xiaoyl import word_length, get_sentence_length, word_bigram_train, word_bigram_test, \
    word_trigram_train, \
    word_trigram_test
from src.data import Dataset
import numpy as np
#
from src.feature.wangdq import word_vector_similarity_train, pos_bigram_train, word_vector_similarity_test, \
    mean_clause, pos_bigram_test
from src.feature.iku import spell_error, Mean_sentence_depth_level, essay_length, semantic_vector_similarity
from src.config import feature_list

from sklearn import preprocessing
class Feature:
    """
    我也不知道定义类还是函数！！！
    """

    def __init__(self):
        # 应该放在dataset中懒得放了
        # 禁止从这个类初始化
        self.wv_tf_vocab = None
        self.wv_idf_diag = None

        self.wv_similarity = None
        self.pos_bigram = None
        self.word_bigram = None
        self.word_trigram = None

        self.mean_clause_length = None
        self.mean_clause_number = None

        self.pos_tf_vocab = None
        self.pos_TF = None

        self.mean_word_length = None
        self.var_word_length = None
        self.mean_sentence_length = None
        self.var_sentence_length = None

        self.word_bigram_TF = None
        self.word_bigram_tf_vocab = None
        self.word_trigram_TF = None
        self.word_trigram_tf_vocab = None

        self.spell_error = None
        self.mean_sentence_depth = None
        self.mean_sentence_level = None
        self.essay_length = None
        self.semantic_vector_similarity = None

        self.train_feature = None

    @staticmethod
    def get_instance(feature):
        """ 从dataset中构建dataset
        feature: {} """
        feature_class = Feature()

        feature_class.wv_idf_diag = feature.get('wv_idf_diag', None)
        feature_class.wv_tf_vocab = feature.get('wv_tf_vocab', None)

        feature_class.wv_similarity = feature.get('wv_similarity', None)
        feature_class.pos_bigram = feature.get('pos_bigram', None)
        feature_class.word_bigram = feature.get('word_bigram', None)
        feature_class.word_trigram = feature.get('word_trigram', None)

        feature_class.mean_clause_length = feature.get('mean_clause_length', None)
        feature_class.mean_clause_number = feature.get('mean_clause_number', None)

        feature_class.pos_tf_vocab = feature.get('pos_tf_vocab', None)
        feature_class.pos_TF = feature.get('pos_TF', None)

        feature_class.mean_word_length = feature.get('mean_word_length', None)
        feature_class.var_word_length = feature.get('var_word_length', None)
        feature_class.mean_sentence_length = feature.get('mean_sentence_length', None)
        feature_class.var_sentence_length = feature.get('var_sentence_length', None)

        feature_class.word_bigram_tf_vocab = feature.get('word_bigram_tf_vocab', None)
        feature_class.word_bigram_TF = feature.get('word_bigram_TF', None)
        feature_class.word_trigram_tf_vocab = feature.get('word_trigram_tf_vocab', None)
        feature_class.word_trigram_TF = feature.get('word_trigram_TF', None)

        feature_class.spell_error = feature.get('spell_error', None)
        feature_class.mean_sentence_depth = feature.get('mean_sentence_depth', None)
        feature_class.mean_sentence_level = feature.get('mean_sentence_level', None)
        feature_class.essay_length = feature.get('essay_length', None)
        feature_class.semantic_vector_similarity = feature.get('semantic_vector_similarity', None)

        # 一个array数组
        feature_class.train_feature = feature.get('train_feature', None)
        return feature_class

    def save_feature(self, train_feature):
        """ 保存feature和中间需要使用的变量"""

        result = {
            "wv_tf_vocab": self.wv_tf_vocab,
            "wv_idf_diag": self.wv_idf_diag,
            "pos_tf_vocab": self.pos_tf_vocab,
            "pos_TF": self.pos_TF,

            "wv_similarity": self.wv_similarity,
            "pos_bigram": self.pos_bigram,
            "word_bigram": self.word_bigram,
            "word_trigram": self.word_trigram,

            "mean_clause_length": self.mean_clause_length,
            "mean_clause_number": self.mean_clause_number,

            "mean_word_length": self.mean_word_length,
            "var_word_length": self.var_word_length,
            "mean_sentence_length": self.mean_sentence_length,
            "var_sentence_length": self.var_sentence_length,

            "word_bigram_tf_vocab": self.word_bigram_tf_vocab,
            "word_bigram_TF": self.word_bigram_TF,
            "word_trigram_tf_vocab": self.word_trigram_tf_vocab,
            "word_trigram_TF": self.word_trigram_TF,

            "spell_error": self.spell_error,
            "mean_sentence_depth": self.mean_sentence_depth,
            "mean_sentence_level": self.mean_sentence_level,
            "essay_length": self.essay_length,
            "semantic_vector_similarity": self.semantic_vector_similarity,

            "train_feature": train_feature
        }

        return result

    def append_feature(self, *feature):
        return np.concatenate(feature, axis=1)

    def concatenate_feature(self, feature, new_feature):
        if feature is None:
            return new_feature
        else:
            return np.concatenate((feature, new_feature), axis=1)

    def get_feature(self, sentences_set, token_set, train_data):
        """ 不用区分训练测试的, 可以写在这类"""
        feature = None

        if 'mean_clause_length' in feature_list or 'mean_clause_number' in feature_list:
            mean_clause_length, mean_clause_number = mean_clause(sentences_set)
            self.mean_clause_number = mean_clause_number
            self.mean_clause_length = mean_clause_length
            if 'mean_clause_length' in feature_list:
                feature = self.concatenate_feature(feature, mean_clause_length)
            if 'mean_clause_number' in feature_list:
                feature = self.concatenate_feature(feature, mean_clause_number)

        if 'mean_word_length' in feature_list or 'var_word_length' in feature_list:
            mean_word_length, var_word_length = word_length(token_set)
            self.mean_word_length = mean_word_length
            self.var_word_length = var_word_length
            if 'mean_word_length' in feature_list:
                feature = self.concatenate_feature(feature, mean_word_length)
            if 'var_word_length' in feature_list:
                feature = self.concatenate_feature(feature, var_word_length)

        if 'mean_sentence_length' in feature_list or 'var_sentence_length' in feature_list:
            mean_sentence_length, var_sentence_length = get_sentence_length(sentences_set)
            self.mean_sentence_length = mean_sentence_length
            self.var_sentence_length = var_sentence_length
            if 'mean_sentence_length' in feature_list:
                feature = self.concatenate_feature(feature, mean_sentence_length)
            if 'var_sentence_length' in feature_list:
                feature = self.concatenate_feature(feature, var_sentence_length)

        if 'spell_error' in feature_list:
            error = spell_error(train_data)
            self.spell_error = error
            feature = self.concatenate_feature(feature, error)

        if 'mean_sentence_depth' in feature_list or 'mean_sentence_level' in feature_list:
            depth, level = Mean_sentence_depth_level(train_data)
            self.mean_sentence_level = level
            self.mean_sentence_depth = depth
            if 'mean_sentence_depth' in feature_list:
                feature = self.concatenate_feature(feature, depth)
            if 'mean_sentence_level' in feature_list:
                feature = self.concatenate_feature(feature, level)

        if 'essay_length' in feature_list:
            length = essay_length(train_data)
            self.essay_length = length
            feature = self.concatenate_feature(feature, length)

        # feature = self.append_feature(mean_clause_length, mean_clause_number, mean_word_length, var_word_length,
        #                               mean_sentence_length, var_sentence_length)


        return feature

    def get_save_feature(self):
        feature = None
        if 'mean_clause_length' in feature_list:
            feature = self.concatenate_feature(feature, self.mean_clause_length)
        if 'mean_clause_number' in feature_list:
            feature = self.concatenate_feature(feature, self.mean_clause_number)

        if 'mean_word_length' in feature_list:
            feature = self.concatenate_feature(feature, self.mean_word_length)
        if 'var_word_length' in feature_list:
            feature = self.concatenate_feature(feature, self.var_word_length)

        if 'mean_sentence_length' in feature_list:
            feature = self.concatenate_feature(feature, self.mean_sentence_length)
        if 'var_sentence_length' in feature_list:
            feature = self.concatenate_feature(feature, self.var_sentence_length)

        if 'spell_error' in feature_list:
            feature = self.concatenate_feature(feature, self.spell_error)

        if 'mean_sentence_depth' in feature_list:
            feature = self.concatenate_feature(feature, self.mean_sentence_depth)
        if 'mean_sentence_level' in feature_list:
            feature = self.concatenate_feature(feature, self.mean_sentence_length)

        if 'essay_length' in feature_list:
            feature = self.concatenate_feature(feature, self.essay_length)

        return feature


    def get_train_feature(self, sentences_list, tokens_list, scores, train_data):
        """ 获取training的feature
        sentences_array, tokens_array 均已经tokenizer处理过
        所有返回的特征的维度: ndarray类型 sample_num * feature_fim
        """

        # if self.train_feature is not None:
        #     return self.train_feature

        feature = self.get_feature(sentences_list, tokens_list, train_data)

        # feature = self.append_feature(feature, wv_similarity, pos_bigram)
        if 'wv_similarity' in feature_list:
            wv_similarity, wv_tf_vocab, wv_idf_diag = word_vector_similarity_train(tokens_list, scores)
            self.wv_similarity = wv_similarity
            self.wv_idf_diag = wv_idf_diag
            self.wv_tf_vocab = wv_tf_vocab
            feature = self.concatenate_feature(feature, self.wv_similarity)
        if 'pos_bigram' in feature_list:
            pos_bigram, pos_TF, pos_tf_vocab = pos_bigram_train(tokens_list)
            self.pos_bigram = pos_bigram
            self.pos_tf_vocab = pos_tf_vocab
            self.pos_TF = pos_TF
            feature = self.concatenate_feature(feature, pos_bigram)
        if 'word_bigram' in feature_list:
            word_bigram, word_bigram_TF, word_bigram_tf_vocab = word_bigram_train(tokens_list)
            self.word_bigram = word_bigram
            self.word_bigram_tf_vocab = word_bigram_tf_vocab
            self.word_bigram_TF = word_bigram_TF
            feature = self.concatenate_feature(feature, word_bigram)
        if 'word_trigram' in feature_list:
            word_trigram, word_trigram_TF, word_trigram_tf_vocab = word_trigram_train(tokens_list)
            self.word_trigram = word_trigram
            self.word_trigram_tf_vocab = word_trigram_tf_vocab
            self.word_trigram_TF = word_trigram_TF
            feature = self.concatenate_feature(feature, word_trigram)
        if 'semantic_vector_similarity' in feature_list:
            semvec_sim = semantic_vector_similarity(train_data, train_data)
            self.semantic_vector_similarity = semvec_sim
            feature = self.concatenate_feature(feature, semvec_sim)

        # feature = preprocessing.normalize(feature, norm='l1', axis=0)
        self.train_feature = feature

        # print(feature)
        # print('train_feature', feature.shape)
        return feature

    def get_save_train_feature(self):
        feature = self.get_save_feature()

        if 'wv_similarity' in feature_list:
            feature = self.concatenate_feature(feature, self.wv_similarity)
        if 'pos_bigram' in feature_list:
            feature = self.concatenate_feature(feature, self.pos_bigram)
        if 'word_bigram' in feature_list:
            feature = self.concatenate_feature(feature, self.word_bigram)
        if 'word_trigram' in feature_list:
            feature = self.concatenate_feature(feature, self.word_trigram)
        if 'semantic_vector_similarity' in feature_list:
            feature = self.concatenate_feature(feature, self.semantic_vector_similarity)

        return feature




    def get_test_feature(self, sentences_list, tokens_list, train_score, train_data, test_data):
        """ """
        feature = self.get_feature(sentences_list, tokens_list, test_data)
        #####################
        # feature = self.append_feature(feature, wv_similarity, pos_bigram, word_bigram, word_trigram)
        if 'wv_similarity' in feature_list:
            wv_similarity = word_vector_similarity_test(tokens_list, train_score, self.wv_tf_vocab, self.wv_idf_diag)
            feature = self.concatenate_feature(feature, wv_similarity)
        if 'pos_bigram' in feature_list:
            pos_bigram = pos_bigram_test(tokens_list, self.pos_TF, self.pos_tf_vocab)
            feature = self.concatenate_feature(feature, pos_bigram)
        if 'word_bigram' in feature_list:
            word_bigram = word_bigram_test(tokens_list, self.word_bigram_TF, self.word_bigram_tf_vocab)
            feature = self.concatenate_feature(feature, word_bigram)
        if 'word_trigram' in feature_list:
            word_trigram = word_trigram_test(tokens_list, self.word_trigram_TF, self.word_trigram_tf_vocab)
            feature = self.concatenate_feature(feature, word_trigram)
        if 'semantic_vector_similarity' in feature_list:
            semvec_sim = semantic_vector_similarity(train_data, test_data)
            feature = self.concatenate_feature(feature, semvec_sim)

        # feature = preprocessing.normalize(feature, norm='l1', axis=0)

        # print('test_feature', feature.shape)
        return feature
