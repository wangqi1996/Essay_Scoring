# encoding=utf-8
import numpy as np
from sklearn import preprocessing

from src.config import feature_list
from src.feature.iku import spell_error, Mean_sentence_depth_level, essay_length
from src.feature.wangdq import word_vector_similarity_train, word_vector_similarity_test, \
    mean_clause, pos_gram_train, pos_gram_test, good_pos_ngrams, vocab_size, pos_tagger, pos_tagger2
from src.feature.xiaoyl import word_length, get_sentence_length, word_bigram_train, word_bigram_test, \
    word_trigram_train, word_trigram_test, bag_of_words_train, bag_of_words_test
from util.util import pos_tagging


class Feature:

    def __init__(self):

        self.wv_idf_diag = None
        self.wv_tf_vocab = None
        self.wv_tfidf = None

        self.word_bigram_TF = None
        self.word_bigram_tf_vocab = None

        self.word_trigram_TF = None
        self.word_trigram_tf_vocab = None

        self.bag_of_words_tf_vocab = None
        self.bag_of_words_TF = None

        self.pos_3tf_vocab = None
        self.pos_3TF = None

        self.pos_2tf_vocab = None
        self.pos_2TF = None

        self.normalizer = None
        self.tagged_data = None

    def get_tagged_data(self, tokens_set):
        """ 省时间啦 """
        if self.tagged_data is None:
            self.tagged_data = pos_tagging(tokens_set)
        return self.tagged_data

    @staticmethod
    def get_instance(feature):
        """ 从dataset中构建dataset
        feature: {} """
        feature_class = Feature()

        feature_class.wv_idf_diag = feature.get('wv_idf_diag', None)
        feature_class.wv_tf_vocab = feature.get('wv_tf_vocab', None)
        feature_class.wv_tfidf = feature.get('wv_tfidf', None)

        feature_class.pos_2tf_vocab = feature.get('pos_2tf_vocab', None)
        feature_class.pos_2TF = feature.get('pos_2TF', None)

        feature_class.pos_3tf_vocab = feature.get('pos_3tf_vocab', None)
        feature_class.pos_3TF = feature.get('pos_3TF', None)

        feature_class.word_bigram_tf_vocab = feature.get('word_bigram_tf_vocab', None)
        feature_class.word_bigram_TF = feature.get('word_bigram_TF', None)

        feature_class.word_trigram_tf_vocab = feature.get('word_trigram_tf_vocab', None)
        feature_class.word_trigram_TF = feature.get('word_trigram_TF', None)

        feature_class.bag_of_words_tf_vocab = feature.get('bag_of_words_tf_vocab', None)
        feature_class.bag_of_words_TF = feature.get('bag_of_word_TF', None)

        return feature_class

    def concatenate_feature(self, feature, new_feature):
        if new_feature is None:
            return feature
        if feature is None:
            return new_feature
        else:
            return np.concatenate((feature, new_feature), axis=1)

    def get_feature_by_name(self, feature_dict, feature_name, sentences_set, token_set, train_data, train_score,
                            name='train'):

        if 'mean_word_length' == feature_name:
            mean_word_length, var_word_length = word_length(token_set)
            feature_dict.update({
                "mean_word_length": mean_word_length,
                "var_word_length": var_word_length
            })
            return mean_word_length

        if 'var_word_length' == feature_name:
            mean_word_length, var_word_length = word_length(token_set)
            feature_dict.update({
                "mean_word_length": mean_word_length,
                "var_word_length": var_word_length
            })
            return var_word_length

        if 'mean_clause_length' == feature_name:
            mean_clause_length, mean_clause_number, mean_sentence_depth, mean_sentence_level = mean_clause(
                sentences_set)
            feature_dict.update({
                "mean_clause_length": mean_clause_length,
                "mean_clause_number": mean_clause_number,
                "mean_sentence_depth": mean_sentence_depth,
                "mean_sentence_level": mean_sentence_level
            })
            return mean_clause_length

        if 'mean_clause_number' == feature_name:
            mean_clause_length, mean_clause_number, mean_sentence_depth, mean_sentence_level = mean_clause(
                sentences_set)
            feature_dict.update({
                "mean_clause_length": mean_clause_length,
                "mean_clause_number": mean_clause_number,
                "mean_sentence_depth": mean_sentence_depth,
                "mean_sentence_level": mean_sentence_level
            })
            return mean_clause_number

        if 'mean_sentence_length' == feature_name:
            mean_sentence_length, var_sentence_length = get_sentence_length(sentences_set)
            feature_dict.update({
                "mean_sentence_length": mean_sentence_length,
                "var_sentence_length": var_sentence_length
            })
            return mean_sentence_length

        if 'var_sentence_length' == feature_name:
            mean_sentence_length, var_sentence_length = get_sentence_length(sentences_set)
            feature_dict.update({
                "mean_sentence_length": mean_sentence_length,
                "var_sentence_length": var_sentence_length
            })
            return var_sentence_length

        if 'spell_error' == feature_name:
            error = spell_error(train_data)
            feature_dict.update({
                "spell_error": error
            })
            return error

        if 'mean_sentence_depth' == feature_name:
            mean_clause_length, mean_clause_number, mean_sentence_depth, mean_sentence_level = mean_clause(
                sentences_set)
            feature_dict.update({
                "mean_clause_length": mean_clause_length,
                "mean_clause_number": mean_clause_number,
                "mean_sentence_depth": mean_sentence_depth,
                "mean_sentence_level": mean_sentence_level
            })
            return mean_sentence_depth

        if 'mean_sentence_level' == feature_name:
            mean_clause_length, mean_clause_number, mean_sentence_depth, mean_sentence_level = mean_clause(
                sentences_set)
            feature_dict.update({
                "mean_clause_length": mean_clause_length,
                "mean_clause_number": mean_clause_number,
                "mean_sentence_depth": mean_sentence_depth,
                "mean_sentence_level": mean_sentence_level
            })
            return mean_sentence_level

        if 'essay_length' == feature_name:
            length = essay_length(train_data)
            feature_dict.update({
                "essay_length": length,
            })
            return length

        if 'current_pos' == feature_name:
            current_pos, error_pos = good_pos_ngrams(self.get_tagged_data(token_set), gram=2)
            feature_dict.update({
                "current_pos": current_pos,
                "error_pos": error_pos
            })
            return current_pos

        if 'error_pos' == feature_name:
            current_pos, error_pos = good_pos_ngrams(self.get_tagged_data(token_set), gram=2)
            feature_dict.update({
                "current_pos": current_pos,
                "error_pos": error_pos
            })
            return error_pos

        if 'current_pos3' == feature_name:
            current_pos3, error_pos3 = good_pos_ngrams(self.get_tagged_data(token_set), gram=3)
            feature_dict.update({
                "current_pos3": current_pos3,
                "error_pos3": error_pos3
            })
            return current_pos3

        if 'error_pos3' == feature_name:
            current_pos3, error_pos3 = good_pos_ngrams(self.get_tagged_data(token_set), gram=3)
            feature_dict.update({
                "current_pos3": current_pos3,
                "error_pos3": error_pos3
            })
            return error_pos3

        if 'vocab_size' == feature_name:
            vocab_len, unique_size = vocab_size(token_set)
            feature_dict.update({
                "vocab_size": vocab_len,
                "unique_size": unique_size
            })
            return vocab_len

        if 'unique_size' == feature_name:
            vocab_len, unique_size = vocab_size(token_set)
            feature_dict.update({
                "vocab_size": vocab_len,
                "unique_size": unique_size
            })
            return unique_size

        if feature_name in ["PRP_result", "MD_result", "NNP_result", "COMMA_result", "JJ_result", "JJS_result",
                            "JJR_result", "RB_result", "RBR_result", "RBS_result", "PDT_result", "NN_result"]:
            label = feature_name[:-7]
            feature_value = pos_tagger(self.get_tagged_data(token_set), label)
            feature_dict.update({
                feature_name: feature_value
            })

            return feature_value

        if feature_name in ['RB_JJ', 'JJR_NN', 'JJS_NNPS', 'RB_VB', 'RB_RB']:
            feature_value = pos_tagger2(self.get_tagged_data(token_set), feature_name)
            feature_dict.update({
                feature_name: feature_value
            })

            return feature_value

        if 'wv_similarity' == feature_name:
            if name == 'train':
                wv_similarity, self.wv_tf_vocab, self.wv_idf_diag, self.wv_tfidf = word_vector_similarity_train(
                    token_set, train_score)

            else:
                wv_similarity = word_vector_similarity_test(token_set, train_score, self.wv_tf_vocab,
                                                            self.wv_idf_diag, self.wv_tfidf)

            feature_dict.update({
                "wv_similarity": wv_similarity
            })
            return wv_similarity

        if 'pos_bigram' == feature_name:
            if name == 'train':
                pos_bigram, self.pos_2TF, self.pos_2tf_vocab = pos_gram_train(self.get_tagged_data(token_set), 2)
            else:
                pos_bigram = pos_gram_test(self.get_tagged_data(token_set), self.pos_2TF, self.pos_2tf_vocab, 2)

            feature_dict.update({
                "pos_bigram": pos_bigram
            })
            return pos_bigram

        if 'pos_trigram' == feature_name:
            if name == 'train':
                pos_trigram, self.pos_3TF, self.pos_3tf_vocab = pos_gram_train(self.get_tagged_data(token_set), 3)
            else:
                pos_trigram = pos_gram_test(self.get_tagged_data(token_set), self.pos_3TF, self.pos_3tf_vocab, 3)

            feature_dict.update({
                "pos_trigram": pos_trigram
            })
            return pos_trigram

        if 'word_bigram' == feature_name:
            if name == 'train':
                word_bigram, self.word_bigram_TF, self.word_bigram_tf_vocab = word_bigram_train(token_set)
            else:

                word_bigram = word_bigram_test(token_set, self.word_bigram_TF, self.word_bigram_tf_vocab)

            feature_dict.update({
                "word_bigram": word_bigram
            })
            return word_bigram

        if 'word_trigram' == feature_name:
            if name == 'train':
                word_trigram, self.word_trigram_TF, self.word_trigram_tf_vocab = word_trigram_train(token_set)
            else:

                word_trigram = word_trigram_test(token_set, self.word_trigram_TF, self.word_trigram_tf_vocab)

            feature_dict.update({
                "word_trigram": word_trigram
            })
            return word_trigram

        if 'semantic_vector_similarity' == feature_name:
            assert False, u'这个特征没用呀，还没实现'

        if 'bag_of_words' == feature_name:
            if name == 'train':
                bag_of_words, self.bag_of_words_TF, self.bag_of_words_tf_vocab = bag_of_words_train(token_set)
            else:
                bag_of_words = bag_of_words_test(token_set, self.bag_of_words_TF, self.bag_of_words_tf_vocab)

            feature_dict.update({
                "bag_of_words": bag_of_words
            })
            return bag_of_words

    def get_saved_feature_all(self, feature_dict, sentences_list, tokens_list, train_data, train_score, name='train',
                              reset_list=[]):

        feature = None
        self.tagged_data = None
        for feature_name in feature_list:
            feature_value = feature_dict.get(feature_name, None)

            if feature_value is None or feature_name in reset_list:
                # 重新计算
                feature_value = self.get_feature_by_name(feature_dict, feature_name, sentences_list,
                                                         tokens_list, train_data, train_score, name)
            assert feature_value is not None, u"feature不能为none呀"
            feature = self.concatenate_feature(feature, feature_value)

        if name == 'train':
            self.normalizer = preprocessing.StandardScaler().fit(feature)
            feature = self.normalizer.transform(feature)
        else:
            feature = self.normalizer.transform(feature)

        print("feature.shape = ", feature.shape)
        return feature, feature_dict
