# # coding=utf-8
#
# # def get_save_feature(self, sentences_list, tokens_list, tagger_data):
# #
# #     feature = None
# #     if 'mean_clause_length' in feature_list:
# #         feature = self.concatenate_feature(feature, self.mean_clause_length)
# #     if 'mean_clause_number' in feature_list:
# #         feature = self.concatenate_feature(feature, self.mean_clause_number)
# #
# #     if 'mean_word_length' in feature_list:
# #         feature = self.concatenate_feature(feature, self.mean_word_length)
# #     if 'var_word_length' in feature_list:
# #         feature = self.concatenate_feature(feature, self.var_word_length)
# #
# #     if 'mean_sentence_length' in feature_list:
# #         feature = self.concatenate_feature(feature, self.mean_sentence_length)
# #     if 'var_sentence_length' in feature_list:
# #         feature = self.concatenate_feature(feature, self.var_sentence_length)
# #
# #     if 'spell_error' in feature_list:
# #         feature = self.concatenate_feature(feature, self.spell_error)
# #
# #     if 'mean_sentence_depth' in feature_list:
# #         feature = self.concatenate_feature(feature, self.mean_sentence_depth)
# #     if 'mean_sentence_level' in feature_list:
# #         feature = self.concatenate_feature(feature, self.mean_sentence_length)
# #
# #     if 'essay_length' in feature_list:
# #         feature = self.concatenate_feature(feature, self.essay_length)
# #
# #     if 'current_pos' in feature_list:
# #         if self.current_pos is None:
# #             self.current_pos, self.error_pos = good_pos_ngrams(tagger_data, gram=2)
# #         feature = self.concatenate_feature(feature, self.current_pos)
# #
# #     if 'error_pos' in feature_list:
# #         if self.error_pos is None:
# #             _, self.error_pos = good_pos_ngrams(tagger_data, gram=2)
# #         feature = self.concatenate_feature(feature, self.error_pos)
# #
# #     if 'current_pos3' in feature_list:
# #         if self.current_pos3 is None:
# #             self.current_pos3, self.error_pos3 = good_pos_ngrams(tagger_data, gram=3)
# #         feature = self.concatenate_feature(feature, self.current_pos3)
# #
# #     if 'error_pos3' in feature_list:
# #         if self.error_pos3 is None:
# #             _, self.error_pos3 = good_pos_ngrams(tagger_data, gram=3)
# #         feature = self.concatenate_feature(feature, self.error_pos3)
# #
# #     if 'vocab_size' in feature_list:
# #         if self.vocab_size is None:
# #             self.vocab_size, self.unique_size = vocab_size(tokens_list)
# #         feature = self.concatenate_feature(feature, self.vocab_size)
# #
# #     if 'unique_size' in feature_list:
# #         if self.unique_size is None:
# #             _, self.unique_size = vocab_size(tokens_list)
# #         feature = self.concatenate_feature(feature, self.unique_size)
# #
# #     if 'PRP_result' in feature_list:
# #         if self.PRP_result is None:
# #             self.PRP_result, self.MD_result, self.NNP_result, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.PRP_result)
# #
# #     if 'MD_result' in feature_list:
# #         if self.MD_result is None:
# #             _, self.MD_result, self.NNP_result, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.MD_result)
# #
# #     if 'NNP_result' in feature_list:
# #         if self.NNP_result is None:
# #             _, _, self.NNP_result, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.NNP_result)
# #
# #     if 'COMMA_result' in feature_list:
# #         if self.COMMA_result is None:
# #             _, _, _, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.COMMA_result)
# #
# #     return feature
# #
# # def get_train_feature(self, sentences_list, tokens_list, scores, train_data):
# #     """ 获取training的feature
# #     sentences_array, tokens_array 均已经tokenizer处理过
# #     所有返回的特征的维度: ndarray类型 sample_num * feature_dim
# #     """
# #     #
# #     # if self.train_feature is not None:
# #     #     return self.train_feature
# #     tagged_data = pos_tagging(tokens_list)
# #
# #     feature, _ = self.get_feature(sentences_list, tokens_list, train_data, tagged_data)
# #
# #     if 'wv_similarity' in feature_list:
# #         wv_similarity, wv_tf_vocab, wv_idf_diag, wv_tfidf = word_vector_similarity_train(tokens_list, scores)
# #         self.wv_similarity = wv_similarity
# #         self.wv_idf_diag = wv_idf_diag
# #         self.wv_tf_vocab = wv_tf_vocab
# #         self.wv_tfidf = wv_tfidf
# #         feature = self.concatenate_feature(feature, wv_similarity)
# #
# #     if 'pos_bigram' in feature_list:
# #         pos_bigram, pos_2TF, pos_2tf_vocab = pos_gram_train(tagged_data, 2)
# #         self.pos_bigram = pos_bigram
# #         self.pos_2tf_vocab = pos_2tf_vocab
# #         self.pos_2TF = pos_2TF
# #         feature = self.concatenate_feature(feature, pos_bigram)
# #
# #     if 'pos_trigram' in feature_list:
# #         pos_trigram, pos_3TF, pos_3tf_vocab = pos_gram_train(tagged_data, 3)
# #         self.pos_trigram = pos_trigram
# #         self.pos_3tf_vocab = pos_3tf_vocab
# #         self.pos_3TF = pos_3TF
# #         feature = self.concatenate_feature(feature, pos_trigram)
# #
# #     if 'word_bigram' in feature_list:
# #         word_bigram, word_bigram_TF, word_bigram_tf_vocab = word_bigram_train(tokens_list)
# #         self.word_bigram = word_bigram
# #         self.word_bigram_tf_vocab = word_bigram_tf_vocab
# #         self.word_bigram_TF = word_bigram_TF
# #         feature = self.concatenate_feature(feature, word_bigram)
# #
# #     if 'word_trigram' in feature_list:
# #         word_trigram, word_trigram_TF, word_trigram_tf_vocab = word_trigram_train(tokens_list)
# #         self.word_trigram = word_trigram
# #         self.word_trigram_tf_vocab = word_trigram_tf_vocab
# #         self.word_trigram_TF = word_trigram_TF
# #         feature = self.concatenate_feature(feature, word_trigram)
# #
# #     if 'semantic_vector_similarity' in feature_list:
# #         semvec_sim = semantic_vector_similarity(train_data, train_data)
# #         self.semantic_vector_similarity = semvec_sim
# #         feature = self.concatenate_feature(feature, semvec_sim)
# #
# #     if 'bag_of_words' in feature_list:
# #         bag_of_words, bag_of_words_TF, bag_of_words_tf_vocab = bag_of_words_train(tokens_list)
# #         self.bag_of_words = bag_of_words
# #         self.bag_of_words_tf_vocab = bag_of_words_tf_vocab
# #         self.bag_of_words_TF = bag_of_words_TF
# #         feature = self.concatenate_feature(feature, bag_of_words)
# #
# #     normalizer = preprocessing.StandardScaler().fit(feature)
# #     feature = normalizer.transform(feature)
# #
# #     # print(normalizer)
# #
# #     # feature = preprocessing.normalize(feature, norm='max', axis=0)
# #     self.train_feature = feature
# #     self.Normalizer = normalizer
# #
# #     # print(feature)
# #     # print('train_feature', feature.shape)
# #     return feature
# #
# # def get_save_train_feature(self, sentences_list, tokens_list):
# #     print("use_save")
# #     feature = self.get_save_feature(sentences_list, tokens_list)
# #
# #     if 'wv_similarity' in feature_list:
# #         feature = self.concatenate_feature(feature, self.wv_similarity)
# #     if 'pos_bigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.pos_bigram)
# #     if 'pos_trigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.pos_trigram)
# #     if 'word_bigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.word_bigram)
# #     if 'word_trigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.word_trigram)
# #     if 'semantic_vector_similarity' in feature_list:
# #         feature = self.concatenate_feature(feature, self.semantic_vector_similarity)
# #
# #     normalizer = preprocessing.StandardScaler().fit(feature)
# #     feature = normalizer.transform(feature)
# #
# #     self.train_feature = feature
# #     self.Normalizer = normalizer
# #
# #     return feature
# #
# # def get_test_feature(self, sentences_list, tokens_list, train_score, train_data, test_data):
# #     """ """
# #     tagged_data = pos_tagging(tokens_list)
# #
# #     feature = self.get_feature(sentences_list, tokens_list, test_data, tagged_data)
# #     # #####################
# #     # # feature = self.append_feature(feature, wv_similarity, pos_bigram, word_bigram, word_trigram)
# #     if 'wv_similarity' in feature_list:
# #         self.wv_similarity = word_vector_similarity_test(tokens_list, train_score, self.wv_tf_vocab,
# #                                                          self.wv_idf_diag,
# #                                                          self.wv_tfidf)
# #         feature = self.concatenate_feature(feature, self.wv_similarity)
# #
# #     if 'pos_bigram' in feature_list:
# #         self.pos_bigram = pos_gram_test(tagged_data, self.pos_2TF, self.pos_2tf_vocab, 2)
# #         feature = self.concatenate_feature(feature, self.pos_bigram)
# #
# #     if 'pos_trigram' in feature_list:
# #         self.pos_trigram = pos_gram_test(tagged_data, self.pos_3TF, self.pos_3tf_vocab, 3)
# #         feature = self.concatenate_feature(feature, self.pos_trigram)
# #
# #     if 'word_bigram' in feature_list:
# #         self.word_bigram = word_bigram_test(tokens_list, self.word_bigram_TF, self.word_bigram_tf_vocab)
# #         feature = self.concatenate_feature(feature, self.word_bigram)
# #
# #     if 'word_trigram' in feature_list:
# #         self.word_trigram = word_trigram_test(tokens_list, self.word_trigram_TF, self.word_trigram_tf_vocab)
# #         feature = self.concatenate_feature(feature, self.word_trigram)
# #
# #     if 'semantic_vector_similarity' in feature_list:
# #         self.semvec_sim = semantic_vector_similarity(train_data, test_data)
# #         feature = self.concatenate_feature(feature, self.semvec_sim)
# #
# #     if 'bag_of_words' in feature_list:
# #         self.bag_of_words = bag_of_words_test(tokens_list, self.bag_of_words_TF, self.bag_of_words_tf_vocab)
# #         feature = self.concatenate_feature(feature, self.bag_of_words)
# #
# #     # print("test_before",feature[:,1])
# #
# #     # feature = preprocessing.normalize(feature, norm='max', axis=0)
# #     # add_feature = self.get_add_feature(sentences_list, tokens_list)
# #     # feature = self.concatenate_feature(feature, add_feature)
# #     feature = self.Normalizer.transform(feature)
# #
# #     # print("test_after", feature[:,1])
# #
# #     # print('test_feature', feature.shape)
# #     return feature
# #
# # def get_save_feature_from_dict(self, feature_dict, sentences_list, tokens_list, tagger_data=None):
# #
# #     feature = None
# #     if 'mean_clause_length' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('mean_clause_length'))
# #     if 'mean_clause_number' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('mean_clause_number'))
# #
# #     if 'mean_word_length' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('mean_word_length'))
# #     if 'var_word_length' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('var_word_length'))
# #
# #     if 'mean_sentence_length' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('mean_sentence_length'))
# #     if 'var_sentence_length' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('var_sentence_length'))
# #
# #     if 'spell_error' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('spell_error'))
# #
# #     if 'mean_sentence_depth' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('mean_sentence_depth'))
# #     if 'mean_sentence_level' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('mean_sentence_length'))
# #
# #     if 'essay_length' in feature_list:
# #         feature = self.concatenate_feature(feature, feature_dict.get('essay_length'))
# #
# #     if 'current_pos' in feature_list:
# #         current_pos = feature_dict.get('current_pos', None)
# #         if current_pos is None:
# #             self.current_pos, self.error_pos = good_pos_ngrams(tagger_data, gram=2)
# #         feature = self.concatenate_feature(feature, self.current_pos)
# #
# #     if 'error_pos' in feature_list:
# #         if self.error_pos is None:
# #             _, self.error_pos = good_pos_ngrams(tagger_data, gram=2)
# #         feature = self.concatenate_feature(feature, self.error_pos)
# #
# #     if 'current_pos3' in feature_list:
# #         if self.current_pos3 is None:
# #             self.current_pos3, self.error_pos3 = good_pos_ngrams(tagger_data, gram=3)
# #         feature = self.concatenate_feature(feature, self.current_pos3)
# #
# #     if 'error_pos3' in feature_list:
# #         if self.error_pos3 is None:
# #             _, self.error_pos3 = good_pos_ngrams(tagger_data, gram=3)
# #         feature = self.concatenate_feature(feature, self.error_pos3)
# #
# #     if 'vocab_size' in feature_list:
# #         if self.vocab_size is None:
# #             self.vocab_size, self.unique_size = vocab_size(tokens_list)
# #         feature = self.concatenate_feature(feature, self.vocab_size)
# #
# #     if 'unique_size' in feature_list:
# #         if self.unique_size is None:
# #             _, self.unique_size = vocab_size(tokens_list)
# #         feature = self.concatenate_feature(feature, self.unique_size)
# #
# #     if 'PRP_result' in feature_list:
# #         if self.PRP_result is None:
# #             self.PRP_result, self.MD_result, self.NNP_result, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.PRP_result)
# #
# #     if 'MD_result' in feature_list:
# #         if self.MD_result is None:
# #             _, self.MD_result, self.NNP_result, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.MD_result)
# #
# #     if 'NNP_result' in feature_list:
# #         if self.NNP_result is None:
# #             _, _, self.NNP_result, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.NNP_result)
# #
# #     if 'COMMA_result' in feature_list:
# #         if self.COMMA_result is None:
# #             _, _, _, self.COMMA_result = pos_tagger(tagger_data)
# #         feature = self.concatenate_feature(feature, self.COMMA_result)
# #
# #     return feature
# #
# # def get_test_save_feature(self, feature_dict, sentences_list, tokens_list, train_scores, train_data,
# #                           dev_data):
# #     print("use_save")
# #     tagged_data = pos_tagging(tokens_list)
# #     feature = self.get_save_feature_from_dict(feature_dict, sentences_list, tokens_list, )
# #
# #     if 'wv_similarity' in feature_list:
# #         feature = self.concatenate_feature(feature, self.wv_similarity)
# #     if 'pos_bigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.pos_bigram)
# #     if 'pos_trigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.pos_trigram)
# #     if 'word_bigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.word_bigram)
# #     if 'word_trigram' in feature_list:
# #         feature = self.concatenate_feature(feature, self.word_trigram)
# #     if 'semantic_vector_similarity' in feature_list:
# #         feature = self.concatenate_feature(feature, self.semantic_vector_similarity)
# #
# #     normalizer = preprocessing.StandardScaler().fit(feature)
# #     feature = normalizer.transform(feature)
# #
# #     self.train_feature = feature
# #     self.Normalizer = normalizer
# #
# #     return feature
#
# # def get_feature(self, sentences_set, token_set, train_data, tagged_data):
# #     """ 不用区分训练测试的, 可以写在这类"""
# #
# #     assert False, u"不能用呀wdq"
# #
# #     feature = None
# #
# #     if 'mean_clause_length' in feature_list or 'mean_clause_number' in feature_list:
# #         mean_clause_length, mean_clause_number = mean_clause(sentences_set)
# #         self.mean_clause_number = mean_clause_number
# #         self.mean_clause_length = mean_clause_length
# #         if 'mean_clause_length' in feature_list:
# #             feature = self.concatenate_feature(feature, mean_clause_length)
# #         if 'mean_clause_number' in feature_list:
# #             feature = self.concatenate_feature(feature, mean_clause_number)
# #
# #     if 'mean_word_length' in feature_list or 'var_word_length' in feature_list:
# #         mean_word_length, var_word_length = word_length(token_set)
# #         self.mean_word_length = mean_word_length
# #         self.var_word_length = var_word_length
# #         if 'mean_word_length' in feature_list:
# #             feature = self.concatenate_feature(feature, mean_word_length)
# #         if 'var_word_length' in feature_list:
# #             feature = self.concatenate_feature(feature, var_word_length)
# #
# #     if 'mean_sentence_length' in feature_list or 'var_sentence_length' in feature_list:
# #         mean_sentence_length, var_sentence_length = get_sentence_length(sentences_set)
# #         self.mean_sentence_length = mean_sentence_length
# #         self.var_sentence_length = var_sentence_length
# #         if 'mean_sentence_length' in feature_list:
# #             feature = self.concatenate_feature(feature, mean_sentence_length)
# #         if 'var_sentence_length' in feature_list:
# #             feature = self.concatenate_feature(feature, var_sentence_length)
# #
# #     if 'spell_error' in feature_list:
# #         error = spell_error(train_data)
# #         self.spell_error = error
# #         feature = self.concatenate_feature(feature, error)
# #
# #     if 'mean_sentence_depth' in feature_list or 'mean_sentence_level' in feature_list:
# #         depth, level = Mean_sentence_depth_level(train_data)
# #         self.mean_sentence_level = level
# #         self.mean_sentence_depth = depth
# #         if 'mean_sentence_depth' in feature_list:
# #             feature = self.concatenate_feature(feature, depth)
# #         if 'mean_sentence_level' in feature_list:
# #             feature = self.concatenate_feature(feature, level)
# #
# #     if 'essay_length' in feature_list:
# #         length = essay_length(train_data)
# #         self.essay_length = length
# #         feature = self.concatenate_feature(feature, length)
# #
# #     # feature = self.append_feature(mean_clause_length, mean_clause_number, mean_word_length, var_word_length,
# #     #                               mean_sentence_length, var_sentence_length)
# #
# #     if "current_pos" in feature_list:
# #         self.current_pos, self.error_pos = good_pos_ngrams(tagged_data, gram=2)
# #         feature = self.concatenate_feature(feature, self.current_pos)
# #         feature = self.concatenate_feature(feature, self.error_pos)
# #
# #     if "current_pos3" in feature_list:
# #         self.current_pos3, self.error_pos3 = good_pos_ngrams(tagged_data, gram=3)
# #         feature = self.concatenate_feature(feature, self.current_pos3)
# #         feature = self.concatenate_feature(feature, self.error_pos3)
# #
# #     if 'vocab_size' in feature_list or 'unique_size' in feature_list:
# #         self.vocab_size, self.unique_size = vocab_size(token_set)
# #         if 'vocab_size' in feature_list:
# #             feature = self.concatenate_feature(feature, self.vocab_size)
# #         if 'unique_size' in feature_list:
# #             feature = self.concatenate_feature(feature, self.unique_size)
# #
# #     if 'PRP_result' in feature_list:
# #         self.PRP_result, self.MD_result, self.NNP_result, self.COMMA_result = pos_tagger(tagged_data)
# #         feature = self.concatenate_feature(feature, self.PRP_result)
# #         feature = self.concatenate_feature(feature, self.MD_result)
# #         feature = self.concatenate_feature(feature, self.NNP_result)
# #         feature = self.concatenate_feature(feature, self.COMMA_result)
# #     return feature
#
# # def save_feature(self, feature_dict):
# #     """ 保存feature和中间需要使用的变量"""
# #
# #     result = {}
# #     for feature_name in feature_list:
# #         feature_value = getattr(self, feature_name, None)
# #
# #         assert feature_value is not None, u"哦我也不知道为啥能为None, 重大bug"
# #
# #     result = {
# #         "wv_tf_vocab": self.wv_tf_vocab,
# #         "wv_idf_diag": self.wv_idf_diag,
# #         "wv_tfidf": self.wv_tfidf,
# #         "pos_2tf_vocab": self.pos_2tf_vocab,
# #         "pos_2TF": self.pos_2TF,
# #         "pos_3TF": self.pos_3TF,
# #         "pos_3tf_vocab": self.pos_3tf_vocab,
# #
# #         "wv_similarity": self.wv_similarity,
# #         "pos_bigram": self.pos_bigram,
# #         "pos_trigram": self.pos_trigram,
# #         "word_bigram": self.word_bigram,
# #         "word_trigram": self.word_trigram,
# #         "bag_of_words": self.bag_of_words,
# #
# #         "mean_clause_length": self.mean_clause_length,
# #         "mean_clause_number": self.mean_clause_number,
# #
# #         "mean_word_length": self.mean_word_length,
# #         "var_word_length": self.var_word_length,
# #         "mean_sentence_length": self.mean_sentence_length,
# #         "var_sentence_length": self.var_sentence_length,
# #
# #         "word_bigram_tf_vocab": self.word_bigram_tf_vocab,
# #         "word_bigram_TF": self.word_bigram_TF,
# #         "word_trigram_tf_vocab": self.word_trigram_tf_vocab,
# #         "word_trigram_TF": self.word_trigram_TF,
# #
# #         "spell_error": self.spell_error,
# #         "mean_sentence_depth": self.mean_sentence_depth,
# #         "mean_sentence_level": self.mean_sentence_level,
# #         "essay_length": self.essay_length,
# #         "semantic_vector_similarity": self.semantic_vector_similarity,
# #
# #         "current_pos": self.current_pos,
# #         "error_pos": self.error_pos,
# #         "current_pos3": self.current_pos3,
# #         "error_pos3": self.error_pos3,
# #         'vocab_size': self.vocab_size,
# #         "train_feature": train_feature,
# #         "PRP_result": self.PRP_result,
# #         "MD_result": self.MD_result,
# #         "NNP_result": self.NNP_result,
# #         'unique_size': self.unique_size,
# #         'COMMA_result': self.COMMA_result
# #     }
# #
# #     return result
# def __init__(self):
#     # 应该放在dataset中懒得放了
#     # 禁止从这个类初始化
#     # self.wv_similarity = None
#     self.wv_idf_diag = None
#     self.wv_tf_vocab = None
#     self.wv_tfidf = None
#
#     # self.word_bigram = None
#     self.word_bigram_TF = None
#     self.word_bigram_tf_vocab = None
#
#     # self.word_trigram = None
#     self.word_trigram_TF = None
#     self.word_trigram_tf_vocab = None
#
#     # self.bag_of_words = None
#     self.bag_of_words_tf_vocab = None
#     self.bag_of_words_TF = None
#
#     # self.mean_clause_length = None
#     # self.mean_clause_number = None
#
#     # self.pos_trigram = None
#     self.pos_3tf_vocab = None
#     self.pos_3TF = None
#
#     # self.pos_bigram = None
#     self.pos_2tf_vocab = None
#     self.pos_2TF = None
#
#     # self.mean_word_length = None
#     # self.var_word_length = None
#     # self.mean_sentence_length = None
#     # self.var_sentence_length = None
#
#     # self.spell_error = None
#     # self.mean_sentence_depth = None
#     # self.mean_sentence_level = None
#     # self.essay_length = None
#     # self.semantic_vector_similarity = None
#
#     self.Normalizer = None
#
#     # self.current_pos = None
#     # self.error_pos = None
#     # self.current_pos3 = None
#     # self.error_pos3 = None
#     # self.vocab_size = None
#     # self.unique_size = None
#     #
#     # self.PRP_result = None
#     # self.MD_result = None
#     # self.NNP_result = None
#     # self.COMMA_result = None
#
#
# @staticmethod
# def get_instance(feature):
#     """ 从dataset中构建dataset
#     feature: {} """
#     feature_class = Feature()
#
#     # feature_class.wv_similarity = feature.get('wv_similarity', None)
#     feature_class.wv_idf_diag = feature.get('wv_idf_diag', None)
#     feature_class.wv_tf_vocab = feature.get('wv_tf_vocab', None)
#     feature_class.wv_tfidf = feature.get('wv_tfidf', None)
#
#     # feature_class.pos_bigram = feature.get('pos_bigram', None)
#     feature_class.pos_2tf_vocab = feature.get('pos_2tf_vocab', None)
#     feature_class.pos_2TF = feature.get('pos_2TF', None)
#
#     # feature_class.pos_trigram = feature.get('pos_trigram', None)
#     feature_class.pos_3tf_vocab = feature.get('pos_3tf_vocab', None)
#     feature_class.pos_3TF = feature.get('pos_3TF', None)
#
#     # feature_class.word_bigram = feature.get('word_bigram', None)
#     feature_class.word_bigram_tf_vocab = feature.get('word_bigram_tf_vocab', None)
#     feature_class.word_bigram_TF = feature.get('word_bigram_TF', None)
#
#     # feature_class.word_trigram = feature.get('word_trigram', None)
#     feature_class.word_trigram_tf_vocab = feature.get('word_trigram_tf_vocab', None)
#     feature_class.word_trigram_TF = feature.get('word_trigram_TF', None)
#
#     feature_class.bag_of_words_tf_vocab = feature.get('bag_of_words_tf_vocab', None)
#     feature_class.bag_of_words_TF = feature.get('bag_of_word_TF', None)
#     # feature_class.bag_of_words = feature.get('bag_of_words', None)
#
#     # feature_class.mean_clause_length = feature.get('mean_clause_length', None)
#     # feature_class.mean_clause_number = feature.get('mean_clause_number', None)
#     #
#     # feature_class.mean_word_length = feature.get('mean_word_length', None)
#     # feature_class.var_word_length = feature.get('var_word_length', None)
#     # feature_class.mean_sentence_length = feature.get('mean_sentence_length', None)
#     # feature_class.var_sentence_length = feature.get('var_sentence_length', None)
#     #
#     # feature_class.spell_error = feature.get('spell_error', None)
#     # feature_class.mean_sentence_depth = feature.get('mean_sentence_depth', None)
#     # feature_class.mean_sentence_level = feature.get('mean_sentence_level', None)
#     # feature_class.essay_length = feature.get('essay_length', None)
#     # feature_class.semantic_vector_similarity = feature.get('semantic_vector_similarity', None)
#     #
#     # feature_class.error_pos = feature.get('error_pos', None)
#     # feature_class.current_pos = feature.get('current_pos', None)
#     #
#     # feature_class.error_pos3 = feature.get('error_pos3', None)
#     # feature_class.current_pos3 = feature.get('current_pos3', None)
#     #
#     # feature_class.PRP_result = feature.get('PRP_result', None)
#     # feature_class.MD_result = feature.get('MD_result', None)
#     # feature_class.NNP_result = feature.get('NNP_result', None)
#     # feature_class.COMMA_result = feature.get('COMMA_result', None)
#     #
#     # feature_class.vocab_size = feature.get('vocab_size', None)
#     # feature_class.unique_size = feature.get('unique_size', None)
#     # # 一个array数组
#     # feature_class.train_feature = feature.get('train_feature', None)
#     return feature_class