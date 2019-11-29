# encoding=utf-8


STANFORDCORENLP_PATH = r'/Users/ikuc/stanfordnlp_resources/stanford-corenlp-full-2016-10-31'

TRAIN_DADA_PATH = r'../data/train.pickle'
TEST_DATA_PATH = r'../data/test.pickle'
DEV_DATA_PATH = r'../data/dev.pickle'

"""
把常量字段也定义在这里吧
"""

ESSAY_SET_FIELD = 'essay_set'
ESSAY_ID_FIELD = 'essay_id'
ESSAY_FIELD = 'essay'
SCORE_FIELD = 'domain1_score'
TOKENIZER_FIELD = 'essay_token'

feature_list = [

    "wv_similarity",
    "pos_bigram",
    # "pos_trigram",
    "word_bigram",
    "word_trigram",
    #
    # "mean_clause_length",
    # "mean_clause_number",
    #
    "mean_word_length",
    "var_word_length",
    "mean_sentence_length",
    "var_sentence_length",
    #
    "spell_error",
    # # "mean_sentence_depth",
    # # "mean_sentence_level",
    "essay_length",

    # "semantic_vector_similarity",
    "current_pos",
    "error_pos",
    "current_pos3",
    "error_pos3",
    "vocab_size",
]
