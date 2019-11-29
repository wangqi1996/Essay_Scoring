# encoding=utf-8


STANFORDCORENLP_PATH = r'../lib/stanford-corenlp-full-2016-10-31'

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
TOKENIZER_FIELD = 'essay_t' \
                  'oken'

pos2gram_dim = 250
pos3gram_dim = 100

word_2gram_dim = 1000
word_3gram_dim = 1000

feature_list = [
    #
    "wv_similarity",
    "pos_bigram",
    # "pos_trigram",

    "word_bigram",
    # "word_trigram",

    "mean_word_length",
    "var_word_length",

    "mean_sentence_length",
    "var_sentence_length",

    "spell_error",
    "essay_length",

    "current_pos",
    "error_pos",
    "current_pos3",
    "error_pos3",

    "vocab_size",
    'unique_size',

    'PRP_result',
    # 'MD_result',
    'NNP_result',
    # 'COMMA_result',

    'JJ_result',  # 形容词
    'JJR_result',  # 比较级词语
    'JJS_result',  # 最高级
    # 'RB_result',  # 副词
    'RBR_result',  # 副词比较级
    # 'RBS_result',  # 副词比较级
    # 'PDT_result',  # 前限定词a

    'NN_result',  # 名词

    # 二元的
    'RB_JJ',  # 副词 + 形容词
    'JJR_NN',  # 最高级 + 名词
    'JJS_NNPS',  # 最高级 + 专有名词复数
    'RB_VB',  # 副词 + 动词
    'RB_RB',  # 副词+副词

    'bag_of_words',

    # "mean_sentence_depth",
    # "mean_sentence_level",
    #
    # "mean_clause_length",
    # "mean_clause_number",

    # "semantic_vector_similarity",

]
