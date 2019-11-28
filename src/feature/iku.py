from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import *
import numpy
from gensim import models, corpora
from gensim.similarities import MatrixSimilarity
from nltk.corpus import stopwords as pw
import matplotlib.pyplot as plt
from src.config import STANFORDCORENLP_PATH


def spell_error(dataset):
    """
    count the number of spelling errors in each essay
    :param dataset: (list)
    :return:
    """
    print("spell_error")
    spell = SpellChecker()
    Max = 0
    Min = 9999
    ret = []
    for data in dataset:
        essay = [token for token in data['essay_token'] if token[0] != '@' and len(token) > 2]
        misspelled = spell.unknown(essay)
        data['spell_error'] = len(misspelled)
        ret.append(len(misspelled))
        if len(misspelled) > Max:
            Max = len(misspelled)
        if len(misspelled) < Min:
            Min = len(misspelled)
    #
    # x = [sample['spell_error'] for sample in dataset]
    # y = [sample['domain1_score'] for sample in dataset]
    # plt.plot(x, y, 'o')
    # plt.show()

    # return {'Max': Max, 'Min': Min}
    return numpy.array(ret).reshape(-1, 1)


def count_tree_depth(root):
    def travel(root, length, res):
        if type(root) == str:
            res.append(length)
            return
        l = len(root)
        for i in range(l):
            if root[i]:
                travel(root[i], length + 1, res)

    if root is None:
        return []
    l = []
    travel(root, 1, l)
    return l


def Mean_sentence_depth_level(dataset):
    """
    Sentence depth in the parser tree
    :param data: (list)
    :return:
    """
    print("Mean_sentence_depth_level")
    # nlp = stanfordnlp.Pipeline()
    nlp = StanfordCoreNLP(STANFORDCORENLP_PATH)
    Max_depth = 0
    Min_depth = 9999
    Max_level = 0
    Min_level = 9999
    ret_depth = []
    ret_level = []
    for data in dataset:
        essay = data['essay']
        # doc = nlp(essay)
        # for sentence in doc.sentences:
        #     print(sentence)
        #     print('')
        sentences = sent_tokenize(essay)
        depth_all = 0
        level_all = 0
        for sentence in sentences:
            parse_tree = nlp.parse(sentence)
            tree = Tree.fromstring(parse_tree)
            distance = count_tree_depth(tree)
            distance = numpy.array(distance)
            depth = numpy.sum(distance)
            level = numpy.amax(distance)
            depth_all += depth
            level_all += level
        data['mean_sentence_depth'] = depth_all / len(sentences)
        data['mean_sentence_level'] = level_all / len(sentences)

        ret_depth.append(data['mean_sentence_depth'])
        ret_level.append(data['mean_sentence_level'])

        if data['mean_sentence_depth'] > Max_depth:
            Max_depth = data['mean_sentence_depth']
        if data['mean_sentence_depth'] < Min_depth:
            Min_depth = data['mean_sentence_depth']
        if data['mean_sentence_level'] > Max_level:
            Max_level = data['mean_sentence_level']
        if data['mean_sentence_level'] < Min_level:
            Min_level = data['mean_sentence_level']

    # return {'Max_depth': Max_depth, 'Min_depth': Min_depth, 'Max_level': Max_level, 'Min_level': Min_level}
    return numpy.array(ret_depth).reshape(-1, 1), numpy.array(ret_level).reshape(-1, 1)


def essay_length(dataset):
    """
    Fourth root of essay length in words
    :param dataset: list
    :return:
    """
    Max = 0
    Min = 9999
    ret = []
    print("essay_length")
    for data in dataset:
        essay = data['essay_token']
        length = len(essay)
        length = pow(length, 1.0 / 4)
        data['essay_length'] = length
        ret.append(length)
        if length > Max:
            Max = length
        if length < Min:
            Min = length

    # x = [sample['essay_length'] for sample in dataset]
    # y = [sample['domain1_score'] for sample in dataset]
    # plt.plot(x, y, 'o')
    # plt.show()

    # return {'Max': Max, 'Min': Min}
    return numpy.array(ret).reshape(-1, 1)


def semantic_vector_similarity(dataset, test_data):
    """
    Mean cosine similarity to other essays’ semantic vector
    :param dataset: list
    :return:
    """
    print("semantic_vector_similarity")
    cacheStopWords = pw.words("english")
    punc = ['.', ',', '?', '!', '@', '"', 'n\'t']
    cacheStopWords.extend(punc)
    token_sets = []
    # print(cacheStopWords)
    for data in dataset:
        essay_token = [word for word in data['essay_token'] if word.lower() not in cacheStopWords]
        token_sets.append(essay_token)

    dictionary = corpora.Dictionary(token_sets)

    corpus = [dictionary.doc2bow(tokens) for tokens in token_sets]
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=20)
    documents = lsi_model[corpus]
    topics = lsi_model.show_topics(num_words=5, log=0)

    Min = 9999
    Max = 0

    scores = numpy.array([sample['domain1_score'] for sample in dataset])

    index = MatrixSimilarity(documents)
    predict_score_list = []
    score_list = []
    for sample, essay in zip(test_data, corpus):
        query = essay
        query_vec = lsi_model[query]
        # print(query)
        sim = index[query_vec]

        idxs = sim.argsort()[-20:-1][::-1]
        # print(idxs)
        _sim = [sim[idx] for idx in idxs]
        _scores = [scores[idx] for idx in idxs]

        # print(sim)
        predict_score = numpy.sum(numpy.multiply(scores, sim)) / len(dataset)
        sample['semantic_vector_similarity'] = predict_score
        # print(predict_score, sample['domain1_score'])
        predict_score_list.append(predict_score)
        # score_list.append(sample['domain1_score'])
    # plt.plot(predict_score_list, score_list, 'o')
    # plt.show()
    return numpy.array(predict_score_list).reshape(-1, 1)
