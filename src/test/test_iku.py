import sys

sys.path.append("../..")
from src.feature.iku import spell_error, Mean_sentence_depth_level, semantic_vector_similarity, essay_length
from src.data import Dataset
from gensim import corpora, models
from gensim.similarities import MatrixSimilarity

# test_dataset= Dataset.load("../../data/test.pickle")
# train_dataset = Dataset.load("../../data/train.pickle")

train_dataset = Dataset()
train_dataset.load_from_raw_file("../../data/train.tsv", ['essay_set', 'essay_id', 'essay', 'domain1_score'])
Dataset.save(train_dataset, '../../data/train.pickle')
dev_dataset = Dataset()
dev_dataset.load_from_raw_file("../../data/dev.tsv", ['essay_set', 'essay_id', 'essay', 'domain1_score'])
Dataset.save(dev_dataset, '../../data/dev.pickle')
test_dataset = Dataset()
test_dataset.load_from_raw_file("../../data/test.tsv", ['essay_set', 'essay_id', 'essay'])
Dataset.save(test_dataset, '../../data/test.pickle')

# print(test_dataset.data)
# spell_error(train_dataset.data['3'])
# semantic_vector_similarity(train_dataset.data['3'])
# essay_length(train_dataset.data['1'])
# for data in test_dataset.data['1']:
#    print(type)
#    # Mean_sentence_depth(data)
#    semantic_vector_similarity(data)

# token_sets = [['human', 'interface', 'computer','human'],
#
# ['survey', 'user', 'computer', 'system', 'response', 'time'],
#
# ['eps', 'user', 'interface', 'system'],
#
# ['system', 'human', 'system', 'eps'],
#
# ['user', 'response', 'time'],
#
# ['trees'],
#
# ['graph', 'trees'],
#
# ['graph', 'minors', 'trees'],
#
# ['graph', 'minors', 'survey']]
#
# dictionary = corpora.Dictionary(token_sets)
#
# corpus = [dictionary.doc2bow(tokens) for tokens in token_sets]
# print(corpus)
#
# lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=1)
# documents = lsi_model[corpus]
# print(documents[0])
# print(documents[1])
#
# topics = lsi_model.show_topics(num_words=10, log=0)
# for tpc in topics:
#   print(tpc)
#
# index = MatrixSimilarity(documents)
# query = [(0, 1)]
# q = lsi_model[query]
# print(index[q])

from gensim import corpora
