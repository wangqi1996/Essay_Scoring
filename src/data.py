import pandas as pd
import spacy
from tqdm import tqdm
import pickle

from src.config import TOKENIZER_FIELD, SCORE_FIELD
import numpy as np


class Dataset:

    def __init__(self):
        self.data = {}
        self.normalize_dict = {}
        self.feature = {}  # 存储feature和中间变量

    def load_from_raw_file(self, filename, field_require):
        tokenizer = spacy.load("en_core_web_sm")

        data = pd.read_csv(filename, delimiter='\t')
        essay_set = set(data['essay_set'])
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
        for set_id in tqdm(essay_set):

            set_df = data[data.essay_set == set_id]
            fields = [set_df[field] for field in field_require]

            self.normalize_dict[str(set_id)] = {}
            self.data[str(set_id)] = []

            for values in tqdm(zip(*fields), total=len(fields[0])):
                sample_dict = {}
                for i, field in (enumerate(field_require)):
                    if field == 'essay':
                        tokens = [token.text for token in tokenizer(values[i])]
                        sample_dict[field + '_token'] = tokens
                    sample_dict[field] = values[i]

                self.data[str(set_id)].append(sample_dict)

    def normalize_feature(self, set_id, field, normalize_dict=None):
        if normalize_dict is None:
            normalize_dict = self.normalize_dict

        min_value = normalize_dict[set_id][field]['min']
        max_value = normalize_dict[set_id][field]['max']
        for sample in self.data[set_id]:
            sample[field] = (sample[field] - min_value) / (max_value - min_value)

    def save_feature(self, set_id, data):
        self.feature.setdefault(str(set_id), {})
        self.feature[str(set_id)].update(data)

    def load_feature(self, set_id):
        return self.feature.get(str(set_id), {})

    @staticmethod
    def get_data_list(data, acquire_score=True):
        """ 根据eaasy_id获取tokenizer后的data和label """
        essay_list = data[:10]

        sentences_set = []
        token_set = []
        score_list = []

        for essay_dict in essay_list:
            essay = essay_dict[TOKENIZER_FIELD]
            score = essay_dict[SCORE_FIELD]
            token_set.append(essay)
            sentences_set.append(' '.join(essay))
            # test data没有score这个key
            if acquire_score:
                score_list.append(score)

        sample_num = len(score_list)
        # sentences_array = np.array(sentences_set)
        # tokens_array = np.array(token_set)
        scores = np.array(score_list).reshape(sample_num, 1)

        return sentences_set, token_set, scores

    @staticmethod
    def save(dataset, path):
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

# load_from_raw_file includes tokenize process, which is time consuming
#
# train_dataset = Dataset()
# train_dataset.load_from_raw_file('../data/train.tsv', ['essay_set', 'essay_id', 'essay', 'domain1_score'])
# Dataset.save(train_dataset, '../data/train.pickle')
# # #
# dev_dataset = Dataset()
# dev_dataset.load_from_raw_file('../data/dev.tsv', ['essay_set', 'essay_id', 'essay', 'domain1_score'])
# Dataset.save(dev_dataset, '../data/dev.pickle')
# # #
# #
#
# test_dataset = Dataset()
# test_dataset.load_from_raw_file('../data/test.tsv', ['essay_set', 'essay_id', 'essay'])
# Dataset.save(test_dataset, '../data/test.pickle')

# train_dataset = Dataset.load("../data/essay_data/train.pickle")
# dev_dataset = Dataset.load("../data/essay_data/dev.pickle")
# test_dataset = Dataset.load("../data/essay_data/test.pickle")

# dataset.data is a dictionary, keys are {1, 2, 3..., 8} means eight essay sets.
# the value of dict is a list, contains the samples of each essay set
# the element of each list is a dictionary, keys are attribute
#
# dataset.data = {'1': , '2': , ..., '8': }
#
# dataset.data['1'] = [
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# ...
# ]
#
# dataset.data['2'] = [
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# ...
# ]
# ...
#
