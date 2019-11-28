# encoding=utf-8
import sys

import numpy as np
sys.path.append("..")

import argparse
import time

from sklearn.svm import SVR

from src.config import TRAIN_DADA_PATH, DEV_DATA_PATH, TEST_DATA_PATH
from src.data import Dataset
from src.feature.feature import Feature
from src.metrics import kappa
import pandas as pd


def train(contain_test=False, use_save=False):
    """ 训练模型 """
    # 1. 加载数据集
    print("start loading data_set")
    train_dataset: Dataset = Dataset.load(TRAIN_DADA_PATH)
    dev_dataset: Dataset = Dataset.load(DEV_DATA_PATH)
    test_dataset: Dataset = Dataset.load(TEST_DATA_PATH)
    print("end loading data_set")

    # 2. 计算特征
    essay_set_num = len(train_dataset.data)
    print(essay_set_num)
    mean_qwk = 0  # 8个指标的平均
    all_test_sample = []  # 按照固定格式保存test
    for set_id in range(1, essay_set_num + 1):
        train_data = train_dataset.data[str(set_id)]
        dev_data = dev_dataset.data[str(set_id)]
        test_data = test_dataset.data[str(set_id)]

        print("start compute the feature for essay set ", set_id)
        st = time.time()

        feature = train_dataset.load_feature(set_id)
        feature_class = Feature.get_instance(feature)
        train_sentences_list, train_tokens_list, train_scores = Dataset.get_data_list(train_data, acquire_score=True)

        if use_save:
            train_feature = feature_class.get_save_train_feature()
        else:
            train_feature = feature_class.get_train_feature(train_sentences_list, train_tokens_list, train_scores,
                                                            train_data)
            train_dataset.save_feature(set_id, feature_class.save_feature(train_feature))

        et = time.time()
        print("end compute the feature for essay set, ", set_id, "time = ", et - st)

        # 3. 构建模型，训练
        # print(train_scores.shape)
        clf = model("SVR", train_feature, train_scores, set_id)

        # 4. 测试
        dev_sentences_list, dev_tokens_list, dev_scores = Dataset.get_data_list(dev_data, acquire_score=True)
        dev_feature = feature_class.get_test_feature(dev_sentences_list, dev_tokens_list, train_scores, train_data,
                                                     dev_data)

        print('dev ends')
        predicted = clf.predict(dev_feature)
        qwk = kappa(dev_scores, predicted, weights='quadratic')
        print(set_id, qwk)
        mean_qwk += qwk

        if contain_test:
            test_sentences_list, test_tokens_list = Dataset.get_data_list(test_data, acquire_score=False)
            test_feature = feature_class.get_test_feature(test_sentences_list, test_tokens_list, train_scores,
                                                          train_data, test_data)
            test_predicted = clf.predict(test_feature)

        for idx, sample in enumerate(test_data):
            # sample['domain1_score'] = int(test_predicted[idx])
            sample['domain1_score'] = int(np.round(float(test_predicted[idx])))
        all_test_sample.extend(test_data)

    save_to_tsv(all_test_sample, '../MG1933004.tsv')
    print(mean_qwk / essay_set_num)

    # break
    # 保存特征 只能保存dataset对象了
    train_dataset.save(train_dataset, TRAIN_DADA_PATH)


def save_to_tsv(samples: list, tsv_file):
    raw_data = {
        'id': [sample['essay_id'] for sample in samples],
        'set': [sample['essay_set'] for sample in samples],
        'score': [sample['domain1_score'] for sample in samples]
    }
    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file, sep='\t', index=False, header=False)


def model(model_name, feature, label, set_id):
    """
    """
    print("start train model for essay set ", set_id)
    clf = None
    if model_name == 'SVR':
        # SVM的回归版本
        clf = SVR(kernel='linear', C=1.0, epsilon=0.2)
        # clf = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.2)
        clf.fit(feature, label.ravel())

    print("end train model for essay set ", set_id)
    return clf


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--run", type=str, default='train', help='train or test', choices=['train', 'test'])
    parse.add_argument("--model", type=str, default='SVR', help='SVR, ', choices=['SVR'])
    parse.add_argument("--use_save", type=bool, default=False, help='use saved feature or not')
    args = parse.parse_args()

    run = args.run
    use_save = args.use_save

    if run == 'train':
        train(contain_test=True, use_save=use_save)
    elif run == 'test':
        train(contain_test=False, use_save=use_save)
    else:
        assert False, u"纳尼，居然还有这个选择能进来"
