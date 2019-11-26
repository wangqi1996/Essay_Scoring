# encoding=utf-8
import argparse
import time

from sklearn.svm import SVR

from config import TRAIN_DADA_PATH, DEV_DATA_PATH
from src.data import Dataset
from src.feature.feature import Feature
from src.metrics import kappa


def train(contain_test=False):
    """ 训练模型 """
    # 1. 加载数据集
    print("start loading data_set")
    train_dataset: Dataset = Dataset.load(TRAIN_DADA_PATH)
    dev_dataset: Dataset = Dataset.load(DEV_DATA_PATH)
    print("end loading data_set")

    # 2. 计算特征
    essay_set_num = len(train_dataset.data)
    for set_id in range(1, essay_set_num + 1):
        train_data = train_dataset.data[str(set_id)]
        dev_data = dev_dataset.data[str(set_id)]

        print("start compute the feature for essay set ", set_id)
        st = time.time()

        feature = train_dataset.load_feature(set_id)
        feature_class = Feature.get_instance(feature)
        train_sentences_list, train_tokens_list, train_scores = Dataset.get_data_list(train_data, acquire_score=True)

        train_feature = feature_class.get_train_feature(train_sentences_list, train_tokens_list, train_scores)

        train_dataset.save_feature(set_id, feature_class.save_feature(train_feature))
        et = time.time()
        print("end compute the feature for essay set, ", set_id, "time = ", et - st)

        # 3. 构建模型，训练
        clf = model("SVR", train_feature, train_scores, set_id)

        # 4. 测试
        dev_sentences_list, dev_tokens_list, dev_scores = Dataset.get_data_list(dev_data, acquire_score=True)
        dev_feature = feature_class.get_test_feature(dev_sentences_list, dev_tokens_list, train_scores)

        predicted = clf.predict(dev_feature)
        print(kappa(dev_scores, predicted))

        if contain_test:
            pass

        break
    # 保存特征 只能保存dataset对象了
    train_dataset.save(train_dataset, TRAIN_DADA_PATH)


def model(model_name, feature, label, set_id):
    """
    """
    print("start train model for essay set ", set_id)
    clf = None
    if model_name == 'SVR':
        # SVM的回归版本
        clf = SVR(kernel='linear', C=1.0, epsilon=0.2)
        # clf = SVR(kernel='rbf', gamma='scale', epsilon=0.2)
        clf.fit(feature, label.ravel())

    print("end train model for essay set ", set_id)
    return clf


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("--run", type=str, default='train', help='train or test', choices=['train', 'test'])
    parse.add_argument("--model", type=str, default='SVR', help='SVR, ', choices=['SVR'])
    args = parse.parse_args()

    run = args.run

    if run == 'train':
        train(contain_test=True)
    elif run == 'test':
        train(contain_test=False)
    else:
        assert False, u"纳尼，居然还有这个选择能进来"
