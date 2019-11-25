import sys
sys.path.append("../..")
from src.feature.iku import spell_error,Mean_sentence_depth
from src.data import Dataset


test_dataset= Dataset.load("../../data/test.pickle")

# print(test_dataset.data)

for data in test_dataset.data['1']:
   Mean_sentence_depth(data)