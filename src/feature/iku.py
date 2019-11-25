from spellchecker import SpellChecker
import stanfordnlp
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import *
import numpy

def spell_error(data):
    """
    count the number of spelling errors in each essay
    :param data: (dict)
    :return: number of spelling errors (int)
    """
    spell = SpellChecker()
    essay = data['essay_token']
    misspelled = spell.unknown(essay)
    return len(misspelled)

def count_tree_depth(root):
    def travel(root, length, res):
        if type(root) == str:
            res.append(length)
            return
        l = len(root)
        for i in range(l):
            if root[i]:
                travel(root[i],length+1,res)
    if root is None:
        return []
    l=[]
    travel(root,1,l)
    return l




def Mean_sentence_depth(data):
    """
    Sentence depth in the parser tree
    :param data: (dict)
    :return:
    """
    # nlp = stanfordnlp.Pipeline()
    nlp = StanfordCoreNLP(r'/Users/ikuc/stanfordnlp_resources/stanford-corenlp-full-2016-10-31')

    essay = data['essay']
    # doc = nlp(essay)
    # for sentence in doc.sentences:
    #     print(sentence)
    #     print('')
    sentences = sent_tokenize(essay)
    for sentence in sentences:
        parse_tree = nlp.parse(sentence)
        tree = Tree.fromstring(parse_tree)
        distance = count_tree_depth(tree)
        distance = numpy.array(distance)
        depth = numpy.sum(distance)
        level = numpy.amax(distance)
        print(sentence)
        print(depth,level)










