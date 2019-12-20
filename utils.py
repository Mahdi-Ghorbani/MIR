import xml.etree.ElementTree as ET
from typing import Set, List
from editdistance import distance
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument


def from_xml(xml_file):
    """
    :param xml_string: a string with the xml format
    :return: texts parsed from the raw xml data
    """
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_string = f.read()

    texts = []
    for child in ET.fromstring(xml_string):
        for node1 in child:
            if 'revision' in node1.tag:
                for node2 in node1:
                    if 'text' in node2.tag:
                        texts.append(node2.text)

    return texts


def jaccard_similarity(A: Set, B: Set):
    union = A.union(B)
    intersection = A.intersection(B)

    return len(intersection) / len(union)


def edit_distance(str1: str, str2: str):
    return distance(str1, str2)


def get_bigrams(token: str):
    token = "$" + token + "$"
    return [token[i:i + 2] for i in range(len(token) - 1)]


def read_corpus(file_path: str, has_tag=True):
    """
    Function used for reading the documents of phase 2
    :param file: path to the documents
    :return: return documents and their tags
    """
    with open(file_path) as f:
        df = pd.read_csv(file_path)

    texts = df['Text'].values.tolist()
    tokens = [simple_preprocess(text) for text in texts]  # List[List[str]]
    labels = None
    if has_tag:
        labels = df['Tag'].values.tolist()

    return tokens, labels


def tf_idf():
    pass


def search_by_subject(tags: List[int], query_tag: int):
    result = []
    for i in range(len(tags)):
        if tags[i] == query_tag:
            result.append(i + 1)
    return result


if __name__ == '__main__':
    read_corpus('data/phase2_test.csv')
