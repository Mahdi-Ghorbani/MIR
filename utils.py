import xml.etree.ElementTree as ET
from typing import Set
from editdistance import distance


def from_xml(xml_string):
    """
    :param xml_string: a string with the xml format
    :return: texts parsed from the raw xml data
    """
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
    print(union)
    intersection = A.intersection(B)
    print(intersection)

    return len(intersection) / len(union)


def edit_distance(str1: str, str2: str):
    return distance(str1, str2)


def get_bigrams(token: str):
    token = "$" + token + "$"
    return [token[i:i+2] for i in range(len(token) - 1)]

def tf_idf():
    pass
