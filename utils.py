import xml.etree.ElementTree as ET
from typing import Set, List
from editdistance import distance
import pandas as pd
from gensim.utils import simple_preprocess
from typing import Dict
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import os


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


def read_corpus(file_path: str, has_tag=True, has_id=False):
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

    if has_id:
        ids = df['ID'].values.tolist()
        return tokens, ids

    return tokens, labels


def tf_idf(docs: List[List[str]], vocab, word2id: Dict):
    """
    :param docs:
    :param vocab: a dictionary with words as keys and their freqs as values
    :return: tf-idf vectors for vocab and documents
    """
    if os.path.exists('tf_idf.npy'):
        return  np.load('tf_idf.npy')

    tf_idf_matrix = np.zeros((len(vocab), len(docs)), dtype=np.float32)
    n_docs = len(docs)
    for word in tqdm(vocab):
        idf = np.log(n_docs / len([1 for doc in docs if word in doc]))
        for i, doc in enumerate(docs):
            tf = np.log(1 + len([w for w in doc if w == word]))
            tf_idf_matrix[word2id[word], i] = tf * idf


    np.save('tf_idf', tf_idf_matrix)

    return tf_idf_matrix


def get_document_vectors_word2vec(tokenized_docs: List[List[str]], tf_idf_matrix: np.ndarray, word2id: Dict,
                                  word_vectors: np.ndarray):
    vocab = list(word2id.keys())
    files = os.listdir(path='.')
    if 'doc_vecs_phase3.npy' in files:
        print('Loading doc vecs...')
        return np.load('doc_vecs_phase3.npy')

    embedding_dim = word_vectors.shape[1]
    doc_vecs = np.zeros((len(tokenized_docs), embedding_dim), dtype=np.float32)
    for doc_id, doc in enumerate(tokenized_docs):
        words = [word for word in doc if word in vocab]
        doc_vecs[doc_id] = np.sum([word_vectors[word2id[word]] * tf_idf_matrix[word2id[word], doc_id]
                                   for word in words])

    np.save('doc_vecs_phase3', doc_vecs)
    return doc_vecs


def save_word2vec_vocab(vocab_docs: List[str], word2vec_model: KeyedVectors):
    valid_words = []
    vecs = []
    for word in vocab_docs:
        if word in word2vec_model.wv.vocab:
            valid_words.append(word)
            vecs.append(word2vec_model[word])

    vecs = np.vstack(vecs)
    np.save('word_vectors', vecs)
    with open('vocab_word2vec.txt', 'w') as f:
        for i, word in enumerate(valid_words):
            f.write(str(i) + ' ' + word + '\n')

    print('Saved...')

def get_document_vectors_tfidf():
    tf_idf_matrix = np.load('doc_vecs_phase3.npy')
    return tf_idf_matrix.T


def load_word2vec_vocab():
    vecs = np.load('word_vectors.npy')

    vocab = []
    word2id = {}
    with open('vocab_word2vec.txt', 'r') as f:
        for row in f:
            index, word = row[:-1].split()
            vocab.append(word)
            word2id[word] = int(index)

    return vocab, word2id, vecs


def search_by_subject(tags: List[int], query_tag: int):
    result = []
    for i in range(len(tags)):
        if tags[i] == query_tag:
            result.append(i + 1)
    return result


if __name__ == '__main__':
    read_corpus('data/phase2_test.csv')
