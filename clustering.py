from sklearn.cluster import KMeans
import numpy as np
from utils import read_corpus, tf_idf
from gensim.models import Word2Vec
import csv
from typing import List, Dict
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
import os


EMBEDDING_DIM = 300


def kmeans(X_train: np.ndarray, n_clusters: int, ):
    print('Fitting kmenas model on the training set...')
    model = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)
    return model


def gaussian_mixture_model():
    pass


def hierarchical_clustering():
    pass


def word2vec(docs: List[List[str]], vector_size: int, window_size: int, min_count: int):
    model = Word2Vec(sentences=docs, size=vector_size, window=window_size, min_count=min_count, workers=4)
    return model


def get_document_vectors_word2vec(tokenized_docs: List[List[str]], tf_idf_matrix: np.ndarray, word2id: Dict, word2vec_model):

    vocab = Counter(chain(*tokenized_docs))

    files = os.listdir(path='.')
    if 'doc_vecs_phase3.np' in files:
        return np.load('doc_vecs.np')

    doc_vecs = np.zeros((len(tokenized_docs)))
    for doc_id, doc in enumerate(tokenized_docs):
        words = np.unique(doc)
        words = [word for word in words if word in word2vec_model.vocab]  #TODO
        doc_vecs[doc_id] = np.sum([word2vec_model[word] * tf_idf_matrix[word2id[word], doc_id] for word in words])

    np.save('doc_vecs', doc_vecs)


def get_document_vectors_tfidf():
    pass


if __name__ == "__main__":
    path = 'data/Data.csv'
    tokenized_docs, ids = read_corpus(file_path=path, has_tag=False, has_id=True)
    #print(tokenized_docs[:10])
    # tokenized_docs: List[List[str]]
    # ids: List[int]
    #model = word2vec(tokenized_docs, vector_size=50, window_size=5, min_count=5)

    #model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative50.bin', binary=True)
