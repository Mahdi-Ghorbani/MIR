from sklearn.cluster import KMeans
import numpy as np
from utils import read_corpus
from gensim.models import Word2Vec
import csv
from typing import List
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts


def kmeans(X: np.ndarray, n_clusters: int, ):
    model = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return model


def gaussian_mixture_model():
    pass


def hierarchical_clustering():
    pass


def word2vec(docs: List[List[str]], vector_size: int, window_size: int, min_count: int):
    model = Word2Vec(sentences=docs, size=vector_size, window=window_size, min_count=min_count, workers=4)
    return model


if __name__ == "__main__":
    path = 'data/Data.csv'
    tokenized_docs, ids = read_corpus(file_path=path, has_tag=False, has_id=True)
    # tokenized_docs: List[List[str]]
    # ids: List[int]
    model = word2vec(tokenized_docs, vector_size=50, window_size=5, min_count=5)
