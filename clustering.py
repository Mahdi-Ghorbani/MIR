from sklearn.cluster import KMeans
import numpy as np
from utils import read_corpus, tf_idf, save_word2vec_vocab, load_word2vec_vocab, get_document_vectors_word2vec
from gensim.models import Word2Vec, KeyedVectors
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


if __name__ == "__main__":
    path = 'data/Data.csv'
    tokenized_docs, ids = read_corpus(file_path=path, has_tag=False, has_id=True)
    vocab_docs = list(Counter(chain(*tokenized_docs)).keys())

    vocab, word2id, vectors = None, None, None
    if not os.path.exists('word_vectors.npy'):
        word2vec_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
        save_word2vec_vocab(vocab_docs=vocab_docs, word2vec_model=word2vec_model)
    else:
        vocab, word2id, vectors = load_word2vec_vocab()

    tf_idf_matrix = tf_idf(tokenized_docs, vocab, word2id)
    doc_vecs = get_document_vectors_word2vec(tokenized_docs, tf_idf_matrix, word2id, vectors)

