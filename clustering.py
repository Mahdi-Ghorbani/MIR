from sklearn.cluster import KMeans
import numpy as np
from utils import read_corpus, tf_idf, save_word2vec_vocab, load_word2vec_vocab, get_document_vectors_word2vec, \
    get_document_vectors_tfidf
from gensim.models import Word2Vec, KeyedVectors
from typing import List, Dict
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
from gensim.test.utils import common_texts
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import os


EMBEDDING_DIM = 300


def kmeans(X: np.ndarray, n_clusters: int):
    print('Fitting kmenas model on the training set...')
    model = KMeans(n_clusters=n_clusters, random_state=0)
    return model.fit_predict(X)


def gaussian_mixture_model(X: np.ndarray, n_components: int):
    print('Fitting Gaussian mixture model on the training set...')
    model = GaussianMixture(n_components=n_components)
    return model.fit_predict(X)


def hierarchical_clustering(X: np.ndarray, n_clusters: int):
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    return model.fit_predict(X)


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
    #model = kmeans(X_train=doc_vecs[:4000], n_clusters=4)
    #doc_vecs = get_document_vectors_tfidf()
    #model = kmeans(X_train=doc_vecs[:4000], n_clusters=4)
    #predictions = model.predict(doc_vecs[4000:])

    #model = gaussian_mixture_model(doc_vecs[:4000], n_components=4)
    #print(model.predict(doc_vecs[4000:]))

    print(hierarchical_clustering(X=doc_vecs[:4000], n_clusters=10))
