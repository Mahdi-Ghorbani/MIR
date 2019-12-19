from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import read_corpus
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from typing import List
from collections import Counter
from itertools import chain


class Doc2VecModel:
    def __init__(self, vector_size, min_count, epochs):
        self.model = Doc2Vec(vector_size=vector_size, min_count=min_count, window=2, epochs=epochs, workers=4)

    def save(self, path: str):
        print('saving the model to %s...' % path)
        self.model.save(path)

    def load(self, path: str):
        self.model = Doc2Vec.load(path)

    def train(self, train_corpus):
        self.model.build_vocab(train_corpus)
        self.model.train(train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def infer(self, doc_words: List[str]):
        """
        :param doc_words: list of the words in the document to be vectorized
        :return: a vector with shape (vector_size, ) as the representation of the document
        """
        return self.model.infer_vector(doc_words=doc_words)


class NaiveBayesClassifier:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def train(self, X_train: List[List[str]], y_train: List[int]):
        self.counts = Counter(y_train)  # counting number of occurrences of each class (1 to 4)
        self.p_y = [self.counts[cls] for cls in range(self.n_classes)]

        self.vocab = {}
        for doc_id, doc in enumerate(X_train):
            for word in doc:
                if word in self.vocab:
                    self.vocab[word].append(doc_id)
                else:
                    self.vocab[word] = [doc_id]

        self.word2id = {word:i for i, word in enumerate(self.vocab)}
        self.p_matrix = np.zeros(len(self.vocab), self.n_classes)  # parameters matrix for naive bayes classifier

        for word, doc_ids in self.vocab.items():
            for cls_id in range(self.n_classes):
                self.p_matrix[self.word2id[word], cls_id] = \
                    len([id for id in doc_ids if y_train[id] == cls_id]) / self.counts[cls_id]

    def infer(self, doc: List[str]):
        """
        :param doc: a new document to be classified as a list of words
        :return: P(y_i=1|doc) as a numpy array with shape (n_classes,)
        """
        probs = self.p_y.copy()
        for word in doc:
            if word in self.vocab:  # TODO: change the implementation so that it works for out of vocabulary words too
                for cls in range(self.n_classes):
                    probs[cls] *= self.p_matrix[self.word2id[word], cls]

        probs /= np.sum(probs, keepdims=True)
        return probs


def svm_clf(X_train: np.ndarray, y_train: np.ndarray, C: float):
    """
    trains an SVM classifier on the documents
    :param X_train: documents as a numpy array with shape (n_samples_a, n_samples)
    :param y_train: labels as a numpy array with shape (n_samples,)
    :return: SVM classifier fitted on the X_train and y_train
    """
    clf = svm.SVC(C=C)
    clf.fit(X_train, y_train)

    return clf


def random_forest_clf(X_train: np.ndarray, y_train: np.ndarray):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    return clf


def knn(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, k: int):
    """
    :param X_train: Design matrix as a numpy array with shape (n_samples_a, n_dim)
    :param y_train: labels matrix as a numpy array with shape (n_samples_a,)
    :param k: k in KNN
    :param X_test: test samples to be classified as a numpy array with shape (n_samples_b, n_dim)
    :return: y_pred as a numpy array with shape (n_samples_b,)
    """
    dists = pairwise_distances(X_test, X_train)  # shape: (n_samples_a, n_samples_b)
    k_nearest = dists.argsort(axis=1)[:, :k]
    closest_y = y_train[k_nearest]
    y_pred = np.array(list(map(lambda x: np.argmax(np.bincount(x)), closest_y)))

    return y_pred


if __name__ == '__main__':
    train_path = 'data/phase2_train.csv'
    test_path = 'data/phase2_test.csv'

    X_train, y_train = read_corpus(train_path)
    X_test, y_test = read_corpus(test_path)

    # using train and test text both to train the dec2vec model
    doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
    dec2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
    print('training the model...')
    dec2vecmodel.train(doc2vec_train_corpus)
    dec2vecmodel.save('model.bin')
