from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from typing import List
from collections import Counter


class Doc2VecModel:
    def __init__(self, vector_size, min_count, epochs):
        if vector_size is None:
            return
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

    def get_docvecs(self):
        docvecs = []
        for i in range(len(self.model.docvecs)):
            docvecs.append(self.model.docvecs[i])
        return docvecs


class NaiveBayesClassifier:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def train(self, X_train: List[List[str]], y_train: List[int], use_tf: bool = None):
        self.counts = Counter(y_train)  # counting number of occurrences of each class (1 to 4)
        self.p_y = [self.counts[cls] for cls in range(1, self.n_classes + 1)]

        if not use_tf:
            self.vocab = {}
            for doc_id, doc in enumerate(X_train):
                for word in doc:
                    if word in self.vocab:
                        self.vocab[word].append(doc_id)
                    else:
                        self.vocab[word] = [doc_id]

            self.word2id = {word: i for i, word in enumerate(self.vocab)}
            self.p_matrix = np.zeros((len(self.vocab), self.n_classes))  # parameters matrix for naive bayes classifier

            for word, doc_ids in self.vocab.items():
                for cls_id in range(self.n_classes):
                    self.p_matrix[self.word2id[word], cls_id] = \
                        (len([id for id in doc_ids if y_train[id] == cls_id + 1]) + 1) / (
                                    self.counts[cls_id + 1] + self.n_classes)
        else:
            self.vocab = {}
            num_words_doc = []  # this list will contain the number of words in the docs

            for doc_id, doc in enumerate(X_train):
                words = Counter(doc)
                num_words_doc.append(len(doc))
                for word, n_occur in words.items():
                    if word in self.vocab:
                        self.vocab[word].append((doc_id, n_occur))
                    else:
                        self.vocab[word] = [(doc_id, n_occur)]

            self.word2id = {word: i for i, word in enumerate(self.vocab)}
            self.p_matrix = np.zeros((len(self.vocab), self.n_classes))  # parameters matrix for naive bayes classifier

            num_words_class = np.zeros((self.n_classes,), dtype=np.int32)

            # counting the number of words in the documents of the i-th category
            for cls_id in range(self.n_classes):
                num_words_class[cls_id] = np.sum(
                    [num for doc_id, num in enumerate(num_words_doc) if y_train[doc_id] == cls_id + 1])

            for word, value in self.vocab.items():
                for cls_id in range(self.n_classes):
                    self.p_matrix[self.word2id[word], cls_id] = \
                        (np.sum([term[1] for term in value if y_train[term[0]] == cls_id + 1]) + 1) / (num_words_class[
                            cls_id] + self.n_classes)

    def infer(self, doc: List[str]):
        """
        :param doc: a new document to be classified as a list of words
        :return: P(y_i=1|doc) as a numpy array with shape (n_classes,)
        """
        log_probs = np.log(self.p_y.copy())
        for word in doc:
            if word in self.vocab:
                log_probs += np.log(self.p_matrix[self.word2id[word]])
            else:
                log_probs += np.log(1.0 / self.n_classes)

        # probs /= np.sum(probs, keepdims=True)
        pred = np.argmax(log_probs) + 1
        return log_probs, pred


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
    closest_y = []
    for near in k_nearest:
        y_trains = []
        for index in near:
            y_trains.append(y_train[index])
        closest_y.append(y_trains)
    y_pred = np.array(list(map(lambda x: np.argmax(np.bincount(x)), closest_y)))

    return y_pred


def find_metrics(y_test, y_predict):
    tp = np.zeros(5, dtype=int)
    tn = np.zeros(5, dtype=int)
    fp = np.zeros(5, dtype=int)
    fn = np.zeros(5, dtype=int)

    for y, y_hat in zip(y_test, y_predict):
        if y == y_hat:
            tp[y] += 1
            tp[0] += 1
            tn[0] += 1
            for x in range(1, 5):
                if x != y:
                    tn[x] += 1
        else:
            fn[y] += 1
            fp[y_hat] += 1
            fn[0] += 1
            fp[0] += 1

    precision = np.divide(tp, np.add(tp, fp))
    recall = np.divide(tp, np.add(tp, fn))
    f1 = 2 * np.divide(np.multiply(precision, recall), np.add(precision, recall))
    accuracy = np.divide(np.add(tp, tn), np.add(tp, np.add(fp, np.add(tn, fn))))

    return precision, recall, f1, accuracy
