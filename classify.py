from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import read_corpus
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


class Doc2VecModel:
    def __init__(self, vector_size, min_count, epochs):
        self.model = Doc2Vec(vector_size=vector_size, min_count=min_count, window=2, epochs=epochs, workers=4)

    def save(self, path: str):
        print('saving the model to %s...' % path)
        self.model.save(path)

    def load(self, fname):
        self.model = Doc2Vec.load(fname)

    def train(self, train_corpus):
        self.model.build_vocab(train_corpus)
        self.model.train(train_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)


def svm_clf(X_train: np.ndarray, y_train: np.ndarray):
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    return clf


def naive_bayes_clf(X_train: np.ndarray, y_train: np.ndarray):
    pass


def random_forest_clf(X_train: np.ndarray, y_train: np.ndarray):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    return clf


def knn(X_train, y_train, n_iter: int):
    for i in range(n_iter):



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
