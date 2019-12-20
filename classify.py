import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from indexer import Positional
from preprocessor import EnglishProcessor
from searcher import TF_IDF
from utils import read_corpus
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from typing import List
from collections import Counter
from searcher import TF_IDF
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils import search_by_subject


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
                        (len([id for id in doc_ids if y_train[id] == cls_id + 1]) + 1) / (self.counts[cls_id + 1] + self.n_classes)
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
                num_words_class[cls_id] = np.sum([num for doc_id, num in enumerate(num_words_doc) if y_train[doc_id] == cls_id + 1])

            print(num_words_doc)
            print(num_words_class)

            for word, value in self.vocab.items():
                for cls_id in range(self.n_classes):
                    self.p_matrix[self.word2id[word], cls_id] = \
                        (np.sum([term[1] for term in value if y_train[term[0]] == cls_id + 1]) + 1) / (num_words_class[cls_id] + self.n_classes)



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

        #probs /= np.sum(probs, keepdims=True)
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


if __name__ == '__main__':
    train_path = 'data/phase2_train.csv'
    test_path = 'data/phase2_test.csv'
    phase1_path = 'data/English.csv'

    X_train, y_train = read_corpus(train_path)
    X_test, y_test = read_corpus(test_path)
    X_phase1, _ = read_corpus(phase1_path, False)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # using train and test text both to train the dec2vec model
    # doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
    # doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
    # print('training the model...')
    # doc2vecmodel.train(doc2vec_train_corpus)
    # doc2vecmodel.save('model.bin')
    # doc2vecmodel = Doc2VecModel(None, None, None)
    # doc2vecmodel.load('model.bin')
    # docvecs = doc2vecmodel.get_docvecs()

    # This part is for modeling the train data in tf-idf format
    english_preprocessor = EnglishProcessor()
    positional_index = Positional(preprocessor=english_preprocessor)
    positional_index.add_docs(train['Text'])
    tf_idf = TF_IDF(positional_index, english_preprocessor)
    docvecs = np.transpose(tf_idf.tf_idf_matrix)

    # Model the test data to doc2vec
    # inferred_X_test = []
    # for doc in X_test:
    #     inferred_X_test.append(doc2vecmodel.infer(doc))

    # Model the test data to tf-idf
    inferred_X_test = []
    for doc in test['Text']:
        _, vec = tf_idf.search(doc, True)
        inferred_X_test.append(vec.T)

    # nb_clf = NaiveBayesClassifier(4)
    # nb_clf.train(X_train, y_train, use_tf=True)
    # nb_predict = []
    # for x_test in X_test:
    #     porb, pred = nb_clf.infer(x_test)
    #     nb_predict.append(pred)
    #
    # nb_precision = precision_score(y_test, nb_predict, average='micro')
    # nb_recall = recall_score(y_test, nb_predict, average='micro')
    # nb_f1 = f1_score(y_test, nb_predict, average='micro')
    # nb_f1_manually = (2 * nb_precision * nb_recall) / (nb_precision + nb_recall)
    # nb_accuracy = accuracy_score(y_test, nb_predict)
    # print('nb_result: ', nb_predict)
    # print('precision nb: ', nb_precision)
    # print('recall nb: ', nb_recall)
    # print('F1 score nb: ', nb_f1)
    # print('F1 manually nb: ', nb_f1_manually)
    # print('accuracy nb: ', nb_accuracy)
    # print('metrics:')
    # print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
    # print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
    # print('nb metrics: ', find_metrics(y_test, nb_predict))

    knn_predict = knn(docvecs, inferred_X_test, y_train, 5)
    # knn_precision = precision_score(y_test, knn_predict, average='micro')
    # knn_recall = recall_score(y_test, knn_predict, average='micro')
    # knn_f1 = f1_score(y_test, knn_predict, average='micro')
    # knn_f1_manually = (2 * knn_precision * knn_recall) / (knn_precision + knn_recall)
    # knn_accuracy = accuracy_score(y_test, knn_predict)
    # print('knn_result: ', knn_predict)
    # print('precision knn: ', knn_precision)
    # print('recall knn: ', knn_recall)
    # print('F1 score knn: ', knn_f1)
    # print('F1 manually knn: ', knn_f1_manually)
    # print('accuracy knn: ', knn_accuracy)
    # print('metrics:')
    # print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
    # print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
    print('knn metrics: ', find_metrics(y_test, knn_predict))

    svmClf = svm_clf(docvecs, y_train, 1.)
    svm_predict = svmClf.predict(inferred_X_test)
    # svm_precision = precision_score(y_test, svm_predict, average='micro')
    # svm_recall = recall_score(y_test, svm_predict, average='micro')
    # svm_f1 = f1_score(y_test, svm_predict, average='micro')
    # svm_f1_manually = (2 * svm_precision * svm_recall) / (svm_precision + svm_recall)
    # svm_accuracy = accuracy_score(y_test, svm_predict)
    # print('svm_result: ', svm_predict)
    # print('precision svm: ', svm_precision)
    # print('recall svm: ', svm_recall)
    # print('F1 score svm: ', svm_f1)
    # print('F1 manually svm: ', svm_f1_manually)
    # print('accuracy svm: ', svm_accuracy)
    # print('metrics:')
    # print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
    # print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
    print('svm metrics: ', find_metrics(y_test, svm_predict))

    rfClf = random_forest_clf(docvecs, y_train)
    rf_predict = rfClf.predict(inferred_X_test)
    # rf_precision = precision_score(y_test, rf_predict, average='micro')
    # rf_recall = recall_score(y_test, rf_predict, average='micro')
    # rf_f1 = f1_score(y_test, rf_predict, average='micro')
    # rf_f1_manually = (2 * rf_precision * rf_recall) / (rf_precision + rf_recall)
    # rf_accuracy = accuracy_score(y_test, rf_predict)
    # print('rf_result: ', rf_predict)
    # print('precision rf: ', rf_precision)
    # print('recall rf: ', rf_recall)
    # print('F1 score rf: ', rf_f1)
    # print('F1 manually rf: ', rf_f1_manually)
    # print('accuracy rf: ', rf_accuracy)
    # print('metrics:')
    # print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
    # print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
    print('rf metrics: ', find_metrics(y_test, rf_predict))

    # Predict phase1 tags and search by tag number
    # inferred_X_phase1 = []
    # for doc in X_phase1:
    #     inferred_X_phase1.append(doc2vecmodel.infer(doc))
    #
    # nb_predict_phase1 = []
    # for x_test in X_phase1:
    #     porb, pred = nb_clf.infer(x_test)
    #     nb_predict_phase1.append(pred)
    #
    # knn_predict_phase1 = knn(docvecs, inferred_X_phase1, y_train, 5)
    #
    # svm_predict_phase1 = svmClf.predict(inferred_X_phase1)
    #
    # rf_predict_phase1 = rfClf.predict(inferred_X_phase1)
    #
    # print(nb_predict_phase1)
    # print(knn_predict_phase1)
    # print(svm_predict_phase1)
    # print(rf_predict_phase1)
    #
    # print(search_by_subject(nb_predict_phase1, 1))
