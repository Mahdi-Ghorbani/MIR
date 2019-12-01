import numpy as np
from typing import Dict
from collections import Counter


class TF_IDF:
    def __init__(self, index: Dict, preprocessor):
        self.index = index
        self.preprocessor = preprocessor

        # token2id: dictionary that maps each word in the vocab to a token
        self.token2id = {token: i for i, token in enumerate(self.index.keys())}
        self.idf = {}
        self._create_tfidf_matrix()

    def find_docs(self, query: str):
        tokenized_query = self.preprocessor.normalize(query)

        query_vector = np.zeros([len(self.index)])

        for token in tokenized_query:
            tf = 1 + np.log10(Counter(tokenized_query)[token])
            if token in self.index:
                query_vector[self.token2id[token]] = tf * self.idf[token]

        query_vector = query_vector / np.sqrt(np.sum(query_vector ** 2))

        max_similarity, best_doc = -10000, None

        for doc_id in self.doc_ids:
            doc_vector = self.tf_idf_matrix[:, doc_id]
            similarity = np.dot(query_vector, doc_vector)

            if max_similarity < similarity:
                max_similarity = similarity
                best_doc = doc_id

        return best_doc

    def _create_tfidf_matrix(self):
        self.doc_ids = set()
        for value in self.index.values():
            self.doc_ids = self.doc_ids.union(set(value['posting'].keys()))

        num_docs = len(self.doc_ids)

        self.tf_idf_matrix = np.zeros([len(self.index), num_docs])

        for token, value in self.index.items():
            num_occur = value['df']
            idf = np.log10(num_docs / len(num_occur))
            self.idf[token] = idf

            for doc_id, occurs in value['posting'].keys():
                self.tf_idf_matrix[self.token2id[token], doc_id - 1] = (1 + np.log(occurs)) / idf

        self.tf_idf_matrix /= np.sqrt(np.sum(self.tf_idf_matrix.sum() ** 2, axis=0, keepdims=True))


    def search(self, query):
        pass

    def proximity_search(self, query):
        pass
