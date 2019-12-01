import numpy as np
from typing import Dict
from collections import Counter


class TF_IDF:
    def __init__(self, positional_index, preprocessor):
        self.positional_index = positional_index
        self.preprocessor = preprocessor
        # token2id: dictionary that maps each word in the vocab to a token
        self.token2id = {token: i for i, token in enumerate(positional_index.index.keys())}
        self.idf = {}
        self._create_tfidf_matrix()

    def search(self, query: str):
        tokenized_query = self.preprocessor.normalize(query)

        query_vector = np.zeros((len(self.positional_index.index),), dtype=np.float32)

        for token in tokenized_query:
            tf = 1 + np.log10(Counter(tokenized_query)[token])
            if token in self.positional_index.index:
                query_vector[self.token2id[token]] = tf * self.idf[token]

        if np.sqrt(np.sum(query_vector ** 2)) > 0:
            query_vector = query_vector / np.sqrt(np.sum(query_vector ** 2))

        max_similarity, best_doc = -10, None

        for doc_id in self.doc_ids:
            doc_vector = self.tf_idf_matrix[:, doc_id-1]
            similarity = np.dot(query_vector, doc_vector)

            if max_similarity < similarity:
                max_similarity = similarity
                best_doc = doc_id

        return best_doc

    def _create_tfidf_matrix(self):
        self.doc_ids = set()
        for value in self.positional_index.index.values():
            self.doc_ids = self.doc_ids.union(set(value['posting'].keys()))

        num_docs = len(self.doc_ids)

        self.tf_idf_matrix = np.zeros([len(self.positional_index.index), num_docs])

        for token, value in self.positional_index.index.items():
            num_occur = value['df']
            idf = np.log10(num_docs / num_occur)
            self.idf[token] = idf

            for doc_id, occurs in value['posting'].items():
                self.tf_idf_matrix[self.token2id[token], doc_id - 1] = (1 + np.log(len(occurs))) / idf

        self.tf_idf_matrix /= np.sqrt(np.sum(self.tf_idf_matrix.sum() ** 2, axis=0, keepdims=True))

    def proximity_search(self, query):
        pass
