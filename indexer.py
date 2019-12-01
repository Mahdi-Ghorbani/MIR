import pickle
from utils import get_bigrams


class Positional:
    # index is mapping from terms to a dictionary which is a mapping from doc_frequency to its number
    # and a dictionary which is a mapping from doc_ids to positions
    # {'hi': {'df': 2, 'posting': {1: [12, 16, 23], 3: [45]}}, 'bye': {'df': 2, 'posting': {1: [34, 43], 17: [15, 90]}}}
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.index = {}

    def add_term(self, term, doc_id, position):
        normalized = self.preprocessor.normalize(term)
        if not normalized:
            return
        normalized = normalized[0]
        if normalized in self.index:
            if doc_id in self.index[normalized]['posting']:
                self.index[normalized]['posting'][doc_id].append(position)
            else:
                self.index[normalized]['df'] += 1
                self.index[normalized]['posting'][doc_id] = [position]
        else:
            self.index[normalized] = {'df': 1, 'posting': {doc_id: [position]}}

    def add_doc(self, doc, doc_id):
        tokenized_doc = self.preprocessor.tokenize(doc)
        begin = 0
        for term in tokenized_doc:
            pos = doc.find(term, begin)
            if pos == -1:
                continue
            self.add_term(term=term, doc_id=doc_id, position=pos)
            begin = pos + len(term)

    def add_df(self, df):
        doc_id = 1
        for doc in df:
            self.add_doc(doc=doc, doc_id=doc_id)
            doc_id += 1

    def delete_term(self, term, doc_id=-1, position=-1):
        normalized = self.preprocessor.normalize(term)
        if not normalized:
            return
        normalized = normalized[0]
        if normalized in self.index:
            if doc_id == -1:
                self.index.pop(normalized)
            elif position == -1:
                if doc_id in self.index[normalized]['posting']:
                    self.index[normalized]['df'] -= 1
                    self.index[normalized]['posting'].pop(doc_id)
                    if self.index[normalized]['df'] == 0:
                        self.index.pop(normalized)
            else:
                if doc_id in self.index[normalized]['posting']:
                    self.index[normalized]['posting'][doc_id].remove(position)
                    if not self.index[normalized]['posting'][doc_id]:
                        self.index[normalized]['df'] -= 1
                        self.index[normalized]['posting'].pop(doc_id)
                        if self.index[normalized]['df'] == 0:
                            self.index.pop(normalized)

    def delete_text(self, doc, doc_id=-1):
        tokenized_doc = self.preprocessor.tokenize(doc)
        begin = 0
        for term in tokenized_doc:
            pos = doc.find(term, begin)
            if pos == -1:
                continue
            self.delete_term(term=term, doc_id=doc_id, position=pos)
            begin = pos + len(term)

    def delete_doc(self, doc_id):
        index = self.preprocessor.tokenize(self.index.copy())
        for x in index:
            if doc_id in self.index[x]['posting']:
                self.index[x]['df'] -= 1
                self.index[x]['posting'].pop(doc_id)
                if self.index[x]['df'] == 0:
                    self.index.pop(x)

    def delete_df(self, df):
        doc_id = 1
        for doc in df:
            self.delete_text(doc=doc, doc_id=doc_id)
            doc_id += 1

    def save_to_file(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.index, f)
            f.close()

    def load_from_file(self, name):
        with open(name, 'rb') as f:
            self.index = pickle.load(f)
            f.close()

    def print_result(self):
        print(self.index)

    def find_posting(self, word):
        normalized = self.preprocessor.normalize(word)
        if not normalized:
            return
        normalized = normalized[0]
        if normalized in self.index:
            print(self.index[normalized]['posting'])
        else:
            print("The word is not in the index")


class Bigram:
    # index is mapping from 2 character sequences to a term list
    # {'$h': [hello, hi, ...], 'hi': [hi, high, ...], 'i$': [hi, kiwi, wiki, ...]}

    def __init__(self, preprocessor, positional_index):
        """
        self.index is a dictionary with bigrams as keys and a set of words in which the bigram
        apperas as values.
        :param preprocessor:
        """
        self.index = {}
        self.preprocessor = preprocessor
        self.positional_index = positional_index

    def add(self, term):
        pass

    def add_doc(self, doc: str):
        """
        :param doc: string
        adds the bigrams in the doc to the index
        """
        tokenized_doc = self.preprocessor.normalize(doc)

        for token in tokenized_doc:
            bigrams = get_bigrams(token)
            for bigram in bigrams:
                if bigram in self.index.keys():
                    self.index[bigram].add(token)
                else:
                    self.index[bigram] = {token}  # store the tokens in a set

    def delete(self, term):
        if term not in self.positional_index.keys():
            bigrams = get_bigrams(term)
            for bigram in bigrams:
                if term in self.index[bigram]:
                    self.index[bigram].remove(term)
                    if not self.index[bigram]:
                        self.index.pop(bigram)

    def delete_doc(self, doc):
        terms = self.preprocessor.tokenize(doc)
        for term in terms:
            self.delete(term)

    def save_to_file(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.index, f)
            f.close()

    def load_from_file(self, name):
        with open(name, 'rb') as f:
            self.index = pickle.load(f)
            f.close()

    def print_result(self):
        print(self.index)
