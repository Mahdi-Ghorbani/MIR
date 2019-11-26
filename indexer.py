class Positional:
    # index is mapping from terms to a dictionary which is a mapping from doc_frequency to its number
    # and a dictionary which is a mapping from doc_ids to positions
    # {'hi': {'df': 2, 'posting': {1: [12, 16, 23], 3: [45]}}, 'bye': {'df': 2, 'posting': {1: [34, 43], 17: [15, 90]}}}
    index = {}

    def add(self, term, doc_id, position):
        pass

    def add_doc(self, doc):
        pass

    def delete(self, term):
        pass

    def delete_doc(self, doc):
        pass

    def save_to_file(self):
        pass

    def read_from_file(self):
        pass

    def print_result(self):
        pass

    def handle_query(self):
        pass


class Bigram:
    # index is mapping from 2 character sequences to a term list
    # {'$h': [hello, hi, ...], 'hi': [hi, high, ...], 'i$': [hi, kiwi, wiki, ...]}
    index = {}

    def add(self, term):
        pass

    def add_doc(self, doc):
        pass

    def delete(self, term):
        pass

    def delete_doc(self, doc):
        pass

    def save_to_file(self):
        pass

    def read_from_file(self):
        pass

    def print_result(self):
        pass

    def handle_query(self):
        pass
