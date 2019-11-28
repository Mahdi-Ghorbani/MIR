import pickle
from sys import getsizeof


class VariableByte:

    def __init__(self, positional_index):
        self.index = positional_index
        self.compressed = {}

    def vb_encode(self, number):
        vb = []
        while True:
            binary = bin(number % 128)[2:]
            number = number // 128
            if number == 0:
                binary = '1' + '0' * (8 - len(binary) - 1) + binary
                vb.append(int(binary, 2))
                break
            else:
                binary = '0' * (8 - len(binary)) + binary
                vb.append(int(binary, 2))
        return bytes(vb)

    def vb_decode_index(self):
        pass

    def compress(self):
        # compressed is a mapping from term to a dictionary which is a mapping from df to its value and
        # a mapping from posting to an array which its first element is vb encoding of the doc_ids and
        # the second element is an array where each element in that is the vb encoding of the positions
        # {'hi': {'df': 2, 'posting': {1: [2, 5], 3: [7, 13]}}}  ---->
        # {'hi': {'df': 2, 'posting': [\x81 \x82, [\x82 \x83, \x87 \x86]]}}
        compressed = {}
        for term in self.index:
            compressed[term] = {'df': self.index[term]['df'], 'posting': [bytes(0), []]}
            prev_doc = -1
            for doc_id in self.index[term]['posting']:
                if prev_doc == -1:
                    enc = self.vb_encode(doc_id)
                else:
                    enc = self.vb_encode(doc_id - prev_doc)
                compressed[term]['posting'][0] += enc
                prev_doc = doc_id
                prev_pos = -1
                for pos in self.index[term]['posting'][doc_id]:
                    if prev_pos == -1:
                        compressed[term]['posting'][1].append(self.vb_encode(pos))
                    else:
                        compressed[term]['posting'][1][len(compressed[term]['posting'][1]) - 1] += \
                            self.vb_encode(pos - prev_pos)
                    prev_pos = pos
        self.compressed = compressed
        return compressed

    def save_to_file(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.compressed, f)
            f.close()

    def load_from_file(self, name):
        with open(name, 'rb') as f:
            self.compressed = pickle.load(f)
            f.close()

    def print_result(self):
        print(self.compressed)

    def print_used_space(self):
        print("Size of index before compressing:", getsizeof(pickle.dumps(self.index)), "bytes")
        print("Size of index after variable byte compressing:", getsizeof(pickle.dumps(self.compressed)), "bytes")
        print("Amount of reduction:", getsizeof(pickle.dumps(self.index)) - getsizeof(pickle.dumps(self.compressed)),
              "bytes")


class GammaCode:

    def compress(self, index):
        pass

    def save_to_file(self):
        pass

    def load_from_file(self):
        pass

    def print_result(self):
        pass
