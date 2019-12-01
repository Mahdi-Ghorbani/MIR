import pickle
from sys import getsizeof


class VariableByte:

    def __init__(self, positional_index):
        self.index = positional_index
        self.compressed = {}

    def vb_encode(self, number):
        vb = []
        flag = True
        while True:
            binary = bin(number % 128)[2:]
            number = number // 128
            if flag:
                binary = '1' + '0' * (8 - len(binary) - 1) + binary
                vb.append(int(binary, 2))
                flag = False
            else:
                binary = '0' * (8 - len(binary)) + binary
                vb.append(int(binary, 2))
            if number == 0:
                break
        vb.reverse()
        return bytes(vb)

    def vb_decode(self, code):
        x = int.from_bytes(code, 'big')
        bin_string = ''
        while True:
            binary = bin(x % 256)[2:]
            if len(binary) == 8:
                bin_string = binary[1:] + bin_string
            else:
                bin_string = binary + bin_string
            x = x // 256
            if x == 0:
                return int(bin_string, 2)

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

    def decompress(self):
        index = {}
        for term in self.compressed:
            index[term] = {'df': self.compressed[term]['df'], 'posting': {}}
            byte_array = []
            prev_value = 0
            cnt = 0
            for byte in self.compressed[term]['posting'][0]:
                byte_array.append(byte)
                if len(bin(byte)[2:]) == 8:
                    doc_id = self.vb_decode(bytes(byte_array)) + prev_value
                    index[term]['posting'][doc_id] = []
                    prev_value = doc_id
                    byte_array.clear()
                    pos_byte_array = []
                    prev_pos = 0
                    for byte_ in self.compressed[term]['posting'][1][cnt]:
                        pos_byte_array.append(byte_)
                        if len(bin(byte)[2:]) == 8:
                            position = self.vb_decode(bytes(pos_byte_array)) + prev_pos
                            index[term]['posting'][doc_id].append(position)
                            prev_pos = position
                            pos_byte_array.clear()
                    cnt += 1
        return index

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

    def __init__(self, positional_index):
        self.index = positional_index
        self.compressed = {}

    def gamma_encode(self, number):
        if number == 1:
            return [0]
        binary = bin(number)[2:]
        gamma_str = '1' * (len(binary) - 1) + '0' + binary[1:]
        res = []
        while True:
            if len(gamma_str) >= 32:
                res.append(int(gamma_str[0:31], 2))
                gamma_str = gamma_str[32:]
            else:
                res.append(int(gamma_str, 2))
                break
        return res

    def gamma_decode(self, code):
        string_res = ''
        res = []
        for x in code:
            string_res += bin(x)[2:]
        cnt = 0
        prev_x = -1
        i = 0
        while i < len(string_res):
            if string_res[i] == '1':
                cnt += 1
                i += 1
            else:
                x = int('1' + string_res[i + 1: i + cnt + 1], 2) + prev_x
                res.append(x)
                prev_x = x
                i = i + cnt + 1
                cnt = 0
        return res

    def compress(self):
        # compressed is a mapping from term to a dictionary which is a mapping from df to its value and
        # a mapping from posting to an array which its first element is gamma encoding of the doc_ids and
        # the second element is an array where each element in that is the gamma encoding of the positions
        # note that as it is not possible to encode 0 with gamma encoding, so we add 1 to first position
        # (to make it start from 1 instead of 0)
        # {'hi': {'df': 2, 'posting': {1: [2, 5], 3: [7, 13]}}}  ---->
        # {'hi': {'df': 2, 'posting': [0 100, [101 101, 1110000 11010]]}}
        compressed = {}
        for term in self.index:
            compressed[term] = {'df': self.index[term]['df'], 'posting': [[], []]}
            prev_doc = -1
            for doc_id in self.index[term]['posting']:
                if prev_doc == -1:
                    enc = self.gamma_encode(doc_id)
                else:
                    enc = self.gamma_encode(doc_id - prev_doc)
                compressed[term]['posting'][0].extend(enc)
                prev_doc = doc_id
                prev_pos = -1
                for pos in self.index[term]['posting'][doc_id]:
                    if prev_pos == -1:
                        compressed[term]['posting'][1].append(self.gamma_encode(pos + 1))
                    else:
                        compressed[term]['posting'][1][len(compressed[term]['posting'][1]) - 1] += \
                            self.gamma_encode(pos - prev_pos)
                    prev_pos = pos
        self.compressed = compressed
        return compressed

    def decompress(self):
        index = {}
        for term in self.compressed:
            index[term] = {'df': self.compressed[term]['df'], 'posting': {}}
            doc_ids = self.gamma_decode(self.compressed[term]['posting'][0])
            positions = []
            for pos in self.compressed[term]['posting'][1]:
                positions.append(self.gamma_decode(pos))
            for i in range(len(doc_ids)):
                index[term]['posting'][doc_ids[i] + 1] = positions[i]
        return index

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
