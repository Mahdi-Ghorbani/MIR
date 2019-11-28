from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import hazm
import xml.etree.ElementTree as ET
import reg
from typing import List
from utils import PERSIAN_GARBAGE


class EnglishProcessor:

    def __init__(self):
        self.preprocessed_df = []

    def normalize(self, text):
        lowered = text.lower()
        # TODO: put same words in same classes, u.s.a and usa and ...
        removed_puncs = self.remove_punctuations(text=lowered)
        tokenized = self.tokenize(text=removed_puncs)
        removed_stopwords = self.remove_stopwords(tokenized_list=tokenized)
        stemmed = self.stem(ulist=removed_stopwords)
        return stemmed

    def tokenize(self, text):
        # TODO: do we need to remove numbers here?
        return word_tokenize(text=text)
        # TODO: it returns a word multiple times (if it appears multiple times in the text),
        #  should we store and print it just 1 time or let it to store and print multiple times?

    def remove_punctuations(self, text):
        return text.translate(str.maketrans(dict.fromkeys(list(string.punctuation))))

    def remove_stopwords(self, tokenized_list):
        return [x for x in tokenized_list if x not in stopwords.words('english')]

    def stem(self, ulist):
        stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        return [stemmer.stem(x) for x in ulist]

    def preprocess(self, df):
        preprocessed_df = []
        for text in df:
            preprocessed_df.append(self.normalize(text=text))
        self.preprocessed_df = preprocessed_df
        return preprocessed_df

    def print_result(self):
        print(self.preprocessed_df)

    def handle_query(self, text):
        normalized = self.normalize(text=text)
        print(normalized)


class PersianProcessor:
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.word_tokenizer = hazm.WordTokenizer()
        self.stemmer = hazm.Stemmer()
        self.stop_words = hazm.stopwords_list()

    def normalize(self, text):
        return self.normalizer.normalize(text)

    def tokenize(self, text):
        return self.word_tokenizer.tokenize(text)

    def remove_punctuations(self, text):
        return text.translate(str.maketrans(dict.fromkeys(list(string.punctuation))))

    def remove_persian_garbage(self, text):
        for old, new in PERSIAN_GARBAGE.items():
            text = text.replace(old, new)

        return text

    def remove_nonpersian_alphabet(self, text):
        pass

    def remove_stopwords(self, tokenized_text: List[str]):
        return [word for word in tokenized_text if word not in self.stop_words]

    def stem(self, word: str):
        return self.stemmer.stem(word)

    def remove_ZWNJ(self, tokenized_text: List[str]):
        return [word.replace(u'\u200c', '') for word in tokenized_text]

    def preprocess(self, df):
        preprocessed_df = []
        for text in df:
            preprocessed_df.append(self.normalize(text=text))
        self.preprocessed_df = preprocessed_df
        return preprocessed_df

    def print_result(self):
        print(self.preprocessed_df)

    def handle_query(self, text):
        normalized = self.normalize(text=text)
        print(normalized)

    def from_xml(self, xml_string):
        """
        :param xml_string: a string with the xml format
        :return: texts parsed from the raw xml data
        """
        texts = []
        for child in ET.fromstring(xml_string):
            for node1 in child:
                if 'revision' in node1.tag:
                    for node2 in node1:
                        if 'text' in node2.tag:
                            texts.append(node2.text)

        return texts
