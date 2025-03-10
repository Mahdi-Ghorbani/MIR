from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import hazm
import re
from typing import List
from collections import Counter
from itertools import chain
from utils import from_xml
from tqdm import tqdm


class EnglishProcessor:

    def __init__(self):
        self.preprocessed_df = []

    def normalize(self, text):
        lowered = text.lower()
        removed_puncs = self.remove_punctuations(text=lowered)
        tokenized = self.tokenize(text=removed_puncs)
        removed_stopwords = self.remove_stopwords(tokenized_list=tokenized)
        stemmed = self.stem(ulist=removed_stopwords)
        return stemmed

    def tokenize(self, text):
        return word_tokenize(text=text)

    def remove_punctuations(self, text):
        return text.translate(str.maketrans(dict.fromkeys(list(string.punctuation))))

    def remove_stopwords(self, tokenized_list):
        return [x for x in tokenized_list if x not in stopwords.words('english')]

    def stem(self, ulist):
        stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        return [stemmer.stem(x) for x in ulist]

    def preprocess(self, df):
        self.preprocessed_df = []

        with tqdm(total=len(df)) as pbar:
            for text in df['Text']:
                self.preprocessed_df.append(self.normalize(text=text))
                pbar.update(n=1)
        pbar.close()

        return self.preprocessed_df

    def print_result(self):
        print(self.preprocessed_df)

    def find_stopwords(self, doc=None):
        if doc is None:
            all_words = list(chain(*self.preprocessed_df))
            return Counter(all_words)
        else:
            tokens = self.normalize(doc)
            return Counter(tokens)

    def handle_query(self, text):
        normalized = self.normalize(text=text)
        return normalized


class PersianProcessor:
    def __init__(self):
        self.preprocessed_docs = []
        self.normalizer = hazm.Normalizer()
        self.word_tokenizer = hazm.WordTokenizer()
        self.stemmer = hazm.Stemmer()
        self.stop_words = hazm.stopwords_list()
        self.persian_garbage = {u'÷': u'',
                                u'ٰ': u'',
                                u'،': ' ',
                                u'؟': ' ',
                                u'؛': '',
                                u'َ': '',
                                u'ُ': '',
                                u'ِ': '',
                                u'ّ': '',
                                u'ٌ': '',
                                u'ٍ': '',
                                u'ئ': u'ی',
                                u'ي': u'ی',
                                u'ة': u'ه',
                                u'ء': u'',
                                u'ك': u'ک',
                                u'ْ': u'',
                                u'أ': u'ا',
                                u'إ': u'ا',
                                u'ؤ': u'و',
                                u'×': u'',
                                u'٪': u'',
                                u'٬': u'',
                                u'آ': u'ا',
                                u'●': u''}

    def tokenize(self, text: str):
        return self.word_tokenizer.tokenize(text)

    def remove_punctuations(self, text):
        return text.translate(str.maketrans(dict.fromkeys(list(string.punctuation))))

    def remove_persian_garbage(self, text):
        for old, new in self.persian_garbage.items():
            text = text.replace(old, new)

        return text

    def remove_nonpersian_alphabet(self, text):
        text = re.sub('[^۰-۹ آ-ی \u200c]', ' ', text)
        return re.sub('[!?@#$%^&*(),.\-_=+<>/|:;~`"\]\[_]', ' ', text)

    def remove_stopwords(self, tokenized_text: List[str]):
        return [word for word in tokenized_text if word not in self.stop_words]

    def stem(self, tokenized_text: List[str]):
        return [self.stemmer.stem(word) for word in tokenized_text]

    # def remove_ZWNJ(self, tokenized_text: List[str]):
    #     return [word.replace(u'\u200c', ' ') for word in tokenized_text]

    def normalize(self, text: str):
        """
        :param text: raw Persian text
        :return: preprocessed tokenized text
        """
        text = self.normalizer.normalize(text)
        text = self.remove_persian_garbage(text)
        text = self.remove_nonpersian_alphabet(text)

        tokenized_text = self.tokenize(text)
        tokenized_text = self.stem(tokenized_text)
        tokenized_text = self.remove_stopwords(tokenized_text)

        return tokenized_text

    def handle_query(self, text):
        preprocessed = self.normalize(text=text)
        print(preprocessed)

    def preprocess_xml_docs(self, docs):
        with tqdm(total=len(docs)) as pbar:
            for doc in docs:
                preprocessed_doc = self.normalize(doc)
                self.preprocessed_docs.append(preprocessed_doc)
                pbar.update(1)
        pbar.close()
        return self.preprocessed_docs

    def find_stopwords(self, doc=None):
        if doc is None:
            all_words = list(chain(*self.preprocessed_docs))
            return Counter(all_words)
        else:
            tokens = self.normalize(doc)
            return Counter(tokens)

