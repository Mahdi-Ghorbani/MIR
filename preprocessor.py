from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


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

    def normalize(self):
        pass

    def tokenize(self):
        pass

    def remove_punctuations(self):
        pass

    def remove_stopwords(self):
        pass

    def stem(self):
        pass

    def preprocess(self):
        pass

    def print_result(self):
        pass

    def handle_query(self):
        pass
