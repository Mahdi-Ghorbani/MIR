from preprocessor import EnglishProcessor, PersianProcessor
import xml.dom.minidom
import xml.etree.ElementTree as ET
import hazm
import re
from pprint import pprint

DEBUG = True


def test_persian_preprocessor():
    processor = PersianProcessor()

    with open('data/Persian.xml') as f:
        content = f.read()

    texts = processor.from_xml(content)

    if DEBUG:
        texts = texts[:20]

    texts = [processor.remove_persian_garbage(text) for text in texts]
    texts = [processor.remove_nonpersian_alphabet(text) for text in texts]

    tokenized_texts = [processor.tokenize(text) for text in texts]
    tokenized_texts = [processor.remove_stopwords(tokenized_text) for tokenized_text in tokenized_texts]
    #tokenized_texts = [processor.remove_ZWNJ(tokenized_text) for tokenized_text in tokenized_texts]

    tokenized_texts = [processor.stem(tokenized_text) for tokenized_text in tokenized_texts]
    pprint(processor.find_stopwords(tokenized_texts[0]))


test_persian_preprocessor()

stemmer = hazm.Stemmer()