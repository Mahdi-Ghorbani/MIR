from preprocessor import EnglishProcessor, PersianProcessor
import xml.dom.minidom
import xml.etree.ElementTree as ET
import hazm
from utils import PERSIAN_GARBAGE

DEBUG = False


def test_persian_preprocessor():
    processor = PersianProcessor()

    with open('data/Persian.xml') as f:
        content = f.read()

    texts = processor.from_xml(content)

    if DEBUG:
        texts = texts[:20]

    texts = [processor.remove_punctuations(text) for text in texts]
    texts = [processor.remove_persian_garbage(text) for text in texts]

    tokenized_texts = [processor.tokenize(text) for text in texts]
    tokenized_texts = [processor.remove_stopwords(tokenized_text) for tokenized_text in tokenized_texts]
    stemmed_texts = [[processor.stem(token) for token in tokenized_text] for tokenized_text in tokenized_texts]

    print(stemmed_texts[0])

    assert len(stemmed_texts) == 1572


test_persian_preprocessor()