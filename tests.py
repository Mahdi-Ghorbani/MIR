from preprocessor import EnglishProcessor, PersianProcessor
import xml.dom.minidom
import xml.etree.ElementTree as ET
import hazm
import re
from pprint import pprint
from utils import from_xml

DEBUG = True


def test_persian_preprocessor_from_xml(xml_file):
    processor = PersianProcessor()

    with open(xml_file) as f:
        xml_string = f.read()

    texts = from_xml(xml_string)

    if DEBUG:
        texts = texts[:20]

    preprocessed_texts = [processor.preprocess(text) for text in texts]

    pprint(processor.find_stopwords(preprocessed_texts[0]))


def test_persian_preprocessor_from_input_text():
    """
    inputs a text in Persian from the user print the preprocessed version of it.
    """
    pass


def test_indexing():
    pass



#test_persian_preprocessor('data/Persian.xml')
