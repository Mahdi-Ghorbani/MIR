from preprocessor import EnglishProcessor, PersianProcessor
import xml.dom.minidom
import xml.etree.ElementTree as ET
import hazm
import re
from pprint import pprint
from utils import from_xml
from collections import Counter

DEBUG = True


def test_persian_preprocessor_from_xml(xml_file):
    processor = PersianProcessor()

    with open(xml_file) as f:
        xml_string = f.read()

    texts = from_xml(xml_string)

    if DEBUG:
        texts = texts[:20]

    preprocessed_texts = [processor.normalize(text) for text in texts]

    pprint(processor.find_stopwords(preprocessed_texts[0]))


def test_persian_preprocessor_from_input_text():
    """
    inputs a text in Persian from the user print the preprocessed version of it.
    """
    pass


def test_indexing():
    pass


def test_spell_checker():
    pass


def test_search():
    pass


persian_garbage = {u'÷': u'',
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



#test_persian_preprocessor('data/Persian.xml')
