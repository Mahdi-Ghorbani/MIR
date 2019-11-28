import pandas as pd

from preprocessor import EnglishProcessor, PersianProcessor
from indexer import Positional, Bigram
from compressor import GamaCode, VariableByte
from query_editor import Editor
from searcher import TF_IDF

if __name__ == '__main__':
    english_df = pd.read_csv('data/English.csv')
    english_preprocessor = EnglishProcessor()
    english_preprocessor.preprocess(english_df['Text'])

    while True:
        print("Please enter the command number:")
        print("0. Exit")
        print("1. Print the English frequent words from csv file")
        print("2. Input a text and get the words after preprocessing")
        cmd = int(input())
        if cmd == 0:
            break
        elif cmd == 1:
            english_preprocessor.print_result()
        elif cmd == 2:
            print("Enter the text")
            cmd = input()
            english_preprocessor.handle_query(cmd)
