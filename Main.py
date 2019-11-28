import pandas as pd

from preprocessor import EnglishProcessor, PersianProcessor
from indexer import Positional, Bigram
from compressor import GammaCode, VariableByte
from query_editor import Editor
from searcher import TF_IDF

if __name__ == '__main__':
    english_df = pd.read_csv('data/English.csv')
    english_preprocessor = EnglishProcessor()
    # TODO: Should we take care about the Titles (Not only Text)?
    english_preprocessor.preprocess(english_df['Text'])

    positional_index = Positional(preprocessor=english_preprocessor)
    positional_index.add_df(english_df['Text'])

    vb_compressor = VariableByte(positional_index=positional_index.index)
    vb_compressor.compress()

    while True:
        print("Please enter the command number:")
        print("0. Exit")
        print("1. Print the English frequent words from csv file")
        print("2. Input a text and get the words after preprocessing")
        print("3. Print the English positional index")
        print("4. Print posting list of a word in positional index")
        print("5. Remove terms of a arbitrary text from positional index")
        print("6. Remove a document by document ID from positional index")
        print("7. Save the positional index in a file")
        print("8. Load a positional index from a file")
        print("9. Print the variable byte encoded format")
        print("10. Save the variable byte encoded format in a file")
        print("11. Load a variable byte encoded format from a file")
        print("12. Compare between used space before and after variable byte compression")
        cmd = int(input())
        if cmd == 0:
            break
        elif cmd == 1:
            english_preprocessor.print_result()
        elif cmd == 2:
            print("Enter the text")
            cmd = input()
            english_preprocessor.handle_query(cmd)
        elif cmd == 3:
            positional_index.print_result()
        elif cmd == 4:
            print("Enter the word")
            cmd = input()
            positional_index.find_posting(cmd)
        elif cmd == 5:
            print("Enter the text")
            cmd = input()
            positional_index.delete_text(cmd)
        elif cmd == 6:
            print("Enter the document ID")
            cmd = int(input())
            positional_index.delete_doc(cmd)
        elif cmd == 7:
            print("Enter the name of the file")
            cmd = input()
            positional_index.save_to_file(cmd)
        elif cmd == 8:
            print("Enter the name of the file")
            cmd = input()
            positional_index.load_from_file(cmd)
        elif cmd == 9:
            vb_compressor.print_result()
        elif cmd == 10:
            print("Enter the name of the file")
            cmd = input()
            vb_compressor.save_to_file(cmd)
        elif cmd == 11:
            print("Enter the name of the file")
            cmd = input()
            vb_compressor.load_from_file(cmd)
        elif cmd == 12:
            vb_compressor.print_used_space()
