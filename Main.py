import pandas as pd

from preprocessor import EnglishProcessor, PersianProcessor
from indexer import Positional, Bigram
from compressor import GammaCode, VariableByte
from spell_check import SpellChecker
from searcher import TF_IDF
from utils import from_xml
from pprint import pprint

DEBUG = True

if __name__ == '__main__':
    english_df = pd.read_csv('data/English.csv')
    persian_docs = from_xml('data/Persian.xml')
    english_preprocessor = EnglishProcessor()
    persian_preprocessor = PersianProcessor()

    #vb_compressor = VariableByte(positional_index=positional_index.index)
    #vb_compressor.compress()

    #gm_compressor = GammaCode(positional_index=positional_index.index)
    #gm_compressor.compress()

    while True:
        print("Please enter the command number:")
        print("0. Exit")
        print("1. Print the English frequent words from csv file")
        print("2. Input a text and get the tokens after preprocessing")
        print("3. Print the English positional index")
        print("4. Print posting list of a word in positional index")
        print("5. ")
        print("6. Remove a document by document ID from positional index")
        print("7. Save the positional index in a file")
        print("8. Load a positional index from a file")
        print("9. Print the variable byte encoded format")
        print("10. Save the variable byte encoded format in a file")
        print("11. Load a variable byte encoded format from a file")
        print("12. Compare between used space before and after variable byte compression")
        print("13. Print the gamma encoded format")
        print("14. Save the gamma encoded format in a file")
        print("15. Load a gamma encoded format from a file")
        print("16. Compare between used space before and after gamma compression")
        print("17. Normalize an input Persian text")
        print("18. print preprocessed persian xml texts")
        print("19. Find frequent words in persian xml docs")
        print("20. Find frequent words in an input text")
        print("21. Print the English frequent words from input text")
        print("22. Print the English bigram index")
        print("23. Print the Persian positional index")
        print("24. Print the Persian bigram index")
        print("25. English spellchecker")
        print("26. Search in the vector space")

        cmd = int(input())
        if cmd == 0:
            break
        elif cmd == 1:
            _ = english_preprocessor.preprocess(english_df)
            words = english_preprocessor.find_stopwords()
            words = [(word, freq) for word, freq in words.items()]
            words = sorted(words, key=lambda x: x[1], reverse=True)
            print(words[:20])
        elif cmd == 2:
            print("Enter the text")
            cmd = input()
            print(english_preprocessor.handle_query(cmd))
        elif cmd == 3:
            if DEBUG:
                english_df = english_df[:20]
            positional_index = Positional(preprocessor=english_preprocessor)
            positional_index.add_docs(english_df['Text'])
            positional_index.print_result()
            break
        elif cmd == 4:
            print("Enter the word")
            cmd = input()

            DEBUG = False

            if DEBUG:
                english_df = english_df[:20]

            positional_index = Positional(preprocessor=english_preprocessor)
            positional_index.add_docs(english_df['Text'])
            positional_index.find_posting(cmd)
            break
        elif cmd == 5:
            pass
            # print("Enter the text")
            # cmd = input()
            # positional_index = Positional(preprocessor=english_preprocessor)
            # positional_index.delete_text(cmd)
        elif cmd == 6:
            print("Enter the document ID")
            cmd = int(input())
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            positional_index.delete_doc(cmd)
        elif cmd == 7:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            positional_index.save_to_file(cmd)
        elif cmd == 8:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            positional_index.load_from_file(cmd)
        elif cmd == 9:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:50]

            positional_index.add_docs(english_df['Text'])
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.compress()
            vb_compressor.print_result()
        elif cmd == 10:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.compress()
            vb_compressor.save_to_file(cmd)
        elif cmd == 11:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.load_from_file(cmd)
        elif cmd == 12:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.compress()
            vb_compressor.print_used_space()
        elif cmd == 13:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.print_result()
        elif cmd == 14:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.save_to_file(cmd)
        elif cmd == 15:
            print("Enter the name of the file")
            cmd = input()
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.load_from_file(cmd)
        elif cmd == 16:
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.print_used_space()
        elif cmd == 17:
            text = input()
            print(persian_preprocessor.normalize(text))
        elif cmd == 18:
            preprocessed_docs = persian_preprocessor.preprocess_xml_docs(persian_docs)
            print(preprocessed_docs[0])
            break
        elif cmd == 19:
            _ = persian_preprocessor.preprocess_xml_docs(persian_docs)
            words = persian_preprocessor.find_stopwords()
            words = [(word, freq) for word, freq in words.items()]
            words = sorted(words, key= lambda x: x[1], reverse=True)
            print(words[:30])
            break
        elif cmd == 20:
            text = input()
            persian_preprocessor.find_stopwords(text)
        elif cmd == 21:
            print('Input a text:')
            text = input()
            words = english_preprocessor.find_stopwords(text)
            words = [(word, freq) for word, freq in words.items()]
            words = sorted(words, key=lambda x: x[1], reverse=True)
            print(words[:20])
        elif cmd == 22:
            positional_index = Positional(preprocessor=english_preprocessor)
            if DEBUG:
                persian_docs = persian_docs[:20]
            positional_index.add_docs(persian_docs)
            bigram = Bigram(preprocessor=english_preprocessor, positional_index=positional_index)
            bigram.add_docs(persian_docs)
            bigram.print_result()
            break
        elif cmd == 23:
            positional_index = Positional(preprocessor=persian_preprocessor)
            if DEBUG:
                persian_docs = persian_docs[:20]

            positional_index.add_docs(persian_docs)
            positional_index.print_result()
        elif cmd == 24:
            positional_index = Positional(preprocessor=persian_preprocessor)

            if DEBUG:
                persian_docs = persian_docs[:20]

            positional_index.add_docs(persian_docs)
            bigram = Bigram(preprocessor=persian_preprocessor, positional_index=positional_index)
            bigram.add_docs(persian_docs)
            bigram.print_result()

        elif cmd == 25:
            positional_index = Positional(preprocessor=english_preprocessor)

            DEBUG = False

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            bigram = Bigram(preprocessor=english_preprocessor, positional_index=positional_index)
            bigram.add_docs(english_df['Text'])
            spell_checker = SpellChecker(english_preprocessor, bigram_index=bigram)
            print(spell_checker.correct_query('Hay Bey!'))
            break

        elif cmd == 26:
            positional_index = Positional(preprocessor=english_preprocessor)

            DEBUG = True

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            searcher = TF_IDF(positional_index, english_preprocessor)
            query = "PARMALAT, the bankrupt Italian food company, is suing outside auditors Grant Thornton and Deloitte  amp; Touche, seeking 5.5 billion in damages."
            print(searcher.search(query))
            break




