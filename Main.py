import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from classify import Doc2VecModel, NaiveBayesClassifier, find_metrics, knn, svm_clf, random_forest_clf
from preprocessor import EnglishProcessor, PersianProcessor
from indexer import Positional, Bigram
from compressor import GammaCode, VariableByte
from spell_check import SpellChecker
from searcher import TF_IDF
from utils import from_xml, read_corpus, search_by_subject
from pprint import pprint

DEBUG = True

if __name__ == '__main__':
    english_df = pd.read_csv('data/English.csv')
    persian_docs = from_xml('data/Persian.xml')
    english_preprocessor = EnglishProcessor()
    persian_preprocessor = PersianProcessor()

    # vb_compressor = VariableByte(positional_index=positional_index.index)
    # vb_compressor.compress()

    # gm_compressor = GammaCode(positional_index=positional_index.index)
    # gm_compressor.compress()

    train_path = 'data/phase2_train.csv'
    test_path = 'data/phase2_test.csv'
    phase1_path = 'data/English.csv'

    X_train, y_train = read_corpus(train_path)
    X_test, y_test = read_corpus(test_path)
    X_phase1, _ = read_corpus(phase1_path, False)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    while True:
        print("Please enter the command number:")
        print("0. Exit")
        print("1. Print the English frequent words from csv file")
        print("2. Input a text and get the tokens after preprocessing")
        print("3. Print the English positional index")
        print("4. Print posting list of a word in positional index")
        print("5. Remove a document by document ID from positional index")
        print("6. Save the positional index in a file")
        print("7. Load a positional index from a file")
        print("8. Print the variable byte encoded format")
        print("9. Save the variable byte encoded format in a file")
        print("10. Load a variable byte encoded format from a file")
        print("11. Compare between used space before and after variable byte compression")
        print("12. Print the gamma encoded format")
        print("13. Save the gamma encoded format in a file")
        print("14. Load a gamma encoded format from a file")
        print("15. Compare between used space before and after gamma compression")
        print("16. Normalize an input Persian text")
        print("17. print preprocessed persian xml texts")
        print("18. Find frequent words in persian xml docs")
        print("19. Find frequent words in an input text")
        print("20. Print the English frequent words from input text")
        print("21. Print the English bigram index")
        print("22. Print the Persian positional index")
        print("23. Print the Persian bigram index")
        print("24. English spellchecker")
        print("25. Search in the vector space")
        print("26. Get naive-bayes metrics")
        print("27. Get k-NN metrics")
        print("28. Get SVM metrics")
        print("29. Get random forest metrics")
        print("30. Search by subject number")
        print("31. Find best k for k-NN")
        print("32. Find best C for svm")

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

        elif cmd == 4:
            print("Enter the word")
            cmd = input()

            DEBUG = False

            if DEBUG:
                english_df = english_df[:20]

            positional_index = Positional(preprocessor=english_preprocessor)
            positional_index.add_docs(english_df['Text'])
            positional_index.find_posting(cmd)

        elif cmd == 5:
            print("Enter the document ID")
            cmd = int(input())
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            positional_index.delete_doc(cmd)

        elif cmd == 6:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            positional_index.save_to_file(cmd)

        elif cmd == 7:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            positional_index.load_from_file(cmd)

        elif cmd == 8:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:50]

            positional_index.add_docs(english_df['Text'])
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.compress()
            vb_compressor.print_result()

        elif cmd == 9:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.compress()
            vb_compressor.save_to_file(cmd)

        elif cmd == 10:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.load_from_file(cmd)

        elif cmd == 11:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            vb_compressor = VariableByte(positional_index=positional_index.index)
            vb_compressor.compress()
            vb_compressor.print_used_space()

        elif cmd == 12:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                english_df = english_df[:20]

            positional_index.add_docs(english_df['Text'])
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.print_result()

        elif cmd == 13:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.save_to_file(cmd)

        elif cmd == 14:
            print("Enter the name of the file")
            cmd = input()
            positional_index = Positional(preprocessor=english_preprocessor)
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.load_from_file(cmd)

        elif cmd == 15:
            positional_index = Positional(preprocessor=english_preprocessor)
            gm_compressor = GammaCode(positional_index=positional_index.index)
            gm_compressor.compress()
            gm_compressor.print_used_space()

        elif cmd == 16:
            text = input()
            print(persian_preprocessor.normalize(text))

        elif cmd == 17:
            preprocessed_docs = persian_preprocessor.preprocess_xml_docs(persian_docs)
            print(preprocessed_docs[0])

        elif cmd == 18:
            _ = persian_preprocessor.preprocess_xml_docs(persian_docs)
            words = persian_preprocessor.find_stopwords()
            words = [(word, freq) for word, freq in words.items()]
            words = sorted(words, key=lambda x: x[1], reverse=True)
            print(words[:30])

        elif cmd == 19:
            text = input()
            persian_preprocessor.find_stopwords(text)

        elif cmd == 20:
            print('Input a text:')
            text = input()

            words = english_preprocessor.find_stopwords(text)
            words = [(word, freq) for word, freq in words.items()]
            words = sorted(words, key=lambda x: x[1], reverse=True)
            print(words[:20])

        elif cmd == 21:
            positional_index = Positional(preprocessor=english_preprocessor)

            if DEBUG:
                persian_docs = persian_docs[:20]

            positional_index.add_docs(persian_docs)
            bigram = Bigram(preprocessor=english_preprocessor, positional_index=positional_index)
            bigram.add_docs(persian_docs)
            bigram.print_result()

        elif cmd == 22:
            positional_index = Positional(preprocessor=persian_preprocessor)
            if DEBUG:
                persian_docs = persian_docs[:20]

            positional_index.add_docs(persian_docs)
            positional_index.print_result()

        elif cmd == 23:
            positional_index = Positional(preprocessor=persian_preprocessor)

            if DEBUG:
                persian_docs = persian_docs[:20]

            positional_index.add_docs(persian_docs)
            bigram = Bigram(preprocessor=persian_preprocessor, positional_index=positional_index)
            bigram.add_docs(persian_docs)
            bigram.print_result()

        elif cmd == 24:
            positional_index = Positional(preprocessor=english_preprocessor)

            DEBUG = False

            if DEBUG:
                english_df = english_df[:20]

            print('Input a query...')
            query = input()

            positional_index.add_docs(english_df['Text'])
            bigram = Bigram(preprocessor=english_preprocessor, positional_index=positional_index)
            bigram.add_docs(english_df['Text'])
            spell_checker = SpellChecker(english_preprocessor, bigram_index=bigram)
            print(spell_checker.correct_query(query))

        elif cmd == 25:
            positional_index = Positional(preprocessor=english_preprocessor)

            DEBUG = True

            if DEBUG:
                english_df = english_df[:20]

            print('Input a query...')
            query = input()
            positional_index.add_docs(english_df['Text'])
            searcher = TF_IDF(positional_index, english_preprocessor)
            print(searcher.search(query))
        elif cmd == 26:
            print("Choose the vector model:")
            print("1. normal")
            print("2. tf-idf")
            cmd = int(input())
            nb_clf = NaiveBayesClassifier(4)
            if cmd == 1:
                nb_clf.train(X_train, y_train, use_tf=False)
            else:
                nb_clf.train(X_train, y_train, use_tf=True)
            nb_predict = []
            for x_test in X_test:
                porb, pred = nb_clf.infer(x_test)
                nb_predict.append(pred)

            nb_precision = precision_score(y_test, nb_predict, average='micro')
            nb_recall = recall_score(y_test, nb_predict, average='micro')
            nb_f1 = f1_score(y_test, nb_predict, average='micro')
            nb_f1_manually = (2 * nb_precision * nb_recall) / (nb_precision + nb_recall)
            nb_accuracy = accuracy_score(y_test, nb_predict)
            print('nb result: ', nb_predict)
            print('nb precision: ', nb_precision)
            print('nb recall: ', nb_recall)
            print('nb F1 score: ', nb_f1)
            print('nb F1 manually: ', nb_f1_manually)
            print('nb accuracy: ', nb_accuracy)
            print('metrics:')
            print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
            print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
            print('nb metrics: ', find_metrics(y_test, nb_predict))

        elif cmd == 27:
            print("Choose the vector model:")
            print("1. doc2vec")
            print("2. tf-idf")
            cmd = int(input())
            if cmd == 1:
                doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
                doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
                print('training the model...')
                doc2vecmodel.train(doc2vec_train_corpus)
                doc2vecmodel.save('model.bin')
                docvecs = doc2vecmodel.get_docvecs()

                inferred_X_test = []
                for doc in X_test:
                    inferred_X_test.append(doc2vecmodel.infer(doc))

            else:
                english_preprocessor = EnglishProcessor()
                positional_index = Positional(preprocessor=english_preprocessor)
                positional_index.add_docs(train['Text'])
                tf_idf = TF_IDF(positional_index, english_preprocessor)
                docvecs = np.transpose(tf_idf.tf_idf_matrix)

                inferred_X_test = []
                for doc in test['Text']:
                    _, vec = tf_idf.search(doc, True)
                    inferred_X_test.append(vec.T)

            k = int(input("Enter the k value"))
            knn_predict = knn(docvecs, inferred_X_test, y_train, k)
            knn_precision = precision_score(y_test, knn_predict, average='micro')
            knn_recall = recall_score(y_test, knn_predict, average='micro')
            knn_f1 = f1_score(y_test, knn_predict, average='micro')
            knn_f1_manually = (2 * knn_precision * knn_recall) / (knn_precision + knn_recall)
            knn_accuracy = accuracy_score(y_test, knn_predict)
            print('knn result: ', knn_predict)
            print('knn precision: ', knn_precision)
            print('knn recall: ', knn_recall)
            print('knn F1 score: ', knn_f1)
            print('knn F1 manually: ', knn_f1_manually)
            print('knn accuracy: ', knn_accuracy)
            print('metrics:')
            print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
            print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
            print('knn metrics: ', find_metrics(y_test, knn_predict))

        elif cmd == 28:
            print("Choose the vector model:")
            print("1. doc2vec")
            print("2. tf-idf")
            cmd = int(input())
            if cmd == 1:
                doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
                doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
                print('training the model...')
                doc2vecmodel.train(doc2vec_train_corpus)
                doc2vecmodel.save('model.bin')
                docvecs = doc2vecmodel.get_docvecs()

                inferred_X_test = []
                for doc in X_test:
                    inferred_X_test.append(doc2vecmodel.infer(doc))

            else:
                english_preprocessor = EnglishProcessor()
                positional_index = Positional(preprocessor=english_preprocessor)
                positional_index.add_docs(train['Text'])
                tf_idf = TF_IDF(positional_index, english_preprocessor)
                docvecs = np.transpose(tf_idf.tf_idf_matrix)

                inferred_X_test = []
                for doc in test['Text']:
                    _, vec = tf_idf.search(doc, True)
                    inferred_X_test.append(vec.T)

            C = float(input("Enter the C value"))
            svmClf = svm_clf(docvecs, y_train, C)
            svm_predict = svmClf.predict(inferred_X_test)
            svm_precision = precision_score(y_test, svm_predict, average='micro')
            svm_recall = recall_score(y_test, svm_predict, average='micro')
            svm_f1 = f1_score(y_test, svm_predict, average='micro')
            svm_f1_manually = (2 * svm_precision * svm_recall) / (svm_precision + svm_recall)
            svm_accuracy = accuracy_score(y_test, svm_predict)
            print('svm result: ', svm_predict)
            print('svm precision: ', svm_precision)
            print('svm recall: ', svm_recall)
            print('svm F1 score: ', svm_f1)
            print('svm F1 manually: ', svm_f1_manually)
            print('svm accuracy: ', svm_accuracy)
            print('metrics:')
            print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
            print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
            print('svm metrics: ', find_metrics(y_test, svm_predict))

        elif cmd == 29:
            print("Choose the vector model:")
            print("1. doc2vec")
            print("2. tf-idf")
            cmd = int(input())
            if cmd == 1:
                doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
                doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
                print('training the model...')
                doc2vecmodel.train(doc2vec_train_corpus)
                doc2vecmodel.save('model.bin')
                docvecs = doc2vecmodel.get_docvecs()

                inferred_X_test = []
                for doc in X_test:
                    inferred_X_test.append(doc2vecmodel.infer(doc))

            else:
                english_preprocessor = EnglishProcessor()
                positional_index = Positional(preprocessor=english_preprocessor)
                positional_index.add_docs(train['Text'])
                tf_idf = TF_IDF(positional_index, english_preprocessor)
                docvecs = np.transpose(tf_idf.tf_idf_matrix)

                inferred_X_test = []
                for doc in test['Text']:
                    _, vec = tf_idf.search(doc, True)
                    inferred_X_test.append(vec.T)

            rfClf = random_forest_clf(docvecs, y_train)
            rf_predict = rfClf.predict(inferred_X_test)
            rf_precision = precision_score(y_test, rf_predict, average='micro')
            rf_recall = recall_score(y_test, rf_predict, average='micro')
            rf_f1 = f1_score(y_test, rf_predict, average='micro')
            rf_f1_manually = (2 * rf_precision * rf_recall) / (rf_precision + rf_recall)
            rf_accuracy = accuracy_score(y_test, rf_predict)
            print('rf result: ', rf_predict)
            print('rf precision: ', rf_precision)
            print('rf recall: ', rf_recall)
            print('rf F1 score: ', rf_f1)
            print('rf F1 manually: ', rf_f1_manually)
            print('rf accuracy: ', rf_accuracy)
            print('metrics:')
            print('metrics[0] == precision, metrics[1] == recall, metrics[2] == f1, metrics[3] == accuracy')
            print('metrics[:][0] == for all classes, metrics[:][i] == for class number i')
            print('rf metrics: ', find_metrics(y_test, rf_predict))

        elif cmd == 30:
            print("Choose the vector model:")
            print("1. doc2vec")
            print("2. tf-idf")
            cmd = int(input())
            if cmd == 1:
                doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]
                doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
                print('training the model...')
                doc2vecmodel.train(doc2vec_train_corpus)
                doc2vecmodel.save('model.bin')
                docvecs = doc2vecmodel.get_docvecs()

                inferred_X_test = []
                for doc in X_phase1:
                    inferred_X_test.append(doc2vecmodel.infer(doc))

            else:
                english_preprocessor = EnglishProcessor()
                positional_index = Positional(preprocessor=english_preprocessor)
                positional_index.add_docs(train['Text'])
                tf_idf = TF_IDF(positional_index, english_preprocessor)
                docvecs = np.transpose(tf_idf.tf_idf_matrix)

                inferred_X_test = []
                for doc in X_phase1:
                    _, vec = tf_idf.search(doc, True)
                    inferred_X_test.append(vec.T)

            print("Enter the classifier:")
            print("1. Naive Bayes")
            print("2. k-NN")
            print("3. SVM")
            print("4. RF")
            cmd = int(input())

            if cmd == 1:
                nb_clf = NaiveBayesClassifier(4)
                print("Choose the vector model:")
                print("1. normal")
                print("2. tf-idf")
                cmd = int(input())
                if cmd == 1:
                    nb_clf.train(X_train, y_train, use_tf=False)
                else:
                    nb_clf.train(X_train, y_train, use_tf=True)
                predict = []
                for x_test in X_phase1:
                    porb, pred = nb_clf.infer(x_test)
                    predict.append(pred)
                print(predict)

            elif cmd == 2:
                k = int(input("Enter the k value"))
                predict = knn(docvecs, inferred_X_test, y_train, k)
                print(predict)

            elif cmd == 3:
                C = float(input("Enter the C value"))
                svmClf = svm_clf(docvecs, y_train, C)
                predict = svmClf.predict(inferred_X_test)
                print(predict)

            else:
                rfClf = random_forest_clf(docvecs, y_train)
                predict = rfClf.predict(inferred_X_test)
                print(predict)

            tag = int(input("Enter the tag"))
            print(search_by_subject(predict, tag))

        elif cmd == 31:
            print("Choose the vector model:")
            print("1. doc2vec")
            print("2. tf-idf")
            cmd = int(input())
            new_X_train = X_train[0: 8100]
            new_y_train = y_train[0: 8100]
            new_X_test = X_train[8100:]
            new_y_test = y_train[8100:]
            if cmd == 1:
                doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(new_X_train)]
                doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
                print('training the model...')
                doc2vecmodel.train(doc2vec_train_corpus)
                doc2vecmodel.save('model.bin')
                docvecs = doc2vecmodel.get_docvecs()

                inferred_X_test = []
                for doc in new_X_test:
                    inferred_X_test.append(doc2vecmodel.infer(doc))

            else:
                new_train = train['Text'][0:8100]
                new_test = train['Text'][8100:]
                english_preprocessor = EnglishProcessor()
                positional_index = Positional(preprocessor=english_preprocessor)
                positional_index.add_docs(new_train)
                tf_idf = TF_IDF(positional_index, english_preprocessor)
                docvecs = np.transpose(tf_idf.tf_idf_matrix)

                inferred_X_test = []
                for doc in new_test:
                    _, vec = tf_idf.search(doc, True)
                    inferred_X_test.append(vec.T)

            knn1 = knn(docvecs, inferred_X_test, new_y_train, 1)
            knn5 = knn(docvecs, inferred_X_test, new_y_train, 5)
            knn9 = knn(docvecs, inferred_X_test, new_y_train, 9)

            acc1 = accuracy_score(new_y_test, knn1)
            acc5 = accuracy_score(new_y_test, knn5)
            acc9 = accuracy_score(new_y_test, knn9)

            if acc1 >= acc5 and acc1 >= acc9:
                print('best k is 1')
            elif acc5 >= acc1 and acc5 >= acc9:
                print('best k is 5')
            else:
                print('best k is 9')

        elif cmd == 32:
            print("Choose the vector model:")
            print("1. doc2vec")
            print("2. tf-idf")
            cmd = int(input())
            new_X_train = X_train[0: 8100]
            new_y_train = y_train[0: 8100]
            new_X_test = X_train[8100:]
            new_y_test = y_train[8100:]
            if cmd == 1:
                doc2vec_train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(new_X_train)]
                doc2vecmodel = Doc2VecModel(vector_size=50, min_count=2, epochs=40)
                print('training the model...')
                doc2vecmodel.train(doc2vec_train_corpus)
                doc2vecmodel.save('model.bin')
                docvecs = doc2vecmodel.get_docvecs()

                inferred_X_test = []
                for doc in new_X_test:
                    inferred_X_test.append(doc2vecmodel.infer(doc))

            else:
                new_train = train['Text'][0:8100]
                new_test = train['Text'][8100:]
                english_preprocessor = EnglishProcessor()
                positional_index = Positional(preprocessor=english_preprocessor)
                positional_index.add_docs(new_train)
                tf_idf = TF_IDF(positional_index, english_preprocessor)
                docvecs = np.transpose(tf_idf.tf_idf_matrix)

                inferred_X_test = []
                for doc in new_test:
                    _, vec = tf_idf.search(doc, True)
                    inferred_X_test.append(vec.T)

            svmClf = svm_clf(docvecs, new_y_train, 0.5)
            svm0_5 = svmClf.predict(inferred_X_test)
            svmClf = svm_clf(docvecs, new_y_train, 1.0)
            svm1 = svmClf.predict(inferred_X_test)
            svmClf = svm_clf(docvecs, new_y_train, 1.5)
            svm1_5 = svmClf.predict(inferred_X_test)
            svmClf = svm_clf(docvecs, new_y_train, 2.0)
            svm2 = svmClf.predict(inferred_X_test)

            acc0_5 = accuracy_score(new_y_test, svm0_5)
            acc1 = accuracy_score(new_y_test, svm1)
            acc1_5 = accuracy_score(new_y_test, svm1_5)
            acc2 = accuracy_score(new_y_test, svm2)

            print(acc0_5)
            print(acc1)
            print(acc1_5)
            print(acc2)
            if acc0_5 >= acc1 and acc0_5 >= acc1_5 and acc0_5 >= acc2:
                print('best C is 0.5')
            elif acc1 >= acc0_5 and acc1 >= acc1_5 and acc1 >= acc2:
                print('best C is 1.0')
            elif acc1_5 >= acc0_5 and acc1_5 >= acc1 and acc1_5 >= acc2:
                print('best C is 1.5')
            else:
                print('best C is 2.0')
