from utils import jaccard_similarity, edit_distance, get_bigrams


class SpellChecker:
    def suggest_word(self, query: str, preprocessor):
        query_tokens =  preprocessor.normalize(query)

        for token in query_tokens:
            bigrams_token = get_bigrams(query)




    def edit_query(self):
        pass

    def print_result(self):
        pass
