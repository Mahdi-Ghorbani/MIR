from utils import jaccard_similarity, edit_distance, get_bigrams
from typing import Dict


class SpellChecker:
    def __init__(self, preprocessor, bigram_index: Dict):
        self.preprocessor = preprocessor
        self.bigram_index = bigram_index

    def suggest_word(self, token: str):

        bigrams_token = get_bigrams(token)

        possible_similar_words = set()
        for bigram in bigrams_token:
            possible_similar_words.union(self.bigram_index[bigram])

        jaccard_sims = []
        for word in possible_similar_words:
            jaccard_sims.append((word, jaccard_similarity(bigrams_token, get_bigrams(word))))

        # sorting the possibly similar words based on their jaccard distance to the main token
        jaccard_sims = sorted(jaccard_sims, key=lambda x: x[1])
        similar_words = jaccard_sims[:5]  # similar words with their jaccard distance to the main token
        distances = [(t[0], edit_distance(token, t[0])) for t in similar_words]
        distances = sorted(distances, key=lambda x: x[1])
        correct_word = distances[0][0]

        return correct_word

    def correct_query(self, query: str):
        query_tokens = self.preprocessor.preprocess(query)
        correct_query = []

        for token in query_tokens:
            correct_query.append(self.suggest_word(token))

        return correct_query

