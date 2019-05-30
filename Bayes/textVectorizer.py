import re

class textVectorizer(object):
    def __init__(self, words=set()):
        self._words = list(words)

    def fit(self, text):
        self._words = list(set(self._words).union(set(self._splitText(text))))

    def transform(self, text):
        vec = [0] * len(self._words)
        for word in self._splitText(text):
            if word in self._words:
                vec[self._words.index(word)] += 1
        return vec

    def _splitText(self, text):
        tokens = []
        for token in re.split('\\W+', text):
            if len(token) > 2:
                tokens.append(token.lower())
        return tokens
