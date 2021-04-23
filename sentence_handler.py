import nltk


class SentenceHandler:
# Seperate source text into individual sentences
    def __init__(self):
        self.sentence_tokenizer = nltk.tokenize.sent_tokenize

    def tokenize(self, text):
        return self.sentence_tokenizer(text)
