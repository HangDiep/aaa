import nltk
nltk.download('punkt')  # tải tokenizer nếu chưa có
nltk.download('punkt_tab')
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag=np.zeros(len(all_words), dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag
sentence=["Hello, how are you?"]
words=tokenize(sentence[0])
print(words)

