import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

with open("sample_writing.txt", "r") as myfile:
    contents_raw = myfile.read()

contents = str(contents_raw)

tokenized_sent = sent_tokenize(contents)

#print(tokenized_sent)

tokenized_word = word_tokenize(contents)

#print(tokenized_word)

from nltk.probability import FreqDist

fdist = FreqDist(tokenized_word)

print(fdist)

print(fdist.most_common(5))

import matplotlib.pyplot as plt

fdist.plot(30, cumulative=False)

plt.show

'''
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

filtered_contents = []

for w in tokenized_sent:
    if w not in stop_words:
        filtered_contents.append(w)

print("Tokenized contents: ", tokenized_sent)

print("Filtered contents: ", filtered_contents)
'''
