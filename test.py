import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

#Reads from a txt document with some sample writing in it not from 20x students
'''
with open("sample_writing.txt", "r") as myfile:
    contents_raw = myfile.read()
'''

#Reads in files from the Reflectivetest1.txt document and puts it into var content_raw
with open("Reflectivetest1.txt", "r") as myfile:
    contents_raw = myfile.read()

contents = str(contents_raw)

tokenized_sent = sent_tokenize(contents)

#print(tokenized_sent)

tokenized_word = word_tokenize(contents)

#print(tokenized_word)

from nltk.probability import FreqDist

'''
fdist = FreqDist(tokenized_word)

print(fdist)

print(fdist.most_common(5))

import matplotlib.pyplot as plt

fdist.plot(30, cumulative=False)

plt.show
'''

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
punctuation = set([".", ",", "''", "''"])

stop_words.update(punctuation)

filtered_contents = []

for w in tokenized_word:
    if w not in stop_words:
        filtered_contents.append(w)

fdist_stop = FreqDist(filtered_contents)

print(fdist_stop)

print(fdist_stop.most_common(10))

import matplotlib.pyplot as plt

fdist_stop.plot(50, cumulative=False)

plt.show




#print("Tokenized contents: ", tokenized_sent)

#print("Filtered contents: ", filtered_contents)

#print("Stop words: ", stop_words)
