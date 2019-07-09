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

'''
Filtering of Stop words and punctuation
'''
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) #Makes a set with the stopwords in the english dictionary
punctuation = set([".", ",", "''", "''"]) #Makes a set with some punctuation that wasn't being filtered out

stop_words.update(punctuation) #Adds to the set some punctuation that wasn't being filtered out

filtered_contents = [] #Creates an empty list

for w in tokenized_word: #Looks at every word in the text,
    if w not in stop_words:
        filtered_contents.append(w)
'''
Text stemming
'''
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words = []
for w in filtered_contents:
    stemmed_words.append(ps.stem(w))

'''
Plots most common words found in the texts
'''

from nltk.probability import FreqDist

fdist_stop = FreqDist(filtered_contents) #Creates a list of words with the number of occurences

print(fdist_stop) #Prints the number of unique words and total number of words

print(fdist_stop.most_common(10)) #Prints the 10 most common words

import matplotlib.pyplot as plt

fdist_stop.plot(50, cumulative=False)

plt.show




#print("Tokenized contents: ", tokenized_sent)

#print("Filtered contents: ", filtered_contents)

#print("Stop words: ", stop_words)
