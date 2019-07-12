'''NLTK Test for use with 20x students' reflective writings.'''
'''***********************************************************'''
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


with open("Reflectivetest1.txt", "r") as myfile: #Reads in files from the Reflectivetest1.txt document and puts it into var content_raw
    contents_raw = myfile.read()

contents = str(contents_raw) #Turns the contents into a string
#tokenized_sent = sent_tokenize(contents) #Tokenizes by sentence
#print(tokenized_sent)
tokenized_word = word_tokenize(contents) #Tokenizes by word
#print(tokenized_word)

'''Filtering of Stop words and punctuation'''
'''********************************************************'''
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) #Makes a set with the stopwords in the english dictionary
punctuation = set([".", ",", "''", "''", "``", "(", ")", "The"]) #Makes a set with some punctuation that wasn't being filtered out

stop_words.update(punctuation) #Adds to the set some punctuation that wasn't being filtered out

filtered_contents = [] #Creates an empty list

for w in tokenized_word: #Looks at every word in the text,
    if w not in stop_words:
        filtered_contents.append(w)

'''Text Stemming - Not yet functional'''
'''***********************************************************'''

'''
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words = []
for w in filtered_contents:
    stemmed_words.append(ps.stem(w))
'''

'''Finding the Most Common Words'''
'''*********************************************************'''
from nltk.probability import FreqDist, DictionaryProbDist
import numpy as np

fdist_stop = FreqDist(filtered_contents) #Creates a list of words with the number of occurences
#print(fdist_stop) #Prints the number of unique words and total number of words
#print(fdist_stop.most_common(10)) #Prints the 10 most common words
#print(list(fdist_stop.keys()))

all_words = [] #Create a list for every word
all_prob = [] #Create a list for every probability

for word in fdist_stop.keys(): #Loops through each word in the freq dist, adds to lists
    all_words.append(word)
    all_prob.append(float(fdist_stop.freq(word)))

#Create a table from the word list and probability list and sorts it in descending order
table = np.column_stack((all_words, all_prob)) #Creates an array from the word and prob list
table_sorted = sorted(table,key=lambda x: x[1], reverse=True) #^Pulls the second column of the array, sorts by descending order of probability

#Pulls however many of the top words from the list
num_words = 15 #Sets how many words you want to plot
some_words = []
some_probs = []

#The frequency is a string, so in this loop we turn it into a number and round it to the 4th decimal
for i in range(num_words): #Populates the nth most common words/probs into lists
    some_words.append(table_sorted[i][0])
    some_probs.append(round(float(table_sorted[i][1]), 4))

#Now some_words and some_probs are lists of the most common words and their corresponponding frequencies

'''BAR PLOT - Frequency of Words'''
'''**********************************************'''
#Creates a bar plot of the frequency of words occuring in the writings
import matplotlib.pyplot as plt

x_pos = np.arange(len(some_words)) #Sets position of the words

axes = plt.bar(x_pos, some_probs, align="center") #Makes the axes, populates plot
plt.xticks(x_pos, some_words, rotation=45, ha="right") #Sets ticks on x axis, rotates labels

plt.ylabel("Frequency")
plt.title("10 most commonly used words in \n students' reflective writings")

plt.tight_layout() #Tight layout doesn't cut off words
plt.show()
