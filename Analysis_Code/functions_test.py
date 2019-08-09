import analysis_functions as af

'''Importing Students' Writing'''
'''********************************************************'''
from nltk.tokenize import word_tokenize

#Reads in files from the Reflectivetest1.txt document and puts it into var content_raw
with open("../Writings/Reflectivetest1.txt", "r") as myfile:
    contents_raw = myfile.read()

'''Reading in individual responses'''
'''*****************************************************'''
response_array = af.ReadInResponses("../Writings/IndvResponses.csv")
#print(response_array[0])

'''Filtering Text'''
'''******************************************************'''
#Can filter out stopwords or a custom list

#wt_content = word_tokenize(str(contents_raw)) #Turns the raw contents into word tokenized content

#af.FilterWords(wt_content)

#filtered_contents = af.FilterWords(wt_content, filter_list=filter, filter_stop_words=True) #Filters the strings in filter from the contents

'''Lemmatization'''
'''*******************************************************'''
#description

#lem_content = af.Lemmatize(wt_content)
#print(lem_content)

'''Part of Speech Tagging'''
'''********************************************************'''
#Tags each word in the content with its appropriate part of speach
#from nltk import pos_tag
#print(pos_tag(wt_content))

'''Creating frequency distribution and plotting it'''
'''********************************************************'''
from nltk.probability import FreqDist, DictionaryProbDist
import numpy as np

#fdist_stop = FreqDist(filtered_contents)
#sorted_fd = af.SortFreqDist(fdist_stop) # Creates a frequency distribution and organizes by frequency descending
#most_common = af.MostCommon(sorted_fd) #First column is the word, second column is frequent. Defaults to 15
#af.PlotWordFrequency(most_common, 3)

'''Content Stemming'''
'''******************************************************'''
#Reduces words to their stems by removing prefixes, suffixes, etc.

#stemmed_contents = af.Stem(filtered_contents)

'''Generating Ngrams'''
'''**************************************************'''
#Shows a list of ngrams for a given value of n
n=3
#ngram = list(af.Ngram(filtered_contents, n)) #Creates a list of ngrams in the filtered contents of the text

#print some of the entries in the list
#for gram in ngram[:2]:
#    print(gram)

'''Collocation for ngrams'''
'''*******************************************************'''
#Collocations of words in the text are measured and put into a list with the ngram and the frequency

#ngram_collocations = af.Collocation(filtered_contents, n) #Generates the collocations for the ngrams

#for ngram in ngram_collocations[:20]: #Prints the top 20 most frequent ngrams
    #print(ngram)

'''Plotting Collocation Tables'''
'''**********************************************************'''
#Puts the top num_grams_to_plot into a plot with associated ngram
#af.CollocationTable(ngram_collocations, num_grams_to_plot=20)

'''Sentiment Analysis'''
'''***********************************************************'''
#This analyzes the content overall to determine if it positive, negative, or neutral
#sentiment_scores = af.Sentiment(response_array[0])

#len(response_array)

af.PrintSentimentValues(response_array)
#print("Sentence:", sentiment_scores[5][])
