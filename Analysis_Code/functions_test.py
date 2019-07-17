import analysis_functions as af

'''Importing Students' Writing'''
'''********************************************************'''
#Reads in files from the Reflectivetest1.txt document and puts it into var content_raw
with open("../Writings/Reflectivetest1.txt", "r") as myfile:
    contents_raw = myfile.read()

filtered_contents = af.FilterWords(contents_raw) #Filters out stop Words

'''Creating frequency distribution and plotting it'''
'''********************************************************'''
from nltk.probability import FreqDist, DictionaryProbDist
import numpy as np

fdist_stop = FreqDist(filtered_contents)
sorted_fd = af.SortFreqDist(fdist_stop) # Creates a frequency distribution and organizes by frequency descending
most_common = af.MostCommon(sorted_fd) #First column is the word, second column is frequent. Defaults to 15
af.PlotWordFrequency(most_common[:,0], most_common[:,1])
