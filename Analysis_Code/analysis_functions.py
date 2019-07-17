'''Function Definitions used in the Analysis Script'''
'''***********************************************************'''
#Filters out stop words and a custom list of words from a file
def FilterWords(text_str, filter_list=[]):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    contents = str(text_str)
    tokenized_word = word_tokenize(contents) #Tokenizes by word

    stop_words = set(stopwords.words("english")) #Makes a set with the stopwords in the english dictionary
    punctuation = set([".", ",", "''", "''", "``", "(", ")", "The"]) #Makes a set with some punctuation that wasn't being filtered out

    stop_words.update(punctuation) #Adds to the set some punctuation that wasn't being filtered out
    stop_words.update(filter_list)

    filtered_contents = [] #Creates an empty list

    for w in tokenized_word: #Looks at every word in the text,
        if w not in stop_words:
            filtered_contents.append(w)

    return(filtered_contents);

#Sorts the frequency distribution in ascending or descending order
def SortFreqDist(fdist, descending=True):
    import numpy as np
    #fdist is the frequency distribution,
    #wordnum is the number
    #ascending true for sorting from most frequent to least, false for least frequent to most

    all_words = [] #Create a list for every word
    all_prob = [] #Create a list for every probability

    for word in fdist.keys(): #Loops through each word in the freq dist, adds to lists
        all_words.append(word)
        all_prob.append(float(fdist.freq(word)))
    table = np.column_stack((all_words, all_prob)) #Creates an array from the word and prob list
    table_sorted = sorted(table,key=lambda x: x[1], reverse=descending)

    return table_sorted;

def MostCommon(sorted_array, num_words=15):
    import numpy as np

    some_words = []
    some_probs = []

    for i in range(num_words): #Populates the nth most common words/probs into lists
        some_words.append(sorted_array[i][0]) #Pulls entries from the first column of array
        some_probs.append(round(float(sorted_array[i][1]), 4)) #Pulls entries from the second column of array
    mc_table = np.column_stack((some_words, some_probs))
    return(mc_table);


#Plots the frequency of the top n words
def PlotWordFrequency(words, probs):
    import matplotlib.pyplot as plt
    import numpy as np

    word_num = len(words)
    probs_float = []

    for i in range(word_num):
        probs_float.append(round(float(probs[i]), 4))

    x_pos = np.arange(len(words))
    axes = plt.bar(x_pos, probs_float, align="center")

    plt.xticks(x_pos, words, rotation=45, ha = "right")
    plt.ylabel("Frequency of Word in Students' Writing")
    plt.title("%i most used words in \n students' reflective writings" %word_num)

    plt.tight_layout()
    plt.show()

    return;
