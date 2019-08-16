Transformative Experiences of Undergraduate Students in Algebra-Based Physics
-------------------------------------------------------------------------------------------

Description of the code
------------------------------------------------------------------------------------------
Description of functions

ReadInResponses(file): This function takes in data from the students' reflective writings and stores it in an array where each student's response is an element. This allows for us to do statistics on individual student's responses as well as looking at the student's as a whole. The benefits also include looking at n-grams for individual students instead of throughout the entire text. The expected input is a string with the path to the student's responses. This file should contain just one column with the responses and should be in the csv format.

FilterWords(tokenized_word, filter_nltk_stop_words, source): This allows for the filtering of the word tokenized content using a custom list of words located at source in the csv format. You can also choose to filter out nltk's stop words by setting filter_nltk_stop_words to False.

Stem(contents): Finds the stem of each word and returns the stemmed contents. For example, jumping and jumped both become jump. THis is helpful for statistics on the student's responses.

Lemmatize(contents): Finds the lemma of each word in contents. For example, great and awesome become good. this is another form of normalization for statistics on the response.s

SortFreqDist(fdist, descending): Takes a frequency distribution and sorts it in ascending or descending order.

MostCommon(sorted_array, num_words): Takes a sorted frequency distribution and returns the most common. The number of most common words returned is defined by num_words

PlotWordFrequency(mc_array, precision="4"): Takes the list of the most common words and plots them using MatPlotLib. The default precision for the frequency displayed is 4 digits.

Ngram(contents, n="2"): Creates an ngram from some contents

Collocation(contents, n): Finds the collocations of words in a text for n=2, 3, or 4.

CollocationTable(scored_gram, num_grams_to_plot=10): Prints a table of Collocations
