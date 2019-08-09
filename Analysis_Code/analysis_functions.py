'''Function Definitions used in the Analysis Script'''
'''***********************************************************'''
#Reads in responses and stores each response in an element of an array
def ReadInResponses(file):
    import csv
    import numpy as np

    responses = []

    with open(file, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            responses.append(row[0])

        return(np.array(responses));

#Filters out stop words and a custom list of words from a file
def FilterWords(tokenized_word, filter_nltk_stop_words=False, source="../Filters/filterwords.csv"):
    from nltk.corpus import stopwords

    with open(source, "r") as myfile:
        filter_words = myfile.read()

        filtered_contents = []

        if filter_nltk_stop_words is True:
            filter_words = set(stopwords.words("english")) #Makes a set with the stopwords in the english dictionary

        for w in tokenized_word: #Looks at every word in the text,
            if w not in filter_words:
                filtered_contents.append(w)

        return(filtered_contents);

#Stems each word in the writings (reduces to the base word by getting rid of -ed, -ing, etc.) Input should be the filtered contents. Normalizes the data. May remove things llike a trailing e
def Stem(contents):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    stemmed_contents = []

    for word in contents:
        stemmed_contents.append(ps.stem(word))

    return(stemmed_contents);

#Reduces words to their lemmas, ex. fantastic and great simplify to good
def Lemmatize(contents):
    from nltk.stem.wordnet import WordNetLemmatizer

    lem = WordNetLemmatizer()

    lem_content = []

    for word in contents:
        lem_content.append(lem.lemmatize(word))

    return(lem_content);

#Sorts the frequency distribution in ascending or descending order
def SortFreqDist(fdist, descending=True):
    import numpy as np
    #fdist is the frequency distribution,
    #wordnum is the number
    #ascending true for sorting from most frequent to least, false for least frequent to most

    all_words = [] #Create a list for every word
    all_freq = [] #Create a list for every frequency

    for word in fdist.keys(): #Loops through each word in the freq dist, adds to lists
        all_words.append(word)
        all_freq.append(float(fdist.freq(word)))
    array = np.column_stack((all_words, all_freq)) #Creates an array from the word and freq list
    array_sorted = sorted(array,key=lambda x: x[1], reverse=descending)

    return array_sorted;

#Sorts a frequency distribution of ngrams
def SortFreqDistNgram(nfdist, descending=True):
    import numpy as np

    all_ngrams = []
    all_freq = []

    for gram in fdist.keys():
        all_ngrams.append(gram)
        all_freq.append(float(fdist.freq(gram)))

#Picks out the most common words from the frequency distribution
def MostCommon(sorted_array, num_words=15):
    import numpy as np

    some_words = []
    some_freqs = []

    for i in range(num_words): #Populates the nth most common words/freqs into lists
        some_words.append(sorted_array[i][0]) #Pulls entries from the first column of array
        some_freqs.append(round(float(sorted_array[i][1]), 4)) #Pulls entries from the second column of array
    mc_array = np.column_stack((some_words, some_freqs)) #table of most common words
    return(mc_array);


#Plots the frequency of the top n words
def PlotWordFrequency(mc_array, precision="4"):
    import matplotlib.pyplot as plt
    import numpy as np

    words = mc_array[:,0]
    freqs = mc_array[:,1]

    word_num = len(words)
    freqs_float = [] #Need a new list to turn the string of freqs in mc_array into floats

    for i in range(word_num): #Loops through each freqability, converts ot float with 4 digit precision
        freqs_float.append(round(float(freqs[i]), precision))

    x_pos = np.arange(len(words))
    axes = plt.bar(x_pos, freqs_float, align="center")

    plt.xticks(x_pos, words, rotation=45, ha = "right")
    plt.ylabel("Frequency of Word in Students' Writing")
    plt.title("%i most used words in \n students' reflective writings" %word_num)

    plt.tight_layout()
    plt.show()

    return;


#Takes a list of tokenized words
def Ngram(contents, n="2"):
    from nltk import ngrams

    ngram = ngrams(contents, n)

    return(ngram);

def Collocation(contents, n):

    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder, QuadgramAssocMeasures, QuadgramCollocationFinder

    from nltk.probability import FreqDist, DictionaryProbDist

    if n==2:
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(contents)
        scored = finder.score_ngrams(bigram_measures.raw_freq)
    elif n==3:
        trigram_measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(contents)
        scored = finder.score_ngrams(trigram_measures.raw_freq)
    elif n==4:
        quadgram_measures = QuadgramAssocMeasures()
        finder = QuadgramCollocationFinder.from_words(contents)
        scored = finder.score_ngrams(quadgram_measures.raw_freq)
    else:
        print("Collocation is only available for n=2, 3, or 4.")

    return(scored)

def CollocationTable(scored_gram, num_grams_to_plot=10):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    gram = list(scored_gram)

    gram_list = []
    freq_list = []
    words = ""

    for i in range(num_grams_to_plot):
        #Takes the elements in the ngram and turns it  into one string
        for k in range(len(gram[0][0])):
            words += gram[i][0][k]+ " "

        gram_list.append(words)
        words="" #resets words in the enxt ngram

        freq_list.append(gram[i][1]) #frequency associated with ngram

    collocation_vals = np.column_stack((gram_list, freq_list))

    col_labels = ("Bigram", "Collocation Value")

    fig, ax = plt.subplots()

    # Hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table = plt.table(cellText = collocation_vals, colLabels = col_labels, loc="center", cellLoc="center")

    for (row, col), cell in table.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    fig.tight_layout()
    plt.show()

def Sentiment(response):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    from nltk.tokenize import sent_tokenize
    import numpy as np

    sia = SIA()

    sent_scores = []

    st_response = list(sent_tokenize(response))

    for sentence in st_response:
        sent_scores.append(sia.polarity_scores(sentence))

    sent_scores = list(sent_scores)

    response_sentiment = np.column_stack((st_response, sent_scores))

    return(response_sentiment);

def PrintSentimentValues(response_array):

    for i in range(len(response_array)):
        print("*******Response number*******", i, "\n")
        sentiment_scores = Sentiment(response_array[i])

        for j in range(len(sentiment_scores)):
            print("Sentence:", sentiment_scores[j][0], "Sentiment", sentiment_scores[j][1], "\n")
