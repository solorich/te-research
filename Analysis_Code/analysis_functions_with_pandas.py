def ReadData(file):
    import pandas as pd

    pd.options.mode.chained_assignment = None  # default='warn'

    df = pd.read_csv(file, encoding="utf8")

    df.fillna('',inplace=True)
    #data = data.set_index('Number')
    return(df);

def WordTokenize(df):
    #Tokenizes each response in the dataframe by word. This is important for most other operations that act on the dataframe.
    from nltk.tokenize import WhitespaceTokenizer

    local_df = df.copy(deep=True) #Super important. It makes a copy of the dataframe. We want to manipulate the copy and leave the original unchanged

    #print("Called WordTokenize. Printing local_df... \n")
    #print("cp_data: \n", local_df, "\n")

    for i in range(1, len(local_df.columns)):
        for j in range(len(local_df)):
            wt_response = WhitespaceTokenizer().tokenize(str(local_df.iloc[:,i][j]))
            local_df.iloc[:,i][j] = wt_response

    return(local_df);

'''
def FilterWords_IR(df, student_id, filter_nltk_stop_words=False, source="../Filters/filterwords.csv")
    #Filters the words from a given list from an individual student's response (IR) for each week.

    local_df = df.copy()
    wt_df = WordTokenize(local_df)
'''

def FilterWords_AR(df, week_id="all", filter_nltk_stop_words=False, source="../Filters/filterwords.csv"):
    #Filters the words from a given list from all responses (AR) in the dataframe. It can also filter words from a given list from all responses for a given week. It will return a dataframe if you filter words for each week. It will return an array if you filter just one week.

    import re

    local_df = df.copy()
    wt_df = WordTokenize(local_df)

    with open(source, "r") as myfile:
        filter_words = myfile.read()

        filtered_contents = []

        if filter_nltk_stop_words is True:
            filter_words = set(stopwords.words("english")) #Makes a set with the stopwords in the english dictionary

        if week_id == "all":
            #Main loop takes a column, goes through each row, then moves onto next column to filter out the stop words in each entry from each week
            for i in range(1, len(local_df.columns)):
                for j in range(len(local_df)):
                    wt_response = wt_df.iloc[:,i][j]

                    wt_response = [w.lower() for w in wt_response]

                    for w in wt_response:
                        w = re.sub('\ |\?|\.|\!|\/|\;|\:|\,|\)|\(|\]|\[|\"', '', w)

                        if w not in filter_words:
                            filtered_contents.append(w)

                    local_df.iloc[:,i][j] = filtered_contents

                    filtered_contents = []

                    #print(local_df.iloc[:,1][j], "\n")

            return_df1 = local_df.copy(deep=True)
            return(return_df1);

        if week_id != "all":
            import pandas as pd

            week_col = local_df.columns.get_loc(week_id)

            week_df = pd.DataFrame(local_df.iloc[:,0])
            week_df.insert(1, week_id, local_df.iloc[:,week_col])

            for j in range(len(local_df)):
                wt_response = wt_df.iloc[:,1][j]

                for w in wt_response:
                    w = re.sub('\ |\?|\.|\!|\/|\;|\:|\,|\)|\(|\]|\[|\"', '', w)

                    if w not in filter_words:
                        filtered_contents.append(w)

                week_df.iloc[:,1][j] = filtered_contents

                filtered_contents=[]
            return(week_df);

def SortFreqDist(fdist, descending=True):
    import numpy as np
    import pandas as pd

    words = [] #Create a list for every word
    freq_of_word = [] #Create a list for every frequency

    for w in fdist.keys(): #Loops through each word in the freq dist, adds to lists
        words.append(w)
        freq_of_word.append(float(fdist.freq(w)))

    #This df has each word and its associated frequency, but in order they appear in fdist.
    fd_df = pd.DataFrame({'Word': words, 'Frequency': freq_of_word})

    #Now we sort the frequency diagram dataframe.
    sorted_fd_df = fd_df.sort_values(by=['Frequency'], ascending=False)
    print(sorted_fd_df)

    return;

def FreqDistWeek(df, week_id):
    #This will do a frequency distribution on each student's response for a given week

    from nltk.probability import FreqDist, DictionaryProbDist
    import numpy as np

    local_df = df.copy() #Creates a local copy of the word tokenized dataframe

    #Tries to find the column with the given week ID
    try:
        week_col = local_df.columns.get_loc(week_id)
    except:
        print("There is no column named", week_id)
        return("")

    word_list = [] #Initializes an empty list that will store each word

    #Loops through each student's response for a given week and adds it to the word_list array
    for i in range(len(local_df)):
        word_list.extend(local_df.iloc[:,week_col][i])

    #Creates the frequency distribution for that list of words
    fdist = FreqDist(word_list)

    return(fdist);

def FreqDistStudent(df, stu_num):
    from nltk.probability import FreqDist, DictionaryProbDist
    import numpy as np

    local_df = df.copy()

    #Tries to find the row of a given student based off of the provided student number
    try:
        student = local_df['Number'] == stu_num #Looks for rows that match student number
        student_df = local_df[student] #Takes subsection of the local_df that containts only the one student's responses for each week
    except:
        print("There is no student with ID number", stu_num, "Or some error occured")
        return("")

    word_list = []

    #Puts each word from each week from the student's response into a list. For i=0, you get the column with the student's number. We don't need that, so we go from 1 to the last column for the student's row
    for i in range(1,len(student_df.iloc[0])):
        word_list.extend(student_df.iloc[0][i])

    fdist = FreqDist(word_list)

    return(fdist);

#Stems each word in the writings (reduces to the base word by getting rid of -ed, -ing, etc.) Input should be the filtered contents. Normalizes the data. May remove things llike a trailing e
def Stem(df):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    local_df = df.copy()
    wt_df = WordTokenize(local_df)

    stemmed_contents = []

    for i in range(1, len(local_df.columns)):
        for j in range(len(local_df)):

            for w in wt_df.iloc[:,i][j]:
                stemmed_contents.append(ps.stem(w))

            wt_df.iloc[:,i][j] = stemmed_contents

            stemmed_contents = []

            #print(local_df.iloc[:,1][j], "\n")

    return(wt_df);

#Reduces words to their lemmas, ex. fantastic and great simplify to good
def Lemmatize(df):
    from nltk.stem.wordnet import WordNetLemmatizer

    local_df = df.copy()
    wt_df = WordTokenize(local_df)

    lem_contents = []

    for i in range(1, len(local_df.columns)):
        for j in range(len(local_df)):

            for w in wt_df.iloc[:,i][j]:
                lem_content.append(lem.lemmatize(w))

            wt_df.iloc[:,i][j] = lem_contents

            lem_contents = []

            #print(local_df.iloc[:,1][j], "\n")

    return(wt_df);

def ResponseAvgCompoundScore(response):
    #This funciton takes in a response from the reflection dataframe and tokenizes each sentence in each response and then does a sentiment analysis on each sentence, and averages it for the response. This function returns a dataframe that contains student ID and the student's average compound sentiment score for each week.

    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
    from nltk.tokenize import sent_tokenize
    import numpy as np

    st_response = list(sent_tokenize(response)) #Tokenizes response into sentences

    sia = SIA() #Sentence
    avg = 0 #Initializes the average to be zero

    for sentence in st_response:
        avg += sia.polarity_scores(sentence)['compound']

    if len(st_response) != 0:
        avg = avg/len(st_response)
    else:
        #avg = "NaN" #Not sure if NaN or 0 is better. NaN is more true, but we can't plot NaN
        avg = 0

    return(avg)

def SentimentAnalysis(df):
    #This function will do a sentiment analysis on each student's response each week and then create a data frame with each student's average compound sentiment score for each week

    local_df = df.copy()

    for i in range(1, len(local_df.columns)):
        for j in range(len(local_df)):
            resp_avg = ResponseAvgCompoundScore(local_df.iloc[:,i][j]) #Gets the average compound score for one response from one week

            local_df.iloc[:,i][j] = resp_avg

    local_df = local_df.astype(float)

    local_df = local_df.astype({"Number":int})

    return(local_df)

def Plot3DSA(sent_df):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import pandas as pd

    local_df = sent_df.copy()

    x = np.arange(len(local_df.columns))
    y = local_df.iloc[:,1][1]

    print(y)
    '''
    X, Y = np.meshgrid(x,y)

    Z = local_df.drop('Number', axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    '''
