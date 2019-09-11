import analysis_functions_with_pandas as af

reflection_df = af.ReadData("../Writings/ReflectiveWritingData.csv")

#Prints the first week's response for every student
#print("Reflection dataframe: \n", reflection_df.iloc[:,1], "\n")

#Prints every week's response for one student
#print("Reflection dataframe: \n", reflection_df.iloc[0,:], "\n")

#print(reflection_df.dtypes)

'''Filtering Text'''
'''******************************************************'''
#wt_df = af.WordTokenize(reflection_df)

filtered_df = af.FilterWords_AR(reflection_df, week_id="all", source="../Filters/filterwords.csv")
#print("Filtered dataframe one \n", filtered_frame.loc[:,1][0], "\n")

'''Creating frequency distribution and plotting it'''
'''********************************************************'''
#fdist = af.FreqDistWeek(filtered_df, week_id="PH201 Week 1")

#sorted_fd = af.SortFreqDist(fdist) # Creates a frequency distribution and organizes by frequency

'''Stemming'''
'''********************************************************'''
#stemmed_df = af.Stem(reflection_df)
#print(stemmed_df.iloc[:,1][1])
