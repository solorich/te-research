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

#filtered_df = af.FilterWords_AR(reflection_df, week_id="all", source="../Filters/filterwords.csv")
#print("Filtered dataframe one \n", filtered_frame.loc[:,1][0], "\n")

'''Stemming'''
'''********************************************************'''
#stemmed_df = af.Stem(reflection_df)
#print(stemmed_df.iloc[:,1][1])


'''Creating frequency distribution for a given week'''
'''********************************************************'''
#fdist = af.FreqDistWeek(filtered_df, week_id="PH201 Week 3")
#sorted_fd = af.SortFreqDist(fdist) # Creates a frequency distribution and organizes by frequency
#print(sorted_fd)

'''Creating frequency distribution for a given student'''
'''********************************************************'''
#fdist = af.FreqDistStudent(filtered_df, stu_num=316840)
#sorted_fd = af.SortFreqDist(fdist)
#print(sorted_fd)

'''Sentiment Analysis'''
'''********************************************************'''
sent_df = af.SentimentAnalysis(reflection_df)
af.Plot3DSA(sent_df)

'''
# library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get the data (csv file is hosted on the web)
url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)
print(data)

# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]

# And transform the old column name in something numeric
df['X']=pd.Categorical(df['X'])
df['X']=df['X'].cat.codes
print(df)

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.show()

# to Add a color bar which maps values to colors.
surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show()

# Rotate it
ax.view_init(30, 45)
plt.show()
'''
