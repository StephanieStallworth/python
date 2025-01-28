
# coding: utf-8

# # Item-Based Collaborative Filtering

# As before, we'll start by importing the MovieLens 100K data set into a pandas DataFrame:

# In[1]:


import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

ratings.head()


# Now we'll pivot this table to construct a nice matrix of users and the movies they rated. NaN indicates missing data, or movies that a given user did not watch:

# In[2]:


userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()


# Now the magic happens - pandas has a built-in corr() method that will compute a correlation score for every column pair in the matrix! This gives us a correlation score between every pair of movies (where at least one user rated both movies - otherwise NaN's will show up.) That's amazing!

# In[3]:


corrMatrix = userRatings.corr()
corrMatrix.head()


# However, we want to avoid spurious results that happened from just a handful of users that happened to rate the same pair of movies. In order to restrict our results to movies that lots of people rated together - and also give us more popular results that are more easily recongnizable - we'll use the min_periods argument to throw out results where fewer than 100 users rated a given movie pair:

# In[4]:


corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()


# Now let's produce some movie recommendations for user ID 0, who I manually added to the data set as a test case. This guy really likes Star Wars and The Empire Strikes Back, but hated Gone with the Wind. I'll extract his ratings from the userRatings DataFrame, and use dropna() to get rid of missing data (leaving me only with a Series of the movies I actually rated:)

# In[5]:


myRatings = userRatings.loc[0].dropna()
myRatings


# Now, let's go through each movie I rated one at a time, and build up a list of possible recommendations based on the movies similar to the ones I rated.
# 
# So for each movie I rated, I'll retrieve the list of similar movies from our correlation matrix. I'll then scale those correlation scores by how well I rated the movie they are similar to, so movies similar to ones I liked count more than movies similar to ones I hated:

# In[6]:


simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))


# This is starting to look like something useful! Note that some of the same movies came up more than once, because they were similar to more than one movie I rated. We'll use groupby() to add together the scores from movies that show up more than once, so they'll count more:

# In[7]:


simCandidates = simCandidates.groupby(simCandidates.index).sum()


# In[8]:


simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)


# The last thing we have to do is filter out movies I've already rated, as recommending a movie I've already watched isn't helpful:

# In[9]:


filteredSims = simCandidates.drop(myRatings.index)
filteredSims.head(10)


# There we have it!

# ## Exercise

# Can you improve on these results? Perhaps a different method or min_periods value on the correlation computation would produce more interesting results.
# 
# Also, it looks like some movies similar to Gone with the Wind - which I hated - made it through to the final list of recommendations. Perhaps movies similar to ones the user rated poorly should actually be penalized, instead of just scaled down?
# 
# There are also probably some outliers in the user rating data set - some users may have rated a huge amount of movies and have a disporportionate effect on the results. Go back to earlier lectures to learn how to identify these outliers, and see if removing them improves things.
# 
# For an even bigger project: we're evaluating the result qualitatively here, but we could actually apply train/test and measure our ability to predict user ratings for movies they've already watched. Whether that's actually a measure of a "good" recommendation is debatable, though!
