# Importing packages and Data
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# os.path.exists('../resources/data/movies.csv')

# Importing data
movies = pd.read_csv('../movies.csv') # Remove this to be modular
# ratings = pd.read_csv('resources/data/ratings.csv') # Remove this to be modular

# Initialising Vectoriser
vectoriser = CountVectorizer()
# Fitting Vectoriser and transforming content column
content_matrix = vectoriser.fit_transform(movies['content'])

# movieIds = movies['movieId']

# Creating Series of movieIds and indices for easier recall
indices = pd.Series(movies.index, index=movies['movieId'])

def content_model(list_title,k=20):

    # Vectorise content for each movie in list_title
    input_matrix = vectoriser.transform(movies.content[movies['title'].isin(list_title)])

    # Initiate list to store indeces of input movies
    m_idx = []
    for title in list_title:
        m_idx.append(indices[movies.movieId[movies['title']==title]].values)
    # Create list of similarities between each input movie and every other movie in the dataset                   
    sim = list(enumerate(cosine_similarity(content_matrix,
                                       input_matrix)))   

    # Sort the list by the average similarity of the movies
    sim_scores = sorted(sim, key=lambda x: x[1].mean(), reverse=True)
                       
    # Select the top-k values for recommendation
    sim_scores = sim_scores[0:k]

    # Select the indices of the top-k movies
    movie_indices = [i[0] for i in sim_scores if i[0] not in m_idx]
    
    # Return a list of the movie titles
    return movies.iloc[movie_indices].title   