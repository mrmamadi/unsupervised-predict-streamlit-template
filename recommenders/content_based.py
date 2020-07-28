"""

    Content-based filtering for item recommendation.

<<<<<<< HEAD
    Author: JHB_EN1_UNSUPERVISED.

    Description: Provided within this file is a content
=======
    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
>>>>>>> master
    filtering algorithm for rating predictions on Movie data.

"""

# Importing packages and Data
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

os.path.exists('../movies.csv')

# Importing data
movies = pd.read_csv('../movies.csv')

# Initialising Vectoriser
vectoriser = CountVectorizer()
# Fitting Vectoriser and transforming content column
content_matrix = vectoriser.fit_transform(movies['content'])

# Creating Series of movieIds and indices for easier recall
indices = pd.Series(movies.index, index=movies['movieId'])

# @st.cache
def content_model(movie_list,top_n):

    """
    Predict a number of recommended movies based off the content of a film

    Parameters:
    -----------
    list_title : list
        List of movies from which to draw comparisons

    k : int
        Number of recommended movies to return

    Returns:
    ----------
    pandas.core.series.Series
        Series of k indexed movie films ranked by similarity to input list. 

    """
    # Vectorise content for each movie in list_title
    input_matrix = vectoriser.transform(movies[movies['title'].isin(movie_list)].content)
    
    # Initiate list to store indeces of input movies
    m_idx = []
    
    for title in movie_list:
        for id in movies.movieId[movies['title']==title]:
            m_idx.append(indices[id])
            
    # Create list of similarities between each input movie and every other movie in the dataset                   
    sim = list(enumerate(cosine_similarity(content_matrix,
                                       input_matrix)))   

    # Sort the list by the average similarity of the movies
    sim_scores = sorted(sim, key=lambda x: x[1].mean(), reverse=True)
                       
    # Select the top-k values for recommendation
    sim_scores = sim_scores[0:20]

    # Select the indices of the top-k movies
    movie_indices = [i[0] for i in sim_scores if i[0] not in m_idx]
    
    # Return a list of the movie titles
    return movies.iloc[movie_indices].title[:top_n] 
