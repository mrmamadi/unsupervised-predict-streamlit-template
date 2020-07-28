"""

    Collaborative-based filtering for item recommendation.

    Author: JHB_EN1_UNSUPERVISED.

<<<<<<< HEAD
    Description: Provided within this file is a collaborative
=======
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
>>>>>>> master
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Importing data
movies = pd.read_csv('../movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

reader = Reader()
data = Dataset.load_from_df(ratings_df, reader)
trainset = data.build_full_trainset()

# Creating Series of movieIds and indices for easier recall
indices = pd.Series(ratings_df.index, index=ratings_df['userId'])

# Building the Model
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def collab_model(movie_list,top_n=10):
    
    # List of users who have rated the input movies
    ids = [movies.movieId[movies['title']==i].values[0] for i in movie_list]
    # Declaring model as object
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

    # Fit the training dataset
    model_knn.fit(ratings_df)

    distances,s_indices = model_knn.kneighbors(ratings_df[ratings_df['movieId'].isin(ids)],
                                                n_neighbors=50)
    # rat = list(zip(distances,s_indices))
    # sim_scores = sorted(sim, key=lambda x: x[0].mean(), reverse=True)
    
    io = 0
    new = []
    for i in s_indices[io]:
        new.append(ratings_df.userId[i])
    mvs = movies[movies['movieId'].isin(new)]

    while len(mvs)<10:
        io += 1 
        for i in s_indices[io]:
            new.append(ratings_df.userId[i])
        mvs = movies[movies['movieId'].isin(new)]

    return mvs.title[:top_n]



