# Movie Recommender
- This project builds a movie recommender using collaborative filtering.

## In this folder, you will find these files and a folder with the movie data:

1. `movie_recommender_demo.ipynb`
2. `movie_recommender_service_class.py`
3. `movie_recommender_class.py`
4. `movie_recommendation_service.py`
5. `CFmodel.py`
6. `requirements.txt`
7. `Dockerfile`
8. `ml-100k`


The assignment includes the following files:

1. `movie_recommender_demo.ipynb`: This notebook demonstrate movie recommender API service. It also explores the movie dataset and shows collaborative filtering models with and without regularization.
2. `movie_recommender_service_class.py`: This data contains restaurant reviews and ground-truth sentiments.
3. `movie_recommender_class.py`: This script defines a MovieRecommender class for collaborative filtering-based movie recommendations using TensorFlow. It includes data loading, preprocessing, and model building functionalities. 
4. `movie_recommendation_service.py`: This file serves a simple web Flask API for movie recommendations, and it can be accessed through the specified routes. The recommendation results are rendered using an HTML template.
5. `CFmodel.py`: This Python file defines a class named CFModel, which represents a collaborative filtering model. The class is designed to be used with TensorFlow for training collaborative filtering models. 
6. `requirements.txt`: The requirements.txt file is used to install the necessary Python libraries for the service.
7. `Dockerfile`: folder used to build an docker image. 
8. `ml-100k`: folder with the MovieLens data downloaded from: https://grouplens.org/datasets/movielens/100k. This dataset contains 100,000 ratings from 1000 users on 1700 movies.

