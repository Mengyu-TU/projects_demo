# Movie recommender class
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from CFmodel import CFModel
from IPython import display
import sklearn
from sklearn.manifold import TSNE
tf.disable_v2_behavior()


class MovieRecommender:
    def __init__(self):
        self.ratings = pd.DataFrame()
        self.users = pd.DataFrame()
        self.movies = pd.DataFrame()
        self.full_movie_user_df = pd.DataFrame()
        self.genre_cols = [
            "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]
        # Load the movie data
        self.load_data()
        
    def load_data(self):
        # Load users dataset
        users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')
        self.users["user_id"] = self.users["user_id"].apply(lambda x: str(x-1))

        # Load ratings dataset
        ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')
        self.ratings["movie_id"] = self.ratings["movie_id"].apply(lambda x: str(x-1))
        self.ratings["user_id"] = self.ratings["user_id"].apply(lambda x: str(x-1))
        self.ratings["rating"] = self.ratings["rating"].apply(lambda x: float(x))

        # Load movies dataset
        movies_cols = [
            'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
        ] + self.genre_cols
        self.movies = pd.read_csv('ml-100k/u.item', sep='|', names=movies_cols, encoding='latin-1')
        self.movies["movie_id"] = self.movies["movie_id"].apply(lambda x: str(x-1))
        self.movies["year"] = self.movies['release_date'].apply(lambda x: str(x).split('-')[-1])

        # Tidy up genres in self.movies
        self.mark_genres()
        
        # Merge into a DataFrame with all the movie and user information
        self.merge_movie_df()
        
    def mark_genres(self):
        """get the genre in string formats"""
        def get_random_genre(gs):
            active = [genre for genre, g in zip(self.genre_cols, gs) if g == 1]
            np.random.seed(2023) 
            return np.random.choice(active) if active else 'Other'

        def get_all_genres(gs):
            active = [genre for genre, g in zip(self.genre_cols, gs) if g == 1]
            return '-'.join(active) if active else 'Other'

        self.movies['genre'] = [get_random_genre(gs) for gs in zip(*[self.movies[genre] for genre in self.genre_cols])]
        self.movies['all_genres'] = [get_all_genres(gs) for gs in zip(*[self.movies[genre] for genre in self.genre_cols])]

    
    def merge_movie_df(self):
        """Merge data frame movies, ratings and users"""
        self.full_movie_user_df = self.ratings.merge(self.movies, on='movie_id').merge(self.users, on='user_id')

    ##################################################################
    ##################################################################
    # Collaborative filtering helper functions
    ##################################################################
    ##################################################################

    def build_sparse_tensor(self, ratings):
        """Build sparse rating tensor"""
        num_users = self.users.shape[0]
        num_movies = self.movies.shape[0]
        indices = ratings[['user_id', 'movie_id']].values
        values = ratings['rating'].values
        dense_shape = [num_users, num_movies] # number users x number movies
        return tf.SparseTensor(indices, values, dense_shape)

    
    # Utility to split the data into training and test sets.
    def split_dataframe(self, df, holdout_fraction=0.1):
        """ 
        Splits a DataFrame into training and test sets.
          Args:
            df: a dataframe.
            holdout_fraction: fraction of data frame rows to use in the test set.
          Returns:
            train: dataframe for training
            test: dataframe for testing
        """
        test = df.sample(frac=holdout_fraction, replace=False)
        train = df[~df.index.isin(test.index)]
        return train, test


    def sparse_mse(self, sparse_ratings, user_embeddings, movie_embeddings):
        """
        Args:
            sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
            U: user_embeddings. A dense Tensor U of shape [N, k] where k is the embedding
            dimension, such that U_i is the embedding of user i.
            V: movie_embeddings. A dense Tensor V of shape [M, k] where k is the embedding
              dimension, such that V_j is the embedding of movie j.
        Returns:
            A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = tf.reduce_sum(
            tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
            tf.gather(movie_embeddings, sparse_ratings.indices[:, 1]), 
            axis=1)
        loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
        return loss

    
    def gravity(self, U, V):
        """Creates a gravity loss given two embedding matrices."""
        return 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
            tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))


    def regularized_loss(self, sparse_ratings, U, V, gravity_coeff, regularization_coeff):
        """Regularized loss: mse loss + gravity loss and regularization loss"""
        loss = self.sparse_mse(sparse_ratings, U, V)
        gravity_loss = gravity_coeff * self.gravity(U, V)
        regularization_loss = regularization_coeff * (
            tf.reduce_sum(U*U)/U.shape[0].value + tf.reduce_sum(V*V)/V.shape[0].value)
        total_loss = loss + regularization_loss + gravity_loss
        return loss, gravity_loss, regularization_loss, total_loss

    
    def build_model(self, embedding_dim=3, init_stddev=1.):
        """
        Args:
        ratings: a DataFrame of the rating
        embedding_dim: the dimension of the embedding vectors.
        init_stddev: float, the standard deviation of the random initial embeddings.
        Returns:
        model: a CFModel.
        """
        # Split the ratings DataFrame into train and test.
        train, test = self.split_dataframe(self.ratings)

        # SparseTensor representation of the train and test datasets.
        self.rating_train = self.build_sparse_tensor(train)
        self.rating_test = self.build_sparse_tensor(test)

        # Initialize the embeddings using a normal distribution.
        U = tf.Variable(tf.random_normal(
            [self.rating_train.dense_shape[0], embedding_dim], stddev=init_stddev))
        V = tf.Variable(tf.random_normal(
            [self.rating_train.dense_shape[1], embedding_dim], stddev=init_stddev))

        # Loss 
        train_loss = self.sparse_mse(self.rating_train, U, V)
        test_loss = self.sparse_mse(self.rating_test, U, V)
    
        metrics = {
            'train_error': train_loss,
            'test_error': test_loss
        }
        embeddings = {
            "user_id": U,
            "movie_id": V
        }
        return CFModel(embeddings, train_loss, [metrics])

    
    def build_regularized_model(self, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
        init_stddev=0.1):
        """
        Args:
        ratings: the DataFrame of movie ratings.
        embedding_dim: The dimension of the embedding space.
        regularization_coeff: The regularization coefficient lambda.
        gravity_coeff: The gravity regularization coefficient lambda_g.
        Returns:
        A CFModel object that uses a regularized loss.
        """
        # Split the ratings DataFrame into train and test.
        train, test = self.split_dataframe(self.ratings)

        # SparseTensor representation of the train and test datasets.
        self.rating_train = self.build_sparse_tensor(train)
        self.rating_test = self.build_sparse_tensor(test)

        # Initialize the embeddings using a normal distribution.
        U = tf.Variable(tf.random_normal(
            [self.rating_train.dense_shape[0], embedding_dim], stddev=init_stddev))
        V = tf.Variable(tf.random_normal(
            [self.rating_train.dense_shape[1], embedding_dim], stddev=init_stddev))

        # Loss 
        train_loss = self.sparse_mse(self.rating_train, U, V)
        test_loss = self.sparse_mse(self.rating_test, U, V)

        _, gravity_loss, regularization_loss, total_loss = self.regularized_loss(
            self.rating_train, U, V, gravity_coeff, regularization_coeff)
    
        losses = {
            'train_error_observed': train_loss,
            'test_error_observed': test_loss,
            }
        loss_components = {
            'observed_loss': train_loss,
            'regularization_loss': regularization_loss,
            'gravity_loss': gravity_loss,
            }
        embeddings = {"user_id": U, "movie_id": V}

        return CFModel(embeddings, total_loss, [losses, loss_components])

    ##################################################################
    ##################################################################
    # Select movies for recommendation after learning
    ##################################################################
    ##################################################################
    def compute_scores(self, query_embedding, item_embeddings, measure='dot'):
        """Computes the scores of the candidates given a query.
        Args:
            query_embedding: a vector of shape [k], representing the query embedding.
            item_embeddings: a matrix of shape [N, k], such that row i is the embedding
            of item i.
            measure: a string specifying the similarity measure to be used. Can be
            either DOT or COSINE.
        Returns:
            scores: a vector of shape [N], such that scores[i] is the score of item i.
        """
        u = query_embedding
        V = item_embeddings
        if measure == 'cosine':
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            u = u / np.linalg.norm(u)
        scores = u.dot(V.T)
        return scores


    def movie_neighbors(self, model, title_substring, measure='dot', k=6):
        """Recommend movies in Jupyter lab: directly print to screen"""
        # Search for movie ids that match the given substring.
        ids =  self.movies[self.movies['title'].str.contains(title_substring)].index.values
        titles = self.movies.iloc[ids]['title'].values
        if len(titles) == 0:
            raise ValueError("Found no movies with title %s" % title_substring)
        print("Nearest neighbors of : %s." % titles[0])
        if len(titles) > 1:
            print("[Found more than one matching movie. Other candidates: {}]".format(
            ", ".join(titles[1:])))
        movie_id = ids[0]
        scores = self.compute_scores(
            model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
            measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores),
            'titles': self.movies['title'],
            'genres': self.movies['all_genres']
        })
        # Find the index of the row with the same name as itself and remove that row
        index_to_remove = df[df['titles'] == titles[0]].index
        df = df.drop(index_to_remove)
        
        display.display(df.sort_values([score_key], ascending=False).head(k))


    def movie_neighbors_flask(self, model, title_substring, measure='dot', k=6):
        """Recommend movies for Flask app"""
        # Search for movie ids that match the given substring.
        ids = self.movies[self.movies['title'].str.contains(title_substring)].index.values
        titles = self.movies.iloc[ids]['title'].values
        if len(titles) == 0:
            raise ValueError("Found no movies with title %s" % title_substring)

        result = {"movie_title": titles[0], "neighbors": []}

        if len(titles) > 1:
            result["other_candidates"] = titles[1:]
            # Convert NumPy arrays to lists
            result["other_candidates"] = result["other_candidates"].tolist()

        movie_id = ids[0]
        scores = list(self.compute_scores(
            model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"],
            measure))

        df = pd.DataFrame({
            'score': scores,
            'titles': self.movies['title'],
            'genres': self.movies['all_genres']
        })

        # Find the index of the row with the same name as itself and remove that row
        index_to_remove = df[df['titles'] == titles[0]].index
        df = df.drop(index_to_remove)
        
        # Convert the DataFrame to a JSON object
        result["neighbors"] = df.sort_values(['score'], ascending=False).head(k).to_dict(orient='records')
        # Format the 'score' values in the resulting JSON
        for record in result["neighbors"]:
            record['score'] = "{:.2f}".format(record['score'])
            
        return result
    ##################################################################
    ##################################################################
    # Visualize embeddings.
    ##################################################################
    ##################################################################
    def movie_tsne(self, model):
        tsne = sklearn.manifold.TSNE(
            n_components=2, perplexity=40, metric='cosine', early_exaggeration=10.0,
            init='pca', verbose=True, n_iter=400
        )

        V_proj = tsne.fit_transform(model.embeddings["movie_id"])
        self.movies.loc[:, 'x'] = V_proj[:, 0]
        self.movies.loc[:, 'y'] = V_proj[:, 1]

    def plt_tsne(self, selected_categories, reg):
        # Filter movies DataFrame based on selected categories
        selected_movies = self.movies[self.movies['genre'].isin(selected_categories)]

        # Create a scatter plot
        plt.figure(figsize=(5, 5))
        for category in selected_categories:
            category_movies = selected_movies[selected_movies['genre'] == category]
            indices = category_movies.index
            plt.scatter(category_movies['x'], category_movies['y'], label=category)

        # Add labels and legend
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        if reg:
            plt.title('Regularized Model: t-SNE of Movie Embeddings')
        else:
            plt.title('Unregularized Model: t-SNE of Movie Embeddings')
        plt.legend()

        # Show the plot
        plt.show()
