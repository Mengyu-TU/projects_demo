from movie_recommender_class import MovieRecommender


class MovieRecommenderServiceClass(MovieRecommender):
    def __init__(self):
        # Call the constructor of the parent class
        super().__init__()
        self.model_learn()


    def model_learn(self, print_int_results=False):
        self.model = self.build_regularized_model(
            regularization_coeff=0.1, gravity_coeff=1.0, embedding_dim=35,
            init_stddev=.05)  
        self.stats = self.model.train(num_iterations=2000, learning_rate=20., print_int_results=print_int_results)

    
    def movie_recommend(self, model, movie):
        return self.movie_neighbors_flask(self.model, movie)

    
    def model_stats(self):
        return str(self.stats)