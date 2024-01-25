from flask import Flask, render_template_string
from flask import request
import os
from movie_recommender_service_class import MovieRecommenderServiceClass

"""Flask app for hosting a rest API of the movie recommender"""

app = Flask(__name__)

# http://localhost:8786/recommend?movie_name=Aladdin and
# http://localhost:8786/recommend?movie_name=Aladdin
# http://localhost:8786/recommend?movie_name=Lion King
# http://localhost:8786/stats

@app.route('/stats', methods=['GET'])
def getStats():
    return str(recommender.model_stats())


@app.route('/recommend', methods=['GET'])
def getInfer():
    movies_recommended = recommender.movie_neighbors_flask(recommender.model, request.args.get('movie_name'))
    return render_template_string(open('result_template.html').read(), result=movies_recommended)


if __name__ == "__main__":
    flaskPort = 8786
    recommender = MovieRecommenderServiceClass()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort, debug=False)

