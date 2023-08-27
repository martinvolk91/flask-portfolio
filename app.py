# recommender_api.py

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit

app = Flask(__name__)

# Load your recommender system from the pickle file
with open('movielens_implicit_cpu.pkl', 'rb') as file:
    model_load = pickle.load(file)


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_indices = data.get('movie_ids')

    # Create the data, indices, and indptr arrays
    data = np.ones(len(movie_indices))
    indices = np.array(movie_indices)
    indptr = np.array([0, len(movie_indices)])
    csr_matrix_manual = csr_matrix((data, indices, indptr))

    model_load.partial_fit_users([162542], csr_matrix_manual)
    ids, scores = model_load.recommend(162542, csr_matrix_manual)

    movies = pd.read_parquet("data/movies.parquet")
    recommended_movies = list(movies[movies.movie_id.isin(ids)].title)

    return jsonify(recommended_movies)


if __name__ == '__main__':
    app.run(port=8080)