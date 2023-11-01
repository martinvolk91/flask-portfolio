from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from flask_mail import Mail, Message
import configparser
import implicit

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.secret_key = 'my_secrest_key'

# Load your recommender system from the pickle file
with open('movielens_implicit_cpu_v02.pkl', 'rb') as file:
    model_load = pickle.load(file)


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_indices = data.get('movie_ids')

    print(f"Input: {movie_indices}")

    # Create the data, indices, and indptr arrays
    data = np.ones(len(movie_indices))
    indices = np.array(movie_indices)
    indptr = np.array([0, len(movie_indices)])
    csr_matrix_manual = csr_matrix((data, indices, indptr))

    model_load.partial_fit_users([162542], csr_matrix_manual)
    ids, scores = model_load.recommend(162542, csr_matrix_manual)

    movies = pd.read_parquet("data/movies.parquet")
    recommended_movies = list(movies[movies.movie_id.isin(ids)].title)

    print(f"Output: {recommended_movies}")
    return jsonify(recommended_movies)


@app.route('/api/contact', methods=['POST'])
def contact():
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    mail = Mail()
    app.config["MAIL_SERVER"] = "smtp.gmail.com"
    app.config["MAIL_PORT"] = 465
    app.config["MAIL_USE_SSL"] = True
    app.config["MAIL_USERNAME"] = config.get('Email', 'username')
    app.config["MAIL_PASSWORD"] = config.get('Email', 'password')
    mail.init_app(app)

    data = request.get_json()
    print(data)
    msg = Message(
        subject=f"New message from your website - {data['name']}",
        recipients=["martinvolk91@gmail.com"],
        body=data["message"],
        sender=data["email"],
        reply_to=data["email"]
    )

    mail.send(msg)
    return "Message sent!"


if __name__ == '__main__':
    app.run(port=8080)
