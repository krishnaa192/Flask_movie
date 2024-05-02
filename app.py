from flask import Flask, request, render_template
import joblib
import os
import pandas as pd
import gzip

# Initialize the Flask app
app = Flask(__name__)
with gzip.open("movie_recommender.pkl.gz", "rb") as f_in:
    with open("movie_recommender.pkl", "wb") as f_out:
        f_out.write(f_in.read())
# Load the model components
model_path = os.path.join(os.path.dirname(__file__), "movie_recommender.pkl")
tfidf, cosine_sim, indices,df_movie = joblib.load(model_path)

# Load the movie dataset


# Define the route for user input and recommendations
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        recommendations = get_recommendations(movie_title, cosine_sim)
    
    return render_template("index.html", recommendations=recommendations,movie_title=movie_title)

# Function to get recommendations
def get_recommendations(title, similarity_matrix):
    if title not in indices:
        return ["Movie not found in the dataset."]
    
    idx = indices[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    sim_indices = [i[0] for i in sim_scores]
    recommended_movies = df_movie["original_title"].iloc[sim_indices].tolist()

    return recommended_movies

# Start the Flask app in debug mode
if __name__ == "__main__":
    app.run(debug=True)
