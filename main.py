import tensorflow as tf
from colabcode import ColabCode
from fastapi import FastAPI
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

cc = ColabCode(port=8000, code=False)
# api test
app = FastAPI()
# from tensorflow import keras

print(tf.version.VERSION)

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('movie_on.pt')

# Show the model architecture
# model.summary()

# setup port
cc = ColabCode(port=8000, code=False)
# api test
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/movie/get-list-suggest/{userId}", tags=["suggest"])
async def get_predictions(userId: int):
    try:
        # Load the data
        df = pd.read_csv("csv/ratings.csv")

        # Prepare training and validation data
        df = df.sample(frac=1, random_state=42)
        user_ids = df["userId"].unique().tolist()
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}

        movie_ids = df["movieId"].unique().tolist()
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

        user_id = np.int64(userId)
        movies_watched_by_user = df[df.userId == user_id]

        # Print  movies watched by user
        movie_df = pd.read_csv(
            "csv/movies.csv")

        movies_not_watched = movie_df[
            ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
        ]["movieId"]
        movies_not_watched = list(
            set(movies_not_watched).intersection(
                set(movie2movie_encoded.keys()))
        )

        movies_not_watched = [
            [movie2movie_encoded.get(x)] for x in movies_not_watched]
        user_encoder = user2user_encoded.get(user_id)
        user_movie_array = np.hstack(
            ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
        )

        user_movie_array_type_int64 = user_movie_array.astype(np.int64)
        ratings = model.predict(user_movie_array_type_int64).flatten()

        top_ratings_indices = ratings.argsort()[-5:][::-1]
        recommended_movie_ids = [
            movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
        ]

        # top_movies_user = (
        #     movies_watched_by_user.sort_values(by="rating", ascending=False)
        #     .head(5)
        #     .movieId.values
        # )
        # movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
        # for row in movie_df_rows.itertuples():
        #     print(row.title, ":", row.genres)

        recommended_movies = movie_df[movie_df["movieId"].isin(
            recommended_movie_ids)]
        response = []
        for row in recommended_movies.itertuples():
            print(row.title, ":", row.movieId)
            response.append(str(row.movieId))

        return {"data": response}
    except:
        return {"suggest": "error roi"}

# run app
cc.run_app(app=app)
