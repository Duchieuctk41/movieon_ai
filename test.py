!pip install colabcode
!pip install fastapi
# import
from colabcode import ColabCode
from fastapi import FastAPI
#setup port
cc = ColabCode(port=8000, code=False)
# api test 
app = FastAPI()

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab_Notebooks/ml-latest-small

# view folder
# %ls

# Download data từ http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

movielens_dir = "/ml-latest-small" # Dir of ml-latest-small
ratings_file =  "ratings.csv" # file ratings.csv

df = pd.read_csv(ratings_file)
# print(df)
movie_names=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/ml-latest-small/movies.csv")

movie_data = pd.merge(df,movie_names,on='movieId')
movie_data = movie_data.drop(['genres','timestamp'],axis= 1)

# print(movie_data)
# Tạo Dataframe để thống kê
trend=pd.DataFrame(movie_data.groupby('title')['rating'].mean())
trend['total number of ratings'] = pd.DataFrame(movie_data.groupby('title')['rating'].count()) 
# print(trend)

#plot ratings with number of movies
# plt.figure(figsize =(10, 4))
ax=plt.barh(trend['rating'].round(),trend['total number of ratings'],color='g')
# plt.show()
#a bar graph 25 movies
# plt.figure(figsize =(10, 4))
ax=plt.subplot()
ax.bar(trend.head(25).index,trend['total number of ratings'].head(25),color='g')
ax.set_xticklabels(trend.index,rotation=40,fontsize='12',horizontalalignment="right")
ax.set_title("Total Number of reviews for each movie")
# plt.show()
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df["rating"] = df["rating"].values.astype(np.float32)

min_rating = min(df["rating"])
max_rating = max(df["rating"])

# print(
#     "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
#         num_users, num_movies, min_rating, max_rating
#     )
# )
# Prepare training and validation data

df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Chia tập dữ liệu 
train_indices = int(0.75 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)
# Create the model
EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    # get config
    def get_config(self):
      print("hello AI")
      return {
          "num_users": self.num_users,
          "num_movies": self.num_movies,
          "embedding_size": self.embedding_size,
      }

    # from config 
    @classmethod
    def from_config(cls, config):
      return cls(num_users=config['num_users'],
               num_movies=config['num_movies'],
               embedding_size=config['embedding_size'])

    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE,)

# Retrieve the config
config = model.get_config()
new_model = keras.Model.from_config(config)

new_model.compile( 
    keras.optimizers.Adam(learning_rate=0.001), 
    loss="mean_squared_error",
    metrics=["mean_absolute_error", "mean_squared_error"])

#train model
history = new_model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_val, y_val),
)

# model.summary()
# Plot training and validation loss
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title("model loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["train", "valid"], loc="upper left")
# plt.show()

# user_id = 26
# movies_watched_by_user = df[df.userId == user_id]

# Print  movies watched by user
# print(movies_watched_by_user)