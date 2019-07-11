import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout
from keras.models import Model

import warnings
warnings.filterwarnings('ignore')

def recommender_for_user(user_id, interact_matrix, df_content, topn = 10):
    
    pred_scores = interact_matrix.loc[user_id].values

    df_scores   = pd.DataFrame({'content_id': list(users_items_matrix_df.columns), 
                               'score': pred_scores})

    df_rec      = df_scores.set_index('content_id')\
                    .join(df_content.set_index('content_id'))\
                    .sort_values('score', ascending=False)\
                    .head(topn)[['score', 'game']]
    
    return df_rec[df_rec.score > 0]
def autoEncoder(X):
   
    # Entrada
    input_layer = Input(shape=(X.shape[1],), name='UserScore')
    
    # Codificando
    # -----------------------------
    enc = Dense(512, activation='selu', name='EncLayer1')(input_layer)

    # Latent Space
    # -----------------------------
    lat_space = Dense(256, activation='selu', name='LatentSpace')(enc)
    lat_space = Dropout(0.8, name='Dropout')(lat_space) # Dropout

    # Decodificando
    # -----------------------------
    dec = Dense(512, activation='selu', name='DecLayer1')(lat_space)

    # Salida
    output_layer = Dense(X.shape[1], activation='linear', name='UserScorePred')(dec)

    # modelo
    model = Model(input_layer, output_layer)    
    
    return model


df = pd.read_csv('../data/raw/rating.csv')
print(df.head(5))
print("---------------------------:")
df = pd.read_csv('../data/interactions_train_df.csv')
df = df[['user_id', 'content_id', 'game', 'view']]
print(df.head(5))
print("---------------------------:")

df_game = pd.read_csv('../data/articles_df.csv')
print(df_game.head(4))

users_items_matrix_df = df.pivot(index   = 'user_id', 
                                 columns = 'content_id', 
                                 values  = 'view').fillna(0)
print(users_items_matrix_df.head(10))
print(users_items_matrix_df.shape)
print(users_items_matrix_df.values.mean()*100)

X = users_items_matrix_df.values
y = users_items_matrix_df.values

model = autoEncoder(X)
model.compile(optimizer = Adam(lr=0.0001), loss='mse')
model.summary()

hist = model.fit(x=X, y=y,
                  epochs=50,
                  batch_size=64,
                  shuffle=True,
                  validation_split=0.1)

new_matrix = model.predict(X) * (X == 0)
new_users_items_matrix_df  = pd.DataFrame(new_matrix, 
                                          columns = users_items_matrix_df.columns, 
                                          index   = users_items_matrix_df.index)
print(new_users_items_matrix_df.head())

print(new_users_items_matrix_df.values.min(), new_users_items_matrix_df.values.max())

#VIDEO JUEGOS ANTERIORES
print(recommender_for_user(
    user_id         = 1011, 
    interact_matrix = users_items_matrix_df, 
    df_content      = df_game))
#RECOMENDACIONES
print(recommender_for_user(
    user_id         = 1011, 
    interact_matrix = new_users_items_matrix_df, 
    df_content      = df_game))