import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neuralnet import Network, FCLayer, tanh, tanh_prime, mse, mse_prime, ActivationLayer

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     cleaning up the data     --{{{
#···············································································

from sqlite3 import connect
conn = connect('SpotifyData.db')
df = pd.read_sql(
    "select F.*, P.genre, T.title, T.artists, T.album, T.popularity, T.track_id from FEATURES F, PLAYLISTS P, TRACKS T where F.track_id = T.track_id and T.playlist_id = P.playlist_id",
    conn)
tid = list(df.columns).index('track_id')
cols = dict([(n, 'float') for n in list(df.columns)[:tid]])
cols.update(dict([(n, 'string') for n in list(df.columns)[tid:]]))
cols['popularity'] = 'float'
cols['duration_ms'] = 'int'
cols['time_signature'] = 'int'

df = df.astype(cols)
df.to_csv('spotifydata.csv')
df
#                                                                            }}}}}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     training and validation sets     --
#···············································································
genres = dict([(g, i) for i, g in enumerate(df['genre'].unique())])

df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
df['key'] /= 11.
df['tempo'] /= 250.

# we get all features from the song database
feats = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature','popularity']
df.columns
# X = df[df.columns[:list(df.columns).index('track_id')]].to_numpy()
X = df[feats].to_numpy() 
# X = dd.to_numpy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
 # and one hot encode the genres
cat = np.array([genres[g] for g in df['genre']])
Y = np.zeros([len(cat), len(genres)])
Y[np.arange(cat.size), cat] = 1

X[0]
X = X.reshape([X.shape[0], 1, X.shape[1]])
Y = Y.reshape([Y.shape[0], 1, Y.shape[1]])

# now we separate between training and validation sets
train_size = int(0.80 * len(Y))

X_train, X_valid = (X[:train_size], X[train_size:])
Y_train, Y_valid = (Y[:train_size], Y[train_size:])
vdf = df.iloc[train_size:]

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     network configuration     --
#···············································································


net = Network()
net.add(FCLayer(X.shape[2], 30))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(30, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 20))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(20, Y.shape[2]))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
X
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Training     --
#···············································································
net.fit(X_train, Y_train, epochs=1000, learning_rate=0.005)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     predictions and plotting     --
#···············································································


plt.plot(net.stats)  # error curve
plt.title('Loss during training')
plt.show()


out = np.array(net.predict(X_valid))
k = np.array(list(genres.keys()))
predicted = np.argmax(out, axis=2).flatten()
real = np.argmax(Y_valid, axis=2).flatten()
vdf['predicted_genre'] = k[predicted]

correct = 100. * vdf[vdf['genre'] == vdf['predicted_genre']
                     ].groupby('genre').size() / vdf.groupby('genre').size()
incorrect = 100. * vdf[vdf['genre'] != vdf['predicted_genre']
                       ].groupby('genre').size() / vdf.groupby('genre').size()
width = 0.8 
fig, ax = plt.subplots(figsize=(8,8))
ax.bar(k, correct, width, label='correct', color = "#08CCBC")
ax.bar(k, incorrect, width, bottom=correct, label='incorrect', color ="#FF8B78")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_ticks([])
ax.set_ylabel('Scores')
ax.set_title('Prediction of genres from song features')
ax.legend()

plt.show()


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                       --     Test Configs     --
#···············································································
#···············································································
genres = dict([(g, i) for i, g in enumerate(df['genre'].unique())])

df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
df['key'] /= 11.
df['tempo'] /= 250.

# we get all features from the song database
feats = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature','popularity']
df.columns
# X = df[df.columns[:list(df.columns).index('track_id')]].to_numpy()
X = df[feats].to_numpy() 
# X = dd.to_numpy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
 # and one hot encode the genres
cat = np.array([genres[g] for g in df['genre']])
Y = np.zeros([len(cat), len(genres)])
Y[np.arange(cat.size), cat] = 1

X[0]
X = X.reshape([X.shape[0], 1, X.shape[1]])
Y = Y.reshape([Y.shape[0], 1, Y.shape[1]])

# now we separate between training and validation sets
train_size = int(0.80 * len(Y))

X_train, X_valid = (X[:train_size], X[train_size:])
Y_train, Y_valid = (Y[:train_size], Y[train_size:])
vdf = df.iloc[train_size:]

#                                                                            }}}

# {{{                   --     network configuration     --
#···············································································


net = Network()
net.add(FCLayer(X.shape[2], 30))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(30, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 60))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(60, 20))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(20, Y.shape[2]))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

#                                                                            }}}

# {{{                         --     Training     --
#···············································································
net.fit(X_train, Y_train, epochs=10, learning_rate=0.005)
#                                                              
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


