import pandas as pd
import numpy as np





## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                                 --  Imports        --
#···············································································
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     cleaning up the data     --
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
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{               --     training and validation sets     --
#···············································································
genres = dict([(g, i) for i, g in enumerate(df['genre'].unique())])

df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
df['key'] /= 11.
df['tempo'] /= 250.

float_cols = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'mode', 'duration_ms', 'time_signature']
for f in float_cols:
    df[f] = df[f].astype('float')
df = df.drop(['track_id', 'artists', 'album', 'title'], axis=1)

df['genre'] = df['genre'].astype('category')
df['genre'] = df['genre'].cat.codes
filtered_df = df.drop(['popularity', 'key', 'mode', 'liveness', 'duration_ms', 'time_signature'], 1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# title_artist_album_df = df[['title', 'artists', 'album']]
X = df.drop(['genre'], axis=1)
scaler.fit_transform(X)
y = df['genre']

X = filtered_df.drop(['genre'], axis=1)
y = filtered_df['genre']

 # X = X.reshape([X.shape[0], 1, X.shape[1]])
 # y = y.reshape([y.shape[0], 1, y.shape[1]])

# now we separate between training and validation sets
# train_size = int(0.80 * len(Y))

# X_train, X_valid = (X[:train_size], X[train_size:])
# y_train, y_test = (y[:train_size], y[train_size:])
# vdf = df.iloc[train_size:]

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────d
X.describe()
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     train_test_split     --
#···············································································
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) 
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
X.shape
df['genre']
X.columns
y
y = y.drop(y.tail(1).index)

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                             --          --
#···············································································

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}
from sklearn.model_selection import cross_val_score




for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(1, 50, 5)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# # Support Vector Classifier
# svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
# grid_svc = GridSearchCV(SVC(), svc_params)
# grid_svc.fit(X_train, y_train)

# # SVC best estimator
# svc = grid_svc.best_estimator_

# # DecisionTree Classifier
# tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              # "min_samples_leaf": list(range(5,7,1))}
# grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
# grid_tree.fit(X_train, y_train)

# # tree best estimator
# tree_clf = grid_tree.best_estimator_
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

X
