

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Imports      --{{{
#···············································································
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import sqlite3
import seaborn as sns
#                                                                            }}}}}}
## ─────────────────────────────────────────────────────────────────────────────




## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     DF Setup     --{{{
#···············································································
cx = sqlite3.connect('SpotifyData.db')
query = open('main_df.sql', 'r')
main_df = pd.read_sql(query.read(), cx)
main_df
float_cols = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'mode', 'duration_ms', 'time_signature']
for f in float_cols:
    main_df[f] = main_df[f].astype('float')
    
filtered_df = main_df.drop(['popularity', 'key', 'mode', 'liveness', 'duration_ms', 'time_signature', 'track_id', 'title', 'artists', 'album'], 1)
col_genre = main_df['genre']
title_artist_album_df = main_df[['title', 'artists', 'album']]

main_df = main_df.drop_duplicates()
Y = main_df['genre']


main_data = main_df.drop(['track_id', 'title', 'artists', 'album', 'instrumentalness'], 1) 
main_data['genre'] = main_data['genre'].astype('category')
main_data['genre'] = main_data['genre'].cat.codes


filtered_df['genre'] = filtered_df['genre'].astype('category')
filtered_df['genre'] = filtered_df['genre'].cat.codes

filtered_df = filtered_df.drop_duplicates()
#                                                                            }}}}}}
## ─────────────────────────────────────────────────────────────────────────────
print("TEST")


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                             --      Naive Bayes - Gaussian     --
#···············································································
def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y


def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]  

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            count = 0
            #print("LABEL: ", labels[j])
            for i in range(len(features)):
                i#print("FEATURE: ", features[i])
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])
                count += 1

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --     Scaling ?      --
#···············································································
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train,test = train_test_split(filtered_df, test_size=0.33)

testfit = scaler.fit(train)

train = testfit.transform(train)
test = testfit.transform(test)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                                              --     Running Model     --
#···············································································
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

train, test = train_test_split(main_data, test_size=.33)

X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
X_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

print("=========TRAINING PREDS=========================")
Y_pred = naive_bayes_gaussian(train, X=X_train, Y="genre")
print(confusion_matrix(y_train, Y_pred))
print(f1_score(y_train, Y_pred, average='macro'))

print("=========TEST PREDS=============================")
Y_test_pred = naive_bayes_gaussian(test, X=X_test, Y="genre")
print(confusion_matrix(y_test, Y_test_pred))
print(f1_score(y_test, Y_test_pred, average='macro'))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     Naive Bayes - Categorical     --
#···············································································
# making df categorical
c = filtered_df

for col in c.columns[:-1]:
    c[col] = pd.cut(c[col].values, bins = 10, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


c['genre'] = filtered_df['genre'] # putting back genre

def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y


def naive_bayes_categorical(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                                              --     Running Model - Categorical     --
#···············································································
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

train, test = train_test_split(main_data, test_size=.33)

X_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values
X_test = test.iloc[:,:-1].values
y_test = test.iloc[:,-1].values

print("=========TRAINING PREDS=========================")
Y_pred = naive_bayes_categorical(train, X=X_train, Y="genre")
print(confusion_matrix(y_train, Y_pred))
print(f1_score(y_train, Y_pred, average='macro'))

print("=========TEST PREDS=============================")
Y_test_pred = naive_bayes_categorical(test, X=X_test, Y="genre")
print(confusion_matrix(y_test, Y_test_pred))
print(f1_score(y_test, Y_test_pred, average='macro'))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Multiclass Logistic Regression     --
#···············································································
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
onehot_encoder = OneHotEncoder(sparse=False)
from sklearn.datasets import load_iris

def loss(X, Y, W):
    """
    Y: onehot encoded
    """
    Z = - X @ W
    N = X.shape[0]
    loss = 1/N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss

def gradient(X, Y, W, mu):
    """
    Y: onehot encoded 
    """
    Z = - X @ W
    P = softmax(Z, axis=1)
    N = X.shape[0]
    gd = 1/N * (X.T @ (Y - P)) + 2 * mu * W
    return gd

def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
    """
    Very basic gradient descent algorithm with fixed eta and mu
    """
    Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []
 
    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W

class Multiclass:
    def fit(self, X, Y):
        self.loss_steps, self.W = gradient_descent(X, Y)

    def loss_plot(self):
        return self.loss_steps.plot(
            x='step', 
            y='loss',
            xlabel='step',
            ylabel='loss'
        )

    def predict(self, H):
        Z = - H @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{          --     Logistic Regression Scaling (optional)     --
#···············································································
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = filtered_df.drop(['genre'], axis=1)
X = scaler.fit_transform(X)
y = filtered_df['genre']
X_train, X_test, y_train,y_test = train_test_split(X, y,  test_size=0.3)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                --     Running Logistic Regression     --
#···············································································
model = Multiclass()
model.fit(X_train, y_train.values)
model.loss_plot()

train_preds = model.predict(X_train)

print(confusion_matrix(y_train, train_preds))
print(f1_score(y_train, Y_pred, average='macro'))
(model.predict(X) == y).value_counts()
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



