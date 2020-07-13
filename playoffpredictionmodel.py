import math
import numpy as np

import pandas as pd

import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.inspection import permutation_importance

from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from models.sqla_utils import ENGINE
from models.sqla_utils import PLAYOFF_ENGINE

diff_train = '''select pg.home_team, pg.away_team, pg.year, 
rthome.win_pct - rtaway.win_pct as win_pct_diff,
rthome.exp_win_pct - rtaway.exp_win_pct as exp_win_pct_diff,
rthome.SB - rtaway.SB as SB_diff,
rthome.CS - rtaway.CS as CS_diff,
rthome.SF - rtaway.SF as SF_diff,
rthome.SH - rtaway.SH as SH_diff,
rthome.AVG - rtaway.AVG as AVG_diff, 
rthome.OPS - rtaway.OPS as OPS_diff, 
rthome.RAA - rtaway.RAA as RAA_diff, 
rthome.ERA - rtaway.ERA as ERA_diff, 
rthome.wRAA - rthome.wRAA as wRAA_diff,
sphome.FIP - spaway.FIP as FIP_diff,
rthome.RpFIP - rtaway.RpFIP as RpFIP_diff,
(rthome.RpIP/(rthome.SpIP + rthome.RpIP)) - (rtaway.RpIP/(rtaway.SpIP + rtaway.RpIP)) as relief_pct_diff, 
rthome.RpERA - rtaway.RpERA as RpERA_diff,
sphome.ERA - spaway.ERA as SpERA_diff, 
(pg.home_team = pg.winning_team) as home_team_won
from retrosheet.game pg
join retrosheet.team rthome
on rthome.year = pg.year and((rthome.team = pg.winning_team) or (rthome.team = pg.losing_team))
join retrosheet.team rtaway
on rtaway.year = pg.year and (rtaway.team = pg.winning_team or rtaway.team = pg.losing_team)
join retrosheet.player sphome
on sphome.player_id = pg.starting_pitcher_home and sphome.year = pg.year and sphome.team = pg.home_team
join retrosheet.player spaway
on spaway.player_id = pg.starting_pitcher_away and spaway.year = pg.year and spaway.team = pg.away_team
where rthome.team = pg.home_team and rtaway.team = pg.away_team;'''

diff_test = '''select pg.home_team, pg.away_team, pg.year, 
rthome.win_pct - rtaway.win_pct as win_pct_diff,
rthome.exp_win_pct - rtaway.exp_win_pct as exp_win_pct_diff,
rthome.SB - rtaway.SB as SB_diff,
rthome.CS - rtaway.CS as CS_diff,
rthome.SF - rtaway.SF as SF_diff,
rthome.SH - rtaway.SH as SH_diff,
rthome.AVG - rtaway.AVG as AVG_diff, 
rthome.OPS - rtaway.OPS as OPS_diff, 
rthome.RAA - rtaway.RAA as RAA_diff, 
rthome.ERA - rtaway.ERA as ERA_diff, 
rthome.wRAA - rthome.wRAA as wRAA_diff,
sphome.FIP - spaway.FIP as FIP_diff,
rthome.RpFIP - rtaway.RpFIP as RpFIP_diff,
(rthome.RpIP/(rthome.SpIP + rthome.RpIP)) - (rtaway.RpIP/(rtaway.SpIP + rtaway.RpIP)) as relief_pct_diff, 
rthome.RpERA - rtaway.RpERA as RpERA_diff,
sphome.ERA - spaway.ERA as SpERA_diff, 
(pg.home_team = pg.winning_team) as home_team_won
from playoffs.game pg
join retrosheet.team rthome
on rthome.year = pg.year and((rthome.team = pg.winning_team) or (rthome.team = pg.losing_team))
join retrosheet.team rtaway
on rtaway.year = pg.year and (rtaway.team = pg.winning_team or rtaway.team = pg.losing_team)
join retrosheet.player sphome
on sphome.player_id = pg.starting_pitcher_home and sphome.year = pg.year and sphome.team = pg.home_team
join retrosheet.player spaway
on spaway.player_id = pg.starting_pitcher_away and spaway.year = pg.year and spaway.team = pg.away_team
where rthome.team = pg.home_team and rtaway.team = pg.away_team;'''

train_query = '''
select pg.home_team, pg.away_team, pg.year, 
rthome.win_pct as home_win_pct,
rthome.exp_win_pct as home_exp_win_pct,
rthome.SB as home_SB,
rthome.CS as home_CS,
rthome.SF as home_SF,
rthome.SH as home_SH,
rthome.AVG as home_AVG, 
rthome.OPS as home_OPS, 
rthome.RAA as home_RAA, 
rthome.ERA as home_ERA, 
(rthome.RpIP/(rthome.SpIP + rthome.RpIP)) as home_relief_pct, 
rthome.RpERA as home_RpERA, 
rtaway.win_pct as away_win_pct,
rtaway.exp_win_pct as away_exp_win_pct,
rtaway.SB as away_SB,
rtaway.CS as away_CS,
rtaway.SF as away_SF,
rtaway.SH as away_SH,
rtaway.AVG as away_AVG, 
rtaway.OPS as away_OPS, 
rtaway.RAA as away_RAA, 
rtaway.ERA as away_ERA, 
(rtaway.RpIP/(rtaway.SpIP + rtaway.RpIP)) as away_relief_pct, 
rtaway.RpERA as away_RpERA, 
(pg.home_team = pg.winning_team) as home_team_won
from retrosheet.game pg
join retrosheet.team rthome
on rthome.year = pg.year and((rthome.team = pg.winning_team) or (rthome.team = pg.losing_team))
join retrosheet.team rtaway
on rtaway.year = pg.year and (rtaway.team = pg.winning_team or rtaway.team = pg.losing_team)
where rthome.team = pg.home_team and rtaway.team = pg.away_team;
'''


test_query = '''
select pg.home_team, pg.away_team, pg.year, 
rthome.win_pct as home_win_pct,
rthome.exp_win_pct as home_exp_win_pct,
rthome.SB as home_SB,
rthome.CS as home_CS,
rthome.SF as home_SF,
rthome.SH as home_SH,
rthome.AVG as home_AVG, 
rthome.OPS as home_OPS, 
rthome.RAA as home_RAA, 
rthome.ERA as home_ERA, 
(rthome.RpIP/(rthome.SpIP + rthome.RpIP)) as home_relief_pct, 
rthome.RpERA as home_RpERA, 
rtaway.win_pct as away_win_pct,
rtaway.exp_win_pct as away_exp_win_pct,
rtaway.SB as away_SB,
rtaway.CS as away_CS,
rtaway.SF as away_SF,
rtaway.SH as away_SH,
rtaway.AVG as away_AVG, 
rtaway.OPS as away_OPS, 
rtaway.RAA as away_RAA, 
rtaway.ERA as away_ERA, 
(rtaway.RpIP/(rtaway.SpIP + rtaway.RpIP)) as away_relief_pct, 
rtaway.RpERA as away_RpERA, 
(pg.home_team = pg.winning_team) as home_team_won
from playoffs.game pg
join retrosheet.team rthome
on rthome.year = pg.year and((rthome.team = pg.winning_team) or (rthome.team = pg.losing_team))
join retrosheet.team rtaway
on rtaway.year = pg.year and (rtaway.team = pg.winning_team or rtaway.team = pg.losing_team)
where rthome.team = pg.home_team and rtaway.team = pg.away_team;'''

# playoff_train_query = '''select * from game where year < 2010'''
# playoff_test_query = '''select * from game where year >= 2010'''

train_df = pd.read_sql_query(
    diff_train,
    con = ENGINE
)

test_df = pd.read_sql_query(
    diff_test,
    con=ENGINE
)

# playoff_train_data = pd.read_sql_query(
#     playoff_train_query,
#     con=PLAYOFF_ENGINE
# )

# playoff_test_data = pd.read_sql_query(
#     playoff_test_query,
#     con=PLAYOFF_ENGINE
# )

i = 0
def hinge_validation(y_true, y_pred):
    '''
    measure accuracy of hinge model
    @y_true - the expected data
    @y_pred - the theoretical data
    '''
    y_pred_pos = K.round(K.clip(y_pred, -1, 1))
    y_pos = K.round(K.clip(y_true, -1, 1))

    correct = K.equal(y_pos, y_pred_pos)
    return K.mean(correct)
    

batch_size = 200

def plot_win_pct():
    '''
    plot the predicted data vs some given data
    '''
    game_winner = 'Winning Team'
    game_loser = 'Losing Team'
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    home_team_won = test_df[test_df['home_team_won'] == 1]
    home_team_lost = test_df[test_df['home_team_won'] == 0]
    print(home_team_won)
    print(home_team_lost)
    ax = sns.distplot(home_team_won.win_pct_diff, 
    bins = 40, label=game_winner, ax = axes, kde = False)
    ax = sns.distplot(home_team_lost.win_pct_diff, 
    bins = 40, label=game_loser, ax = axes, kde = False)
    ax.legend()
    ax.set_title('Winning Percentage')
    # ax = sns.distplot(home_team_won.OPS_diff, 
    # bins = 40, label=game_winner, ax = axes[1], kde = False)
    # ax = sns.distplot(home_team_lost.OPS_diff, 
    # bins = 40, label=game_loser, ax = axes[1], kde = False)
    # ax.legend()
    # ax.set_title('OPS')
    # ax = sns.distplot(home_team_won.SpERA_diff, 
    # bins = 40, label=game_winner, ax = axes[2], kde = False)
    # ax = sns.distplot(home_team_lost.SpERA_diff, 
    # bins = 40, label=game_loser, ax = axes[2], kde = False)
    # ax.legend()
    # ax.set_title('Starting Pitcher ERA')
    # ax = sns.countplot(x='home_team_won', data=train_df, ax=axes[3])
    # ax.invert_xaxis()
    
    plt.show()

def permutation_importance_model(model, X_train, Y_train):
    '''
    print the features by importance for the model
    @param model - the model to measure
    @param X_train - the given data
    @param Y_train - the data to predict    
    '''
    print()
    print('Feature Importance')
    r = permutation_importance(model, X_train, Y_train,
                           n_repeats=50,
                           random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        print(f"{X_train.columns[i] :<8}  "
                f"{r.importances_mean[i] :.3f}  "
                f" +/- {r.importances_std[i]:.3f}")

def accuracy(model, X_train, Y_train):
    '''
    print the model accuracy
    @param model - the model to measure
    @param X_train - the given data
    @param Y_train - the data to predict
    '''
    print()
    print(type(model).__name__)
    print()
    accuracy = round(model.score(X_train, Y_train) * 100, 2)
    print('Accuracy', accuracy)
    return accuracy



def cross_validation(model, X_train, Y_train, Y_pred):
    '''
    print out cross validation and other metrics given a model
    @param model - the model to measure
    @param X_train - the given data
    @param Y_train - the data to predict
    @param Y_pred - the prediction the model made on the test data
    '''
    Y_test = test_df['home_team_won']
    # Y_test.replace(0, -1, inplace=True)
    print()
    predictions = cross_val_predict(model, X_train, Y_train, cv=3)
    print(confusion_matrix(Y_train, predictions))

    print("Precision:", precision_score(Y_train, predictions))
    print("Recall:",recall_score(Y_train, predictions))
    print()
    print('Cross validation')
    scores = cross_val_score(model, X_train, Y_train, cv=4, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    print()
    print('Test Data')
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        print('pred', pred, 'test', test)
        if pred == 1 and test == pred:
            tp += 1
        elif pred == 1 and test != pred:
            fp += 1
        elif pred == 0 and test != pred:
            fn += 1
        elif pred == 0 and test == pred:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)

def build_model(model, X_train, Y_train, X_test):
    '''
    Taking a model and data train the model and predict test data before printing a series of metrics
    @param model - the model to train
    @param X_train - the given data
    @param Y_train - the data to predict
    @param X_test - the given data for the test
    '''
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy(model, X_train, Y_train)
    cross_validation(model, X_train, Y_train, Y_pred)
    permutation_importance_model(model, X_train, Y_train)

def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))

def main():
    # train_df.info()
    #plot_win_pct()
    X_train = train_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('CS_diff', axis=1).drop('RAA_diff', axis=1).drop('ERA_diff', axis=1).drop('SF_diff', axis=1).drop('SB_diff', axis=1).drop('SH_diff', axis=1).drop('OPS_diff', axis=1).drop('RpERA_diff', axis=1).drop('SpERA_diff', axis=1)
    #X_train = train_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1)
    Y_train = train_df['home_team_won']
    X_test = test_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('CS_diff', axis=1).drop('RAA_diff', axis=1).drop('ERA_diff', axis=1).drop('SF_diff', axis=1).drop('SB_diff', axis=1).drop('SH_diff', axis=1).drop('OPS_diff', axis=1).drop('RpERA_diff', axis=1).drop('SpERA_diff', axis=1)
    #X_test = test_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1)
    scaler = preprocessing.StandardScaler().fit(X_train)
    columns = X_train.columns
    X_train = pd.DataFrame(scaler.transform(X_train), columns=columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=columns)
    Y_test = test_df['home_team_won']

    Y_train.replace(0, -1, inplace=True)
    Y_test.replace(0, -1, inplace=True)

    steps_per_epoch = calculate_spe(X_train.size)

    # Stochastic Gradient Descent
    build_model(linear_model.SGDClassifier(max_iter=1000, tol=None), X_train, Y_train, X_test)

    # Random Forest
    build_model(RandomForestClassifier(n_estimators=10), X_train, Y_train, X_test)
  
    # Logistic Regression
    build_model(LogisticRegression(max_iter=10000), X_train, Y_train, X_test)

    # KNN 
    # build_model(KNeighborsClassifier(n_neighbors = 3), X_train, Y_train, X_test)

    # Gaussian
    gaussian = GaussianNB()
    build_model(gaussian, X_train, Y_train, X_test)
    cross_val_score(gaussian, X_train, Y_train, cv=5, scoring='accuracy')

    # Perceptron
    build_model(Perceptron(max_iter=10000), X_train, Y_train, X_test)

    # Decision Tree
    build_model(DecisionTreeClassifier(), X_train, Y_train, X_test)

    # Hinge
    model = Sequential()
    model.add(Dense(50, input_dim=7, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='tanh'))
    opt = optimizers.SGD(lr = 0.01, momentum= 0.9)
    model.compile(loss='hinge', optimizer = 'Nadam', metrics=['accuracy', 'binary_accuracy', hinge_validation])


    history = model.fit(X_train, Y_train, epochs=200, verbose=0, validation_split=0.2)
    Y_pred = model.predict(X_test)
    cross_validation(model, X_train, Y_train, Y_pred)

    # plot hinge accuracy
    plt.plot(history.history['hinge_validation'])
    plt.plot(history.history['val_hinge_validation'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.show()


main()
#plot_win_pct()
