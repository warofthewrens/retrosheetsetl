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
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from models.sqla_utils import ENGINE
from models.sqla_utils import PLAYOFF_ENGINE


ranking_test_query = '''
select pg.home_team, pg.away_team, pg.year, pg.date,
away.Catcher_Rank - home.Catcher_Rank as catcher_diff,
away.First_Baseman_Rank - home.First_Baseman_Rank as first_diff,
away.Second_Baseman_Rank - home.Second_Baseman_Rank as second_diff,
away.Third_Baseman_Rank - home. Third_Baseman_Rank as third_diff,
away.Shortstop_Rank - home.Shortstop_Rank as short_diff,
away.Left_Field_Rank - home.Left_Field_Rank as left_diff,
away.Center_Field_Rank - home.Center_Field_Rank as center_diff,
away.Right_Field_Rank - home.Right_Field_Rank as right_diff,
reliefhome.closer_WAR - reliefaway.closer_WAR as closer_diff,
(reliefhome.relief_1_WAR + reliefhome.relief_2_WAR + reliefhome.relief_3_WAR - reliefhome.relief_4_WAR) - 
(reliefaway.relief_1_WAR + reliefaway.relief_2_WAR + reliefaway.relief_3_WAR - reliefaway.relief_4_WAR) as bullpen_diff,
sphome.WAR - spaway.WAR as spWAR_diff,
(pg.home_team = pg.winning_team) as home_team_won
from 
playoffs.Game pg
join retrosheet.TeamPositionRank home
on home.year = pg.year and((home.team = pg.winning_team) or (home.team = pg.losing_team))
join retrosheet.TeamPositionRank away
on away.year = pg.year and (away.team = pg.winning_team or away.team = pg.losing_team)
join retrosheet.ReliefPosition reliefhome
on reliefhome.year = pg.year and (reliefhome.team = pg.winning_team or reliefhome.team = pg.losing_team)
join retrosheet.ReliefPosition reliefaway
on reliefaway.year = pg.year and (reliefaway.team = pg.winning_team or reliefaway.team = pg.losing_team)
join retrosheet.Player sphome
on sphome.player_id = pg.starting_pitcher_home and sphome.year = pg.year and sphome.team = pg.home_team
join retrosheet.Player spaway
on spaway.player_id = pg.starting_pitcher_away and spaway.year = pg.year and spaway.team = pg.away_team
where home.team = pg.home_team and away.team = pg.away_team and reliefhome.team = pg.home_team and reliefaway.team = pg.away_team and pg.year >= 2015
;
'''

ranking_playoff_train_query = '''
select pg.home_team, pg.away_team, pg.year, pg.date,
away.Catcher_Rank - home.Catcher_Rank as catcher_diff,
away.First_Baseman_Rank - home.First_Baseman_Rank as first_diff,
away.Second_Baseman_Rank - home.Second_Baseman_Rank as second_diff,
away.Third_Baseman_Rank - home. Third_Baseman_Rank as third_diff,
away.Shortstop_Rank - home.Shortstop_Rank as short_diff,
away.Left_Field_Rank - home.Left_Field_Rank as left_diff,
away.Center_Field_Rank - home.Center_Field_Rank as center_diff,
away.Right_Field_Rank - home.Right_Field_Rank as right_diff,
reliefhome.closer_WAR - reliefaway.closer_WAR as closer_diff,
(reliefhome.relief_1_WAR + reliefhome.relief_2_WAR + reliefhome.relief_3_WAR - reliefhome.relief_4_WAR) - 
(reliefaway.relief_1_WAR + reliefaway.relief_2_WAR + reliefaway.relief_3_WAR - reliefaway.relief_4_WAR) as bullpen_diff,
sphome.WAR - spaway.WAR as spWAR_diff,
(pg.home_team = pg.winning_team) as home_team_won
from 
playoffs.Game pg
join retrosheet.TeamPositionRank home
on home.year = pg.year and((home.team = pg.winning_team) or (home.team = pg.losing_team))
join retrosheet.TeamPositionRank away
on away.year = pg.year and (away.team = pg.winning_team or away.team = pg.losing_team)
join retrosheet.ReliefPosition reliefhome
on reliefhome.year = pg.year and (reliefhome.team = pg.winning_team or reliefhome.team = pg.losing_team)
join retrosheet.ReliefPosition reliefaway
on reliefaway.year = pg.year and (reliefaway.team = pg.winning_team or reliefaway.team = pg.losing_team)
join retrosheet.Player sphome
on sphome.player_id = pg.starting_pitcher_home and sphome.year = pg.year and sphome.team = pg.home_team
join retrosheet.Player spaway
on spaway.player_id = pg.starting_pitcher_away and spaway.year = pg.year and spaway.team = pg.away_team
where home.team = pg.home_team and away.team = pg.away_team and reliefhome.team = pg.home_team and reliefaway.team = pg.away_team and pg.year < 2015
;
'''

ranking_train_query = '''
select pg.home_team, pg.away_team, pg.year, pg.date,
away.Catcher_Rank - home.Catcher_Rank as catcher_diff,
away.First_Baseman_Rank - home.First_Baseman_Rank as first_diff,
away.Second_Baseman_Rank - home.Second_Baseman_Rank as second_diff,
away.Third_Baseman_Rank - home. Third_Baseman_Rank as third_diff,
away.Shortstop_Rank - home.Shortstop_Rank as short_diff,
away.Left_Field_Rank - home.Left_Field_Rank as left_diff,
away.Center_Field_Rank - home.Center_Field_Rank as center_diff,
away.Right_Field_Rank - home.Right_Field_Rank as right_diff,
reliefhome.closer_WAR - reliefaway.closer_WAR as closer_diff,
(reliefhome.relief_1_WAR + reliefhome.relief_2_WAR + reliefhome.relief_3_WAR - reliefhome.relief_4_WAR) - 
(reliefaway.relief_1_WAR + reliefaway.relief_2_WAR + reliefaway.relief_3_WAR - reliefaway.relief_4_WAR) as bullpen_diff,
sphome.WAR - spaway.WAR as spWAR_diff,
(pg.home_team = pg.winning_team) as home_team_won
from 
retrosheet.Game pg
join retrosheet.TeamPositionRank home
on home.year = pg.year and((home.team = pg.winning_team) or (home.team = pg.losing_team))
join retrosheet.TeamPositionRank away
on away.year = pg.year and (away.team = pg.winning_team or away.team = pg.losing_team)
join retrosheet.ReliefPosition reliefhome
on reliefhome.year = pg.year and (reliefhome.team = pg.winning_team or reliefhome.team = pg.losing_team)
join retrosheet.ReliefPosition reliefaway
on reliefaway.year = pg.year and (reliefaway.team = pg.winning_team or reliefaway.team = pg.losing_team)
join retrosheet.Player sphome
on sphome.player_id = pg.starting_pitcher_home and sphome.year = pg.year and sphome.team = pg.home_team
join retrosheet.Player spaway
on spaway.player_id = pg.starting_pitcher_away and spaway.year = pg.year and spaway.team = pg.away_team
where home.team = pg.home_team and away.team = pg.away_team and reliefhome.team = pg.home_team and reliefaway.team = pg.away_team
;
'''

train_df = pd.read_sql_query(
    ranking_train_query,
    con = ENGINE
)

playoff_train_df = pd.read_sql_query(
    ranking_playoff_train_query,
    con=ENGINE
)

test_df = pd.read_sql_query(
    ranking_test_query,
    con=ENGINE
)

param_distributions = {}
loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
penalty = ['l1', 'l2', 'elasticnet']
penalty_p = ['l1', 'l2']
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
eta0 = [1, 10, 100]


n_estimators = [1, 10, 50, 100, 200]
criterion = ['gini', 'entropy']
min_samples_split = [2, 3, 4]
min_samples_leaf = [1, 2, 3]
max_features = ['auto', 'sqrt', 'log2']
class_weight_forest = ['balanced', 'balanced_subsample']
ccp_alpha = [0, .0001, .001, .01, 1]

tol = [None, 0.0001, 0.001, 0.01]
C = [0.1, 1.0, 10, 100]
solver = ['newton-cg', 'liblinear', 'sag', 'saga']
multi_class = ['auto', 'ovr', 'multinomial']

splitter = ['best', 'random']

param_distributions[type(SGDClassifier()).__name__] = dict(loss=loss,
                        penalty=penalty,
                        alpha=alpha,
                        learning_rate=learning_rate,
                        class_weight=class_weight,
                        eta0=eta0)

param_distributions[type(RandomForestClassifier()).__name__] = dict(
    n_estimators = n_estimators,
    criterion = criterion,
    min_samples_split = min_samples_split,
    max_features = max_features,
    class_weight = class_weight_forest,
    ccp_alpha = ccp_alpha
)

param_distributions[type(LogisticRegression()).__name__] = dict(
    penalty = penalty,
    tol = tol,
    C = C,
    class_weight = class_weight,
    solver = solver,
    multi_class = multi_class
)

param_distributions[type(Perceptron()).__name__] = dict(
    penalty=penalty_p,
    alpha=alpha,
    class_weight=class_weight,
    eta0=eta0
)

param_distributions[type(DecisionTreeClassifier()).__name__] = dict(
    criterion = criterion,
    splitter = splitter,
    min_samples_split = min_samples_split,
    max_features = max_features,
    class_weight = class_weight,
    ccp_alpha = ccp_alpha
)

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

def continuous_validation(model, X_train, Y_train, Y_pred):
    Y_test = test_df['home_team_won']
    # Y_test.replace(0, -1, inplace=True)
    print()
    predictions = cross_val_predict(model, X_train, Y_train, cv=3)
    # print(confusion_matrix(Y_train, predictions))

    # print("Precision:", precision_score(Y_train, predictions))
    # print("Recall:",recall_score(Y_train, predictions))
    print()
    print('Cross validation')
    # scores = cross_val_score(model, X_train, Y_train, cv=4, scoring = "accuracy")
    # print("Scores:", scores)
    # print("Mean:", scores.mean())
    # print("Standard Deviation:", scores.std())
    print()
    print('Test Data')
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        if test == 1 and pred >= 0.5:
            tp += 1
        elif test == 1 and pred < 0.5:
            fn += 1
        elif test == 0 and pred >= 0.5:
            fp += 1
        elif test == 0 and pred < 0.5:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)

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

def plot_learning_curve(model, X_train, Y_train):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model, 
                                                            X_train, 
                                                            Y_train,
                                                            # Number of folds in cross-validation
                                                            cv=4,
                                                            # Evaluation metric
                                                            scoring='accuracy',
                                                            # Use all computer cores
                                                            n_jobs=-1, 
                                                            # 50 different sizes of the training set
                                                            train_sizes=np.linspace(0.01, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title(type(model).__name__ + " Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def build_model(model, X_train, Y_train, X_test, hyperparam=True):
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
    # if not continuous:
    cross_validation(model, X_train, Y_train, Y_pred)
    # else:
    #     continuous_validation(model, X_train, Y_train, Y_pred)
    permutation_importance_model(model, X_train, Y_train)
    #plot_learning_curve(model, X_train, Y_train)
    if hyperparam:
        hyper_parameters(model, X_train, Y_train)

def calculate_spe(y):
  return int(math.ceil((1. * y) / batch_size))

def hyper_parameters(model, X_train, Y_train):


    random = RandomizedSearchCV(estimator=model,
                                param_distributions=param_distributions[type(model).__name__],
                                scoring='accuracy',
                                verbose=1, n_jobs=-1,
                                n_iter=100)
    print(random.estimator.get_params().keys())
    random_result = random.fit(X_train, Y_train)

    print('Best Score: ', random_result.best_score_)
    print('Best Params: ', random_result.best_params_)

def main():
    X_train = train_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('date', axis=1)
    Y_train = train_df['home_team_won']

    X_playoff_train = playoff_train_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('date', axis=1)
    Y_playoff_train = playoff_train_df['home_team_won']

    X_test = test_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('date', axis=1)

    scaler = preprocessing.StandardScaler().fit(X_train)
    columns = X_train.columns
    X_train = pd.DataFrame(scaler.transform(X_train), columns=columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=columns)
    Y_test = test_df['home_team_won']

    sgd = build_model(linear_model.SGDClassifier(max_iter=1000, tol=None), X_train, Y_train, X_test, True)

    print('TRAINED on PLAYOFFS')
    playoff_sgd = build_model(linear_model.SGDClassifier(max_iter=1000, tol=None), X_playoff_train, Y_playoff_train, X_test, True)
    # Random Forest
    random_forest = build_model(RandomForestClassifier(n_estimators=50, min_samples_split = 4, max_features='log2', criterion='entropy', 
                                class_weight='balanced', ccp_alpha=.001), X_train, Y_train, X_test, True)
    
    playoff_random_forest = build_model(RandomForestClassifier(n_estimators=50, min_samples_split = 4, max_features='log2', criterion='entropy', 
                                class_weight='balanced', ccp_alpha=.001), X_playoff_train, Y_playoff_train, X_test, True)
  
    # Logistic Regression
    log = build_model(LogisticRegression(max_iter=10000, tol=0.001, solver='liblinear', penalty='l1', multi_class='ovr', 
                                        class_weight={1:0.5, 0:0.5}, C=0.1), X_train, Y_train, X_test, True)
    
    playoff_log = build_model(LogisticRegression(max_iter=10000, tol=0.001, solver='liblinear', penalty='l1', multi_class='ovr', 
                                        class_weight={1:0.5, 0:0.5}, C=0.1), X_playoff_train, Y_playoff_train, X_test, True)


    # KNN 
    # build_model(KNeighborsClassifier(n_neighbors = 3), X_train, Y_train, X_test)

    # Gaussian
    gaussian = GaussianNB()
    build_model(gaussian, X_train, Y_train, X_test, True)
    cross_val_score(gaussian, X_train, Y_train, cv=5, scoring='accuracy')

    playoff_gaussian = GaussianNB()
    build_model(playoff_gaussian, X_playoff_train, Y_playoff_train, X_test, True)
    cross_val_score(playoff_gaussian, X_train, Y_train, cv=5, scoring='accuracy')


    # Perceptron
    perceptron = build_model(Perceptron(max_iter=10000, penalty='l2', eta0=10, class_weight={1: 0.6, 0: 0.4}, alpha=0.0001), X_train, Y_train, X_test)

    playoff_perceptron = build_model(Perceptron(max_iter=10000, penalty='l2', eta0=10, class_weight={1: 0.6, 0: 0.4}, alpha=0.0001), X_playoff_train, Y_playoff_train, X_test)

    # Decision Tree
    d_tree = build_model(DecisionTreeClassifier(splitter='best', min_samples_split=4, max_features='log2', criterion='entropy', class_weight={1:0.5, 0:0.5}, ccp_alpha=0.0001), X_train, Y_train, X_test)
    playoff_d_tree = build_model(DecisionTreeClassifier(splitter='best', min_samples_split=4, max_features='log2', criterion='entropy', class_weight={1:0.5, 0:0.5}, ccp_alpha=0.0001), X_playoff_train, Y_playoff_train, X_test)

main()