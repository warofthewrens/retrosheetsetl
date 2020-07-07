import numpy as np

import pandas as pd

import seaborn as sns
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
where pg.year < 2015 and rthome.team = pg.home_team and rtaway.team = pg.away_team;'''

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
where pg.year >= 2015 and rthome.team = pg.home_team and rtaway.team = pg.away_team;'''

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
from playoffs.game pg
join retrosheet.team rthome
on rthome.year = pg.year and((rthome.team = pg.winning_team) or (rthome.team = pg.losing_team))
join retrosheet.team rtaway
on rtaway.year = pg.year and (rtaway.team = pg.winning_team or rtaway.team = pg.losing_team)
where pg.year < 2015 and rthome.team = pg.home_team and rtaway.team = pg.away_team;
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
where pg.year >= 2015 and rthome.team = pg.home_team and rtaway.team = pg.away_team;'''

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

def plot_win_pct():
    game_winner = 'Winning Team'
    game_loser = 'Losing Team'
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,10))
    home_team_won = train_df[train_df['home_team_won'] == 1]
    home_team_lost = train_df[train_df['home_team_won'] == 0]
    print(home_team_won)
    print(home_team_lost)
    ax = sns.distplot(home_team_won.win_pct_diff, 
    bins = 40, label=game_winner, ax = axes[0], kde = False)
    ax = sns.distplot(home_team_lost.win_pct_diff, 
    bins = 40, label=game_loser, ax = axes[0], kde = False)
    ax.legend()
    ax.set_title('Winning Percentage')
    ax = sns.distplot(home_team_won.OPS_diff, 
    bins = 40, label=game_winner, ax = axes[1], kde = False)
    ax = sns.distplot(home_team_lost.OPS_diff, 
    bins = 40, label=game_loser, ax = axes[1], kde = False)
    ax.legend()
    ax.set_title('Expected Winning Percentage')
    ax = sns.distplot(home_team_won.SpERA_diff, 
    bins = 40, label=game_winner, ax = axes[2], kde = False)
    ax = sns.distplot(home_team_lost.SpERA_diff, 
    bins = 40, label=game_loser, ax = axes[2], kde = False)
    ax.legend()
    ax.set_title('OPS')
    ax = sns.countplot(x='home_team_won', data=train_df, ax=axes[3])
    ax.invert_xaxis()
    
    plt.show()

def main():
    # train_df.info()
    plot_win_pct()
    X_train = train_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('CS_diff', axis=1).drop('RAA_diff', axis=1).drop('ERA_diff', axis=1).drop('SF_diff', axis=1).drop('SB_diff', axis=1).drop('SH_diff', axis=1)
    Y_train = train_df['home_team_won']
    X_test = test_df.drop('home_team', axis=1).drop('away_team', axis=1).drop('year', axis=1).drop('home_team_won', axis=1).drop('CS_diff', axis=1).drop('RAA_diff', axis=1).drop('ERA_diff', axis=1).drop('SF_diff', axis=1).drop('SB_diff', axis=1).drop('SH_diff', axis=1)
    Y_test = test_df['home_team_won']
    scaler = preprocessing.StandardScaler().fit(X_train)
    print(scaler.mean_)
    print(scaler.scale_)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Stochastic Gradient Descent
    sgd = linear_model.SGDClassifier(max_iter=1000, tol=None)
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    sgd.score(X_train, Y_train)

    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    print(acc_sgd)
    predictions = cross_val_predict(sgd, X_train, Y_train, cv=3)
    print(confusion_matrix(Y_train, predictions))

    print("Precision:", precision_score(Y_train, predictions))
    print("Recall:",recall_score(Y_train, predictions))
    scores = cross_val_score(sgd, X_train, Y_train, cv=4, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        if pred == 1 and pred == test:
            tp += 1
        elif pred == 1 and pred != test:
            fp += 1
        elif pred == 0 and pred != test:
            fn += 1
        elif pred == 0 and pred == test:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100, oob_score=True)
    random_forest.fit(X_train, Y_train)

    Y_pred = random_forest.predict(X_test)
    Y_test = Y_test.to_numpy()
    pred_win = Y_pred == 1
    test_win = Y_test == 1
    wins = np.where(pred_win & test_win)
    sum = (Y_pred * Y_test)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        if pred == 1 and pred == test:
            tp += 1
        elif pred == 1 and pred != test:
            fp += 1
        elif pred == 0 and pred != test:
            fn += 1
        elif pred == 0 and pred == test:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

    # importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
    # importances = importances.sort_values('importance',ascending=False).set_index('feature')

    print("oob score:", round(random_forest.oob_score, 4)*100, "%")
    print(acc_random_forest)
    # Logistic Regression
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X_train, Y_train)

    Y_pred = logreg.predict(X_test)

    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_log)

    predictions = cross_val_predict(logreg, X_train, Y_train, cv=3)

    print(confusion_matrix(Y_train, predictions))

    print("Precision:", precision_score(Y_train, predictions))
    print("Recall:",recall_score(Y_train, predictions))
    scores = cross_val_score(logreg, X_train, Y_train, cv=4, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        if pred == 1 and pred == test:
            tp += 1
        elif pred == 1 and pred != test:
            fp += 1
        elif pred == 0 and pred != test:
            fn += 1
        elif pred == 0 and pred == test:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)
    #KNN 
    knn = KNeighborsClassifier(n_neighbors = 3) 
    knn.fit(X_train, Y_train)  
    Y_pred = knn.predict(X_test)  
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    print()
    print('KNN', acc_knn)

    predictions = cross_val_predict(knn, X_train, Y_train, cv=10)

    print(confusion_matrix(Y_train, predictions))

    print("Precision:", precision_score(Y_train, predictions))
    print("Recall:",recall_score(Y_train, predictions))
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        if pred == 1 and pred == test:
            tp += 1
        elif pred == 1 and pred != test:
            fp += 1
        elif pred == 0 and pred != test:
            fn += 1
        elif pred == 0 and pred == test:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)

    # Gaussian
    gaussian = GaussianNB() 
    gaussian.fit(X_train, Y_train)  
    Y_pred = gaussian.predict(X_test)  
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    print()
    print('gaussian', acc_gaussian)

    predictions = cross_val_predict(gaussian, X_train, Y_train, cv=4)

    print(confusion_matrix(Y_train, predictions))

    print("Precision:", precision_score(Y_train, predictions))
    print("Recall:",recall_score(Y_train, predictions))
    scores = cross_val_score(gaussian, X_train, Y_train, cv=4, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(Y_pred)):
        pred = Y_pred[i]
        test = Y_test[i]
        if pred == 1 and pred == test:
            tp += 1
        elif pred == 1 and pred != test:
            fp += 1
        elif pred == 0 and pred != test:
            fn += 1
        elif pred == 0 and pred == test:
            tn += 1
    print('correct', tp + tn)
    print('incorrect', fp + fn)
    print('true positives', tp)
    print('true negatives', tn)
    print('false positives', fp)
    print('false negatives', fn)

    #Perceptron
    perceptron = Perceptron(max_iter=10000)
    perceptron.fit(X_train, Y_train)

    Y_pred = perceptron.predict(X_test)

    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    print('Perceptron', acc_perceptron)

    # #Linear Support Vector machine
    # linear_svc = LinearSVC(max_iter=10000)
    # linear_svc.fit(X_train, Y_train)

    # Y_pred = linear_svc.predict(X_test)

    # acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    # print()
    # print('Linear Support Vector Machine', acc_linear_svc)

    # predictions = cross_val_predict(linear_svc, X_train, Y_train, cv=3)

    # print(confusion_matrix(Y_train, predictions))

    # print("Precision:", precision_score(Y_train, predictions))
    # print("Recall:",recall_score(Y_train, predictions))
    # scores = cross_val_score(linear_svc, X_train, Y_train, cv=4, scoring = "accuracy")
    # print("Scores:", scores)
    # print("Mean:", scores.mean())
    # print("Standard Deviation:", scores.std())

    #Decision Tree
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, Y_train)  
    Y_pred = decision_tree.predict(X_test)  
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

    print('decision tree', acc_decision_tree)

    predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)

    # results = pd.DataFrame({
    #     'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
    #             'Random Forest', 'Naive Bayes', 'Perceptron', 
    #             'Stochastic Gradient Decent', 
    #             'Decision Tree'],
    #     'Score': [acc_linear_svc, acc_knn, acc_log, 
    #             acc_random_forest, acc_gaussian, acc_perceptron, 
    #             acc_sgd, acc_decision_tree]})
    # result_df = results.sort_values(by='Score', ascending=False)
    # result_df = result_df.set_index('Score')
    # result_df.head(9)
    # print(get_team_row('PHI', 2008)['win_pct'])
    # print(winning_pct)
# main()

main()
