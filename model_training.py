import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
import joblib
from xgboost import XGBClassifier


"""
Returns a data, that already exists in file
"""


def get_data_from_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


"""
Function responsible for data preprocessing. Creates a feature vector for every match, normalizes it and splits into
train and test sets.
"""


def get_train_test_data(data):
    X = []
    y = []
    for match in data:
        # We collect stats of both teams
        stats_team1 = list(match['team1'].values())
        stats_team2 = list(match['team2'].values())
        # As a result we need Score of only one team
        result = stats_team1[-1]

        # Then we append features vector of data to our X set
        features = create_features_vector(stats_team1, stats_team2)

        X.append(features)
        y.append(result)

    X = np.array(X)
    y = np.array(y)

    # Normalizing data
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Returning split train and test data with test_size=0.2
    return train_test_split(X_normalized, y, test_size=0.2, random_state=42)


"""
Creates and returns features vector of data od both teams (drops "Longest Win Streak" and "Matches" column).
"""


def create_features_vector(stats_team1, stats_team2):
    features = [stats_team1[0], stats_team1[2]] + stats_team1[4:10] + [stats_team2[0], stats_team2[2]] + stats_team2[
                                                                                                         4:10]
    return features


"""
Functions that are responsible for training model with best parameters and returning that model
"""


def svm_data_training(X_train, X_test, y_train, y_test):
    model = SVC(probability=True)
    params = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [1.0, 2.0, 4.0],
        'gamma': [0.001, 0.1, 1.]
    }

    svm, best_params = find_best_parameters(model, params, X_train, y_train, verbose=3)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy score: ", accuracy)
    print("Best params: ", best_params)
    return svm


def xbc_data_training(X_train, X_test, y_train, y_test):
    model = XGBClassifier(objective='binary:logistic', random_state=42)

    parameters = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    xbc, best_params = find_best_parameters(model, parameters, X_train, y_train, verbose=3)
    y_pred = xbc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score: ", accuracy)
    print("Best params: ", best_params)
    return xbc


def rd_data_training(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()

    parameters = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
                  }

    rd, best_params = find_best_parameters(model, parameters, X_train, y_train, verbose=3)

    y_pred = rd.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score: ", accuracy)
    print("Best params: ", best_params)
    return rd


def lgb_data_training(X_train, X_test, y_train, y_test):
    #train_data = lgb.Dataset(X_train, label=y_train)
    #test_data = lgb.Dataset(X_test, label=y_test)

    params = {
        'objective': ['binary'],
        'metric': ['binary_error'],
        'verbosity': [2],
        'learning_rate': [0.1, 0.01, 0.001],
        'num_iterations': [100],
        'max_leaves': [10, 20, 31],
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7]
    }
    # model = lgb.train(params, train_data, valid_sets=[test_data])
    model = LGBMClassifier()
    model, best_params = find_best_parameters(model, params, X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_binary = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy score:", accuracy)
    print("Best params: ", best_params)
    return model


"""
Performs a grid search on a trained model and returns the best model and its parameters
"""


def find_best_parameters(model, parameters, X, y, cv=5, verbose=1, n_jobs=-1):
    grid_object = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score), cv=cv, verbose=verbose, n_jobs=n_jobs)
    grid_object = grid_object.fit(X, y)
    return grid_object.best_estimator_, grid_object.best_params_


if __name__ == '__main__':
    # Collecting data
    data = get_data_from_file('ropl_matches_data_with_maps.json')
    X_train, X_test, y_train, y_test = get_train_test_data(data)

    # Training models
    svm = svm_data_training(X_train, X_test, y_train, y_test)
    lgb = lgb_data_training(X_train, X_test, y_train, y_test)
    rd = rd_data_training(X_train, X_test, y_train, y_test)
    xbc = xbc_data_training(X_train, X_test, y_train, y_test)

    # Getting cross_val_score for models
    svm_scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=5)
    lgb_scores = cross_val_score(lgb, X_train, y_train, scoring='accuracy', cv=5)
    rd_scores = cross_val_score(rd, X_train, y_train, scoring='accuracy', cv=5)
    xbc_scores = cross_val_score(xbc, X_train, y_train, scoring='accuracy', cv=5)

    results = [
        ['SVM', np.mean(svm_scores)],
        ['LightGBM', np.mean(lgb_scores)],
        ['Random Forest', np.mean(rd_scores)],
        ['XGBoost', np.mean(xbc_scores)]
    ]

    # Displaying results
    results.sort(key=lambda model: model[1])
    results = pd.DataFrame(data=results, columns=['Model', 'Score'])
    print(results)

    # Saving trained models in files
    joblib.dump(svm, 'models/svm.pkl')
    joblib.dump(lgb, 'models/lgb.pkl')
    joblib.dump(rd, 'models/rd.pkl')
    joblib.dump(xbc, 'models/xbc.pkl')
