from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model_training import get_data_from_file, get_train_test_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib


"""
Given a key returns a list of values from data
"""


def get_list_of_data_by_key(key, data):
    result_list = []
    for item in data:
        for team, stats in item.items():
            result_list.append(stats[key])
    return result_list


"""
Given a key returns a list of values from data, but only for one team
"""


def get_list_of_data_for_1_team(key, data):
    result_list = []
    for item in data:
        result_list.append(item['team1'][key])
    return result_list


"""
Generates histogram of "Score" values
"""


def score_histogram(data):
    # Gets values of "Score" for one team
    data_list = get_list_of_data_for_1_team('Score', data)

    df = pd.DataFrame(data_list, columns=['Value'])

    count_values = df['Value'].value_counts()
    plt.figure(figsize=(6, 4))
    count_values.plot(kind='bar', color=['skyblue', 'orange'])
    plt.title('Distribution of Score')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()


"""
Generates histogram of values for given key
"""


def generate_histogram(data, key):
    data_list = get_list_of_data_by_key(key, data)

    df = pd.DataFrame(data_list, columns=['Value'])

    plt.figure(figsize=(8, 6))
    plt.hist(df['Value'], bins=len(df['Value'].unique()), color='skyblue')
    plt.title(f'Distribution of values of {key}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


"""
Generates histograms of all keys in data, except "Score"
"""


def generate_histograms(data):
    for key, value in data[0]['team1'].items():
        if not key == 'Score':
            generate_histogram(data, key)


"""
Generates confusion matrix for a given model and data
"""


def confusion_matrix_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test.tolist(), y_pred.tolist(), labels=model.classes_.tolist())
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["1.0", "0.0"],
    )
    disp.plot()
    plt.title(f'Confusion Matrix of {model.__class__.__name__}')
    plt.show()


"""
Generates a violin plot that shows dependency between values of given key and "Score"
"""


def dependency_with_score_plot(data, key):
    key_list = get_list_of_data_for_1_team(key, data)
    score_list = get_list_of_data_for_1_team('Score', data)
    df = pd.DataFrame({key: key_list, 'Score': score_list})
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=key, hue='Score', data=df, palette='muted', split=True)
    plt.title(f'{key} to Score')
    plt.xlabel('Matches')
    plt.ylabel(key)
    plt.grid(True)
    plt.show()


"""
Generates a plot that displays results of out experiment
"""


def result_plot():
    results = {'Model': ['SVM', 'Random Forest', 'LightGBM', 'XGBoost'],
               'Score': [0.694247, 0.700251, 0.721349, 0.742481]}
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plot = sns.barplot(x='Model', y='Score', data=df)
    plot.set(ylim=(0, 1))
    plt.show()


"""
Generates a plot that shows correlation between difference of "Rating" of two teams and "Score"
"""


def rating_to_score_plot(data):
    rating_data = []
    # First we create a list of tuples that contain: rating of winning team, rating of losing team and Boolean value
    # that says if the winning team had higher rating
    for match in data:
        rating_team1 = match['team1']['Rating']
        rating_team2 = match['team2']['Rating']
        score_team1 = match['team1']['Score']
        score_team2 = match['team2']['Score']
        if score_team1 > score_team2:
            winner_rating = rating_team1
            loser_rating = rating_team2
        else:
            winner_rating = rating_team2
            loser_rating = rating_team1
        rating_data.append((winner_rating, loser_rating, winner_rating > loser_rating))

    df = pd.DataFrame(rating_data, columns=['Winner_rating', 'Loser_rating', 'Team with higher rating won'])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Loser_rating', y='Winner_rating', data=df, hue_order=[True, False],
                    hue='Team with higher rating won', palette={True: 'green', False: 'red'})
    plt.title('Dependency between rating difference of teams and score')
    plt.xlabel('Rating of losing team')
    plt.ylabel('Rating of winning team')
    plt.grid(True)
    plt.show()


"""
Creates a correlation map of differences of values
"""


def var_correlation_map(data):
    # Iterates over every match in data and creates lsit of dicts that contains differences of every value in team1 and team2
    data_one_team = []
    for item in data:
        difference_dict = {}
        for key, value in item['team1'].items():
            if not key == 'Score':
                difference_dict[key] = value - item['team2'][key]
            difference_dict['Score'] = item['team1']['Score']
        data_one_team.append(difference_dict)

    df = pd.DataFrame(data=data_one_team)
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Map of Differences of Variables')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = get_data_from_file('ropl_matches_data_with_maps.json')
    X_train, X_test, y_train, y_test = get_train_test_data(data)
    generate_histograms(data)
    score_histogram(data)
    dependency_with_score_plot(data, 'MWin Rate %')
    dependency_with_score_plot(data, 'Average K/D Ratio')
    rating_to_score_plot(data)
    dependency_with_score_plot(data, 'Average Headshots %')
    var_correlation_map(data)
    svm = joblib.load('models/svm.pkl')
    rd = joblib.load('models/rd.pkl')
    lgb = joblib.load('models/lgb.pkl')
    xbc = joblib.load('models/xbc.pkl')
    models = [svm, rd, lgb, xbc]
    for model in models:
        confusion_matrix_plot(model, X_test, y_test)
    result_plot()