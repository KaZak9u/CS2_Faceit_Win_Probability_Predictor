import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from avg_team_stats import get_avg_team_stats
from get_response import get_response
from model_training import create_features_vector


"""
Function that returns a probabilities of each team winning for a given match. room_url is a link to your matchroom.
"""


def predict_proba_for_match(model, room_url):
    # Getting match id from matchroom
    url_splitted = room_url.split('/')
    room_index = url_splitted.index("room")
    match_id = url_splitted[room_index + 1]

    # Getting stats of both team in this match
    request_url = f'https://open.faceit.com/data/v4/matches/{match_id}'
    match_data = get_response(request_url)
    match_stats = get_avg_team_stats(match_data)
    stats_team1 = list(match_stats[0].to_dict().values())
    stats_team2 = list(match_stats[0].to_dict().values())
    X = []

    # Creating a features vector to pass to a model
    features = create_features_vector(stats_team1, stats_team2)
    X.append(features)
    X = np.array(X).reshape(1, -1)

    # Normalizing data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Returning predicted probability by a model
    return model.predict_proba(X)


if __name__ == "__main__":
    model = joblib.load('models/xbc.pkl')
    matchroom = 'https://www.faceit.com/en/cs2/room/1-a47ea657-2281-4dd4-a36a-80e34e3a7605'
    try:
        prob = predict_proba_for_match(model, matchroom)
        print("Probability team 1 will win: " + str(prob[0][0]))
        print("Probability team 2 will win: " + str(prob[0][1]))
    except ValueError:
        print("Incorrect matchroom url")
