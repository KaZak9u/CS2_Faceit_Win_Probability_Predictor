import pandas as pd
from get_response import get_response
from constants import *

"""
Returns a series of lifetime stats of a player on a given map
"""


def get_avg_player_map_stats(player_id, map_name):
    player_stats_url = f'https://open.faceit.com/data/v4/players/{player_id}/stats/cs2'
    data = get_response(player_stats_url)
    # We choose the most relevant stats:
    selected_columns = ["MAverage MVPs", "MAverage K/R Ratio", "MAverage K/D Ratio", "MWin Rate %", "MAverage Kills"]
    stats = {}
    try:
        for segment in data['segments']:
            if segment['label'] == map_name:
                stats = segment['stats']
        # We want to rename every map stats, because lifetime stats of a player have the same names of columns
        stats = rename_keys_in_dict(stats)
        series = pd.Series(stats)
        series_selected = series[selected_columns]
        series_selected = series_selected.apply(pd.to_numeric)
        return series_selected
    except KeyError:
        return pd.Series()


"""
For a given dictionary, adds letter "M" at the front of every key and returns new dictionary with renamed keys
"""


def rename_keys_in_dict(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        new_dict["M" + key] = value
    return new_dict


if __name__ == '__main__':
    print(get_avg_player_map_stats(MY_ID, "Mirage"))