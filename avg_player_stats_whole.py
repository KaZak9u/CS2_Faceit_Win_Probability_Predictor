import pandas as pd
from get_response import get_response
from constants import *


"""
Returns a series that contains lifetime stats of a player
"""


def get_avg_player_stats(player_id):
    player_stats_url = f'https://open.faceit.com/data/v4/players/{player_id}/stats/cs2'
    data = get_response(player_stats_url)
    # From overall stats, we want the most relevant ones, so we choose these selected:
    selected_columns = ['Win Rate %', 'Matches', 'Average K/D Ratio', 'Longest Win Streak', 'Average Headshots %']
    try:
        series = pd.Series(data['lifetime'])
        series_selected = series[selected_columns]
        series_selected = series_selected.apply(pd.to_numeric)
        return series_selected
    except KeyError:
        return pd.Series()


if __name__ == '__main__':
    print(get_avg_player_stats(MY_ID))
