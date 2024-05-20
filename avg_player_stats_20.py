import pandas as pd
from get_response import get_response
from constants import *


""" 
This script is currently not used anywhere. Both of the functions collect recent stats for given amount of matches
"""


def get_avg_player_stats_20(player_id, num_of_matches):
    player_stats_url = f'https://open.faceit.com/data/v4/players/{player_id}/games/cs2/stats?offset=0&limit={num_of_matches}'
    data = get_response(player_stats_url)
    all_stats = []
    for item in data['items']:
        stats = item['stats']
        all_stats.append(stats)
    df = pd.DataFrame(all_stats)
    df = df.apply(pd.to_numeric, errors='coerce')
    avg_column = df.mean()
    return avg_column.dropna(axis=0)


def get_player_stats_for_num_of_matches(player_id, timestamp, num_of_matches):
    history_url = (f'https://open.faceit.com/data/v4/players/{player_id}/history?game=cs2&to={timestamp}'
                   f'&offset=0&limit={num_of_matches}')
    history_data = get_response(history_url)
    all_stats = []
    for item in history_data['items']:
        match_id = item['match_id']
        match_stats_url = f'https://open.faceit.com/data/v4/matches/{match_id}/stats'
        match_data = get_response(match_stats_url)
        for team in match_data['rounds'][0]['teams']:
            for player in team['players']:
                if player['player_id'] == player_id:
                    all_stats.append(player['player_stats'])
    df = pd.DataFrame(all_stats)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.mean()


if __name__ == '__main__':
    print(get_avg_player_stats_20(MY_ID, 20))
    print('------------------------------------------------------')
    print(get_player_stats_for_num_of_matches(MY_ID, 1715121248, 20))
