import pandas as pd
from avg_player_stats_whole import get_avg_player_stats
from avg_player_map_stats import get_avg_player_map_stats


"""
Returns a 2 element series that contain average lifetime + map stats and score for both teams in a match
"""


def get_avg_team_stats(match_data):
    # First we get a map name of this match
    map_name = get_map_name_of_a_match(match_data)
    teams = []
    # Second we iterate over 2 teams in a match
    for faction_num in (1, 2):
        faction = f'faction{faction_num}'
        team_stats = []
        # Then we iterate over players in a team
        for player_info in match_data['teams'][faction]['roster']:
            # We get lifetime player stats given id of a player
            player_stats = get_avg_player_stats(player_info['player_id'])
            # If everything is correct, we want to get map stats of a player and connect them with lifetime stats
            if not player_stats.empty:
                map_stats = get_avg_player_map_stats(player_info['player_id'], map_name)
                player_stats = player_stats._append(map_stats)
                team_stats.append(player_stats)
        df = pd.DataFrame(team_stats)
        # In the end we add rating of a team, and score: 1.0 for a win and 0.0 for a loss
        rating = pd.Series([match_data['teams'][faction]['stats']['rating']], index=['Rating'])
        score = pd.Series([match_data['results']['score'][faction]], index=['Score'])
        teams.append(df.mean()._append(rating)._append(score))
    return teams


"""
Returns the name of a map that given match is played on
"""


def get_map_name_of_a_match(match_data):
    try:
        map_name = match_data['voting']['map']['pick'][0]
        # Map name in a match_data is in format: "de_mapname", but we need "Mapname", so we remove first 3
        # letters and capitalize
        map_name = map_name[3:].capitalize()
    except KeyError:
        # If somehow match_data is incorrect and there is no map name in data, then we assume match was played on mirage
        # because it's the most common map
        map_name = "Mirage"
    return map_name
