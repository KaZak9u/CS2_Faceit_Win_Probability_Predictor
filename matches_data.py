import json
from json import JSONDecodeError
from constants import *
from avg_team_stats import get_avg_team_stats
from get_response import get_response


def get_match_data(hub_id, offset, num_of_matches):
    url = f'https://open.faceit.com/data/v4/hubs/{hub_id}/matches?type=past&offset={offset}&limit={num_of_matches}'
    return get_response(url)


"""
Given data of some number of matches, gets average stats of both teams and adds it to .json file for every match
"""


def save_matches_data(matches_data):
    matches = []
    DEBUG = 0
    path = 'ropl_matches_data_with_maps.json'
    # First we collect data that already exist in file
    existing_matches = retrieve_existing_data(path)
    # Second we iterate over matches
    for item in matches_data['items']:
        # We need to check whether the match actually took place. Status "FINISHED" makes sure of that, otherwise we
        # would get matches that haven't started
        if item['status'] == 'FINISHED':
            # We get average stats for both teams and add separate them in dictionary
            teams_stats = get_avg_team_stats(item)
            df_dict = {'team1': teams_stats[0].to_dict(), 'team2': teams_stats[1].to_dict()}
            # At the end we append this dict to list that consist stats of every match
            matches.append(df_dict)
            print(DEBUG)
            DEBUG += 1
    # Finally we connect existing data and new one and save it to .json file
    existing_matches = existing_matches + matches
    with open(path, 'w') as file:
        json.dump(existing_matches, file, indent=True)
    print("Data succesfully saved to file: ", path)


"""
Returns data that already exists in file
"""


def retrieve_existing_data(path):
    with open(path, 'r') as file:
        try:
            existing_matches = json.load(file)
        except JSONDecodeError:
            existing_matches = []
    return existing_matches


"""
In one request we can get at most 100 matches data, so if we want to get 1000 matches data, we have to send 10 request
with different offsets (starting positions of stored data).
"""


def save_big_num_of_matches(num_of_100):
    for i in range(num_of_100):
        offset = i * 100
        matches_data = get_match_data(ROPL_ID, offset, 100)
        save_matches_data(matches_data)
        print(f"Saved {offset + 100} matches")


if __name__ == '__main__':
    save_big_num_of_matches(12)
    # matches_data = get_match_data(ROPL_ID, 0, 9)
    # save_matches_data(matches_data)