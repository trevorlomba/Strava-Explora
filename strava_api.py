import requests
import urllib3

"""
This module imports the `requests` and `urllib3` libraries for making HTTP requests.

`requests` is a Python library for making HTTP requests, which provides a simpler and more human-friendly interface than the built-in `urllib` module.

`urllib3` is a powerful HTTP client library for Python, which provides advanced features like connection pooling, retries, and redirect handling.
"""

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"

payload = {
    'client_id': "105014",
    'client_secret': '5e4d56ba9319a35f988bfdae1f511cc99d0bdf21',
    'refresh_token': '7b38f3ec6de0510c8c5d44afec614d49b7572420',
    'grant_type': "refresh_token",
    'f': 'json'
}

# print("Requesting Token...\n")
# res = requests.post(auth_url, data=payload, verify=False)
# access_token = res.json()['access_token']
# print("Access Token = {}\n".format(access_token))

# header = {'Authorization': 'Bearer ' + access_token}
# param = {'per_page': 200, 'page': 1}
# my_dataset = requests.get(activites_url, headers=header, params=param).json()


def get_strava_data():
    print("Requesting Token...\n")
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()['access_token']
    print("Access Token = {}\n".format(access_token))

    header = {'Authorization': 'Bearer ' + access_token}
    param = {'per_page': 200, 'page': 1}
    my_dataset = requests.get(
        activites_url, headers=header, params=param).json()

    return my_dataset

# print(my_dataset[0]["name"])
# print(my_dataset[0]["map"]["summary_polyline"])
