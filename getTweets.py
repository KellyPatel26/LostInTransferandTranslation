import pandas as pd
import numpy as np
import tweepy

import requests
import os
import json
import csv

# API_KEY = "YOUR API KEY"
# API_KEY_SECRET = "YOUR SECRET API KEY"
# Bearer_Token = "YOUR BEARER TOKEN"
API_KEY = "5kctWxsh7bsSK8gf3UNKI1OW8"
API_KEY_SECRET = "pcDHkXQCyQ7gu1LjP7W9bpiC8NWhtzdoYbQQLv8U0raBYjwHYH"
Bearer_Token = "AAAAAAAAAAAAAAAAAAAAAFRFigEAAAAAcvYhSXD0pOv0gJK%2FrxKGGgfpBhI%3DMnpYFjb5saB6Uor0HbpqflGBWi6D50ukGkK4YtNfH8LmW71B7X"

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = Bearer_Token


def create_url(ids):
    tweet_fields = "tweet.fields=lang,author_id"
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def main():

    
    # filename = "data/Croatian_Twitter_sentiment.csv"    # Croatian data
    # filename = "data/Slovenian_Twitter_sentiment.csv"  # Slovenian data
    #filename = "data/Polish_Twitter_sentiment.csv"    # Croatian data
    filename = "data/English_Twitter_sentiment.csv"     # English data
    df = pd.read_csv(filename)

    ids_list = df.TweetID
    labels = dict()
    for index, row in df.iterrows():
        labels[str(row["TweetID"])] = row["HandLabel"]

    tweet_text = []
    tweet_labels = []
    for i in range(89700, len(ids_list), 100):
        id_batch = ids_list[i:i+100]
        id_batch = [str(id) for id in id_batch]
        ids = "ids="+",".join(id_batch)

        url = create_url(ids)
        json_response = connect_to_endpoint(url)
        # print(json.dumps(json_response, indent=4, sort_keys=False))
        # print(json_response["data"])
        print(i)
        with open("english_tweets.txt", "a", encoding='utf-8') as file1, open("english_labels.txt", "a") as file2:
            for entry in json_response["data"]:
                tweet_text.append(entry["text"])
                tweet_labels.append(labels[entry["id"]])
                file1.write(entry["text"]+"STOPSTOPSTOP\n")
                file2.write(labels[entry["id"]]+"\n")
                # print(entry, labels[entry["id"]])          
    


if __name__ == "__main__":
    main()

