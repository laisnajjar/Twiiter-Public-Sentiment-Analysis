"""Palestine Tweets Search."""
import os
import json
import requests

bearer_token = os.environ.get("BEARER_TOKEN")

search_url = "https://api.twitter.com/2/tweets/search/recent"

search_term = '(Israel OR israel) lang:en' # Replace this value with your search term
# search_term = '(Palestine OR palestine) lang:en' # Replace this value with your search term

query_params = {
    'query': search_term,
    'tweet.fields': 'id,text,author_id,conversation_id,created_at,lang,public_metrics,'
                    'possibly_sensitive,source',
    'expansions': 'author_id,attachments.media_keys,attachments.poll_ids,referenced_tweets.id',
    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
    'media.fields': 'url,preview_image_url,type',
    'poll.fields': 'options,duration_minutes,end_datetime,voting_status',
    'place.fields': 'full_name,id,country,country_code,geo',
    'max_results': 100  #  (max 100 for recent search)
}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url, params):
    """Connect to Twitter Endpoint."""
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    """Run Search."""
    for i in range(0, 1):
        json_response = connect_to_endpoint(search_url, query_params)
        file_path = f"Collected_Tweets/Eval_Israel"
        with open(file_path, 'a', encoding="UTF-8") as file:
            json.dump(json_response, file, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
