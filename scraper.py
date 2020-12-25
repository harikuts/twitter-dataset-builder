import tweepy
import argparse
import pdb
import string

# Import authentication information
import auth


# Connect to API 
def authenticate_api():
    authorization = tweepy.OAuthHandler(auth.consumer_key, auth.consumer_secret)
    authorization.set_access_token(auth.access_token, auth.access_secret)
    api = tweepy.API(authorization)
    return api

# Check if tweet has been retweeted
def is_retweeted(tweet):
    if tweet.retweeted or "RT @" in tweet.text:
        return True
    else:
        return False

# Return text clean of media
def clean_status(status):
    text = status.full_text
    categories = status.entities.keys()
    if "hashtags" in categories:
        for hashtag in status.entities["hashtags"]:
            text = text.replace("#" + hashtag["text"], "")
    if "symbols" in categories:
        for symbol in status.entities["hashtags"]:
            pass
    if "user_mentions" in categories:
        for mention in status.entities["user_mentions"]:
            text = text.replace("@" + mention["screen_name"], "")
    if "urls" in categories:
        for url in status.entities["urls"]:
            text = text.replace(url["url"], "")
    if "media" in categories:
        for medium in status.entities["media"]:
            text = text.replace(medium["url"], "")
    return text

    
    

# Clean any links

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", "-a", required=True, action="store", \
        help = "Target account to be scraped.")
    parser.add_argument("--num", "-n", required=True, action="store", type=int, \
        help = "Number of tweets to scrape.")
    parser.add_argument("--originals", action="store_true", default=False, \
        help = "Keep original tweets and ignore retweets.")
    args = parser.parse_args()
    print("Processing account", args.account)
    # Create API object
    api = authenticate_api()
    # Process tweets as they come, up until a certain number
    corpus = []
    counter = 0
    for tweet in tweepy.Cursor(api.user_timeline, id=args.account).items():
        if counter >= args.num:
            break
        elif not args.originals or not is_retweeted(tweet):
            status = api.get_status(tweet.id, tweet_mode='extended')
            # Clean the status (removing all mentions, media links, etc.)
            text = clean_status(status)
            # Remove punctuation
            translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            text = text.translate(translator)
            text = text.split()
            corpus.append(text)
            counter += 1