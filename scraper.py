import tweepy
import argparse
import pdb
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
    print(text)
    categories = status.entities.keys()
    # Hashtags
    if "hashtags" in categories:
        print("HASHTAGS:")
        for hashtag in status.entities["hashtags"]:
            print(hashtag)
            # Add a hashtag and remove the word from the text
            text = text.replace("#" + hashtag["text"], "")
    if "symbols" in categories:
        print("SYMBOLS:")
        for symbol in status.entities["hashtags"]:
            print(symbol)
    if "user_mentions" in categories:
        print("USER MENTIONS:")
        for mention in status.entities["user_mentions"]:
            print(mention)
    if "urls" in categories:
        print("URLS:")
        for url in status.entities["urls"]:
            print(url)
    if "media" in categories:
        print("MEDIA:")
        for medium in status.entities["media"]:
            print(medium)
    print(text)

    
    

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
    counter = 0
    for tweet in tweepy.Cursor(api.user_timeline).items():
        if counter >= args.num:
            break
        elif not args.originals or not is_retweeted(tweet):
            print("")
            status = api.get_status(tweet.id, tweet_mode='extended')
            text = status.full_text
            clean_status(status)
            counter += 1