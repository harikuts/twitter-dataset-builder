import tweepy
import argparse
import pdb
import string
import itertools
import datetime
import demoji
import io
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
import time

# Import authentication information
import auth

# demoji.download_codes()

STOP_WORDS = set(stopwords.words('english'))
WORDS = set(words.words())

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
    text = demoji.replace(text, "")
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
    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator)
    text = text.lower()
    text = [word for word in text.split() if word in WORDS]
    text = text.split()
    text = " ".join(text)
    return text

def remove_repeats(data):
    # Remove repeated words
    z = []
    for i in data.split():
        if i not in z:
            z.append(i)
    data = ' '.join(z)
    return data

def save_corpus(corpus, filename):
    lines = "\n".join([' '.join(words) for words in corpus])
    with io.open(filename, 'w', encoding="utf-8") as f:
        f.write(lines)
    print("Saved to", filename)

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
    parser.add_argument("--wait", action="store_true", default=False, \
        help = "Wait between tweet retrieval limits. If not enabled, scraper will exit and save at current point.")
    parser.add_argument("--output", "-o", default=None, action="store", \
        help = "Output filename. If not included, a name will be generated.")
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
            try:
                status = api.get_status(tweet.id, tweet_mode='extended')
                # Clean the status (removing all mentions, media links, etc.)
                text = clean_status(status)
                if text != "":
                    print(counter, "::", text)
                    text = text.split()
                    corpus.append(text)
                    counter += 1
            except tweepy.error.RateLimitError:
                if not args.wait:
                    break
                else:
                    print("Waiting out the tweet retrieval limit. (%s)" % (datetime.datetime.now().strftime("%H:%M:%S")))
                    time.sleep(900)
            except Exception as e:
                print(e)
                break

    # Save corpus
    now = datetime.datetime.now()
    filename = (args.account + "_" + str(counter) + "_" \
        + now.strftime("%Y-%m-%d_%H-%M-%S") + ".twt")\
             if args.output is None else args.output
    save_corpus(corpus, filename)
