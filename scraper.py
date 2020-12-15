import tweepy

# Import authentication information
import auth

def authenticate_api():
    authorization = tweepy.OAuthHandler(auth.consumer_key, auth.consumer_secret)
    authorization.set_access_token(auth.access_token, auth.access_secret)
    api = tweepy.API(authorization)
    return api

# Create API object
api = authenticate_api()

public_tweets = api.user_timeline('hardkombucha')
for tweet in public_tweets:
    if not tweet.retweeted:
        print(tweet.text)