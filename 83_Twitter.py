#Import the necessary methods from tweepy library
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import pandas as pd

#provide your access details below
access_token = "2466359646-OIQVr46cmst5rI5zYE6qQGFEfyONoZ3Kg8OoK5U"
access_token_secret = "xKHtaTv1QVzZr0DfPsY86eWQPF8tU22GHw6GPwkw3FDI3"
consumer_key = "438MVYaAgd5vJlFlHgXfalwt8"
consumer_secret = "wv8LpFTRVGVRV732BkTkyJREYLn34AoPB1JvuJaz06DOh4cdYE"

# establish a connection
auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#fetch recent 10 tweets containing words iphone7 camera
fetched_tweets = api.search(q=['iPhone 7','iPhone7','camera'], result_type='recent', lang='en', count=10)
print ("Number of tweets: ", len(fetched_tweets))

for tweet in fetched_tweets:
    print ('Tweet ID: ', tweet.id)
    print ('Tweet Text: ', tweet.text, '\n')


def populate_tweet_df(tweets):
    # Create an empty dataframe
    df = pd.DataFrame()

    df['id'] = list(map(lambda tweet: tweet.id, tweets))
    df['text'] = list(map(lambda tweet: tweet.text, tweets))
    df['retweeted'] = list(map(lambda tweet: tweet.retweeted, tweets))
    df['place'] = list(map(lambda tweet: tweet.user.location, tweets))
    df['screen_name'] = list(map(lambda tweet: tweet.user.screen_name, tweets))
    df['verified_user'] = list(map(lambda tweet: tweet.user.verified, tweets))
    df['followers_count'] = list(map(lambda tweet: tweet.user.followers_count, tweets))
    df['friends_count'] = list(map(lambda tweet: tweet.user.friends_count, tweets))

    # Highly popular user's tweet could possibly seen by large audience, so lets check the popularity of user
    df['friendship_coeff'] = list(
        map(lambda tweet: float(tweet.user.followers_count) / float(tweet.user.friends_count), tweets))
    return df

