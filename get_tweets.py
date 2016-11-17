import got
import codecs

def receiveBuffer(tweets, outputFile):
    for t in tweets:
      outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
    outputFile.flush();
    print 'More %d saved on file...\n' % len(tweets)
    

def get_tweets(filename, maxtweets, query, username='', since='', until='', top=False):
    tweetCriteria = got.manager.TweetCriteria()
    tweetCriteria.maxTweets = maxtweets
    tweetCriteria.querySearch = query
    
    if username:
        tweetCriteria.username = username

    if since:
        tweetCriteria.since = since

    if until:
        tweetCriteria.until = until

    if top:
        tweetCriteria.topTweets = True

    outputFile = codecs.open(filename, "w+", "utf-8")
    outputFile.write('username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink')
    got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

    outputFile.close()
    print 'Done. Output file generated "output_got.csv".'

