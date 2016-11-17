import urllib2
from bs4 import BeautifulSoup
import re
from urlparse import urljoin
import time
from pprint import pprint
import pickle
import sys
from BOM_movie import BOM_movie
import pandas as pd
def get_soup_from_url(url):
    """
    Create a soup from a complete url
    """
    opener = urllib2.build_opener()
    opener.addheader = [('User-agent', "Mozilla/5.0")]
    page = opener.open(url)  
    soup = BeautifulSoup(page)
    return soup
    

def get_all_mojo_films(filename):
    """
    Obtain a dictionary with boxofficemojo id and movie title scraping alphabetical index page.\
    Save pickle file with dictionary to <filename>
    """
    
    alphabet = ['NUM','A','B','C','D','E','F','G','H','I','J','K','L',
                    'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    mojo_films = {}
    for num, char in enumerate(alphabet):
        first_page = 'http://www.boxofficemojo.com/movies/alphabetical.htm?letter=' + char + '&p=.htm'
        soup = get_soup_from_url(first_page)
        rows = [td.parent for td in soup.find_all('td', align='left')]
        for row in rows:
            mojo_films[row.a['href'].replace(u'\xa0','%A0').split('=')[1].split('.')[0]]=row.a.text.replace(u'\xa0','%A0')
        time.sleep(1)
        sub_pages = [urljoin('http://www.boxofficemojo.com/', r['href'])\
                     for r in soup.find(class_="alpha-nav-holder").find_all('a')]
        for page in sub_pages:
            soup = get_soup_from_url(page)
            rows = [td.parent for td in soup.find_all('td', align='left')]
            for row in rows:
                mojo_films[row.a['href'].replace(u'\xa0','%A0').split('=')[1].split('.')[0]]=row.a.text.replace(u'\xa0','%A0')
            time.sleep(1)
        with open(filename, 'w') as picklefile:    
            pickle.dump(mojo_films, picklefile)
    return mojo_films

with open("movie_ids.pkl", 'r') as picklefile: 
    movie_ids = pickle.load(picklefile)


movies_scraped = []
movies_skipped = []

for key in movie_ids.iterkeys():
    try:
    	print key
        movie = BOM_movie(key)
        movies_scraped.append(movie.get_movie_data())
    except:
        movies_skipped.append(key)

with open('movies_data1.pkl', 'w') as picklefile:
    movies_data = movies_scraped, movies_skipped
    pickle.dump(movies_data, picklefile)
with open("movies_data1.pkl", 'r') as picklefile: 
    movies_scraped, movies_skipped = pickle.load(picklefile)
    
print 'Movies scraped: ', len(movies_scraped)
print 'Movies skipped: ', len(movies_skipped)
print '\n'
print 'First elements: '
for i in range(5):
    pprint(movies_scraped[i])