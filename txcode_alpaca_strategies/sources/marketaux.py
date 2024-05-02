import os
import json
import http.client, urllib.parse
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
production = os.getenv('PRODUCTION')
article_key = os.getenv('MARKETAUX_API_KEY')

class MarketAux:
    def __init__(self):
        self.market_aux_key = article_key

    def get_news_articles(self, symbol, limit=3):
        """
        Get the sentiment of the news articles for the given symbol using MarketAux API and OpenAI sentiment analysis
        :param symbol: Stock symbol
        return: True if the sentiment is bullish
        return: False if the sentiment is bearish
        """
        # Get the news articles for the given symbol
        conn = http.client.HTTPSConnection("api.marketaux.com")
        params = urllib.parse.urlencode({
            'api_token': self.market_aux_key,
            'symbols': symbol,
            'limit': limit,
            })
        
        conn.request('GET', '/v1/news/all?{}'.format(params))

        res = conn.getresponse()
        data = json.loads(res.read())
        data = pd.json_normalize(data['data'])

        return data