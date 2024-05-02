import os
import json
import http.client, urllib.parse

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
production = os.getenv('PRODUCTION')
extract_key = os.getenv('ARTICLEEXTRACT_API_KEY')

class ArticleExtractor:
    def __init__(self):
        self.article_extractor_key = extract_key

    ########################################################
    # Define the extract_article function
    ########################################################
    def extract_article(self, url):
        """
        Get the article extracted from the URL using ArticleXtractor API
        :param url: URL
        return: Extracted article
        """
        conn = http.client.HTTPSConnection('api.articlextractor.com')

        params = urllib.parse.urlencode({
            'api_token': self.article_extractor_key,
            'url': "{article}".format(article=url),
            })

        conn.request('GET', '/v1/extract?{}'.format(params))

        res = conn.getresponse()
        data = json.loads(res.read())
        
        # Return the extracted article title and text
        return {
            'title': data['data']['title'], 
            'article': data['data']['text']
            }