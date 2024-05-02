import os
from openai import OpenAI

from dotenv import dotenv_values
config = dotenv_values(".env")

# Load environment variables
if not os.getenv("PRODUCTION") or not config['PRODUCTION']:
    production = config['PRODUCTION']
    openai_key = config['OPENAI_API_KEY']
else:
    production = os.getenv('PRODUCTION')
    openai_key = os.getenv('OPENAI_API_KEY')

class OpenAiAPI:
    def __init__(self):
        self.openai_api_key = openai_key

        ########################################################
    # Define the chat function
    ########################################################
    def chat(self, msgs):
        """
        Chat with the OpenAI API
        :param msgs: List of messages
        return: OpenAI response
        """
        openai = OpenAI(api_key=self.openai_api_key)
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=msgs
        )
        message = response
        return message
    
    ########################################################
    # Define the get_market_sentiment function
    ########################################################
    def get_market_sentiment(self, title, article):
        """
        Get the sentiment of the article using OpenAI API sentiment analysis
        :param article: Article
        return: Sentiment of the article (BULLISH, BEARISH, NEUTRAL)
        """
        message_history = []
        sentiments = []
        # Send the system message to the OpenAI API
        system_message = 'You will work as a Sentiment Analysis for Financial news. I will share news headline and article. You will only answer as:\n\n BEARISH,BULLISH,NEUTRAL. No further explanation. \n Got it?'
        message_history.append({'content': system_message, 'role': 'user'})
        response = self.chat(message_history)

        # Send the article to the OpenAI API
        user_message = '{}\n{}'.format(title, article)
        
        message_history.append({'content': user_message, 'role': 'user'})
        response = self.chat(message_history)
        sentiments.append(
            {'title': title, 'article': article, 'signal': response.choices[0].message.content})
        message_history.pop()
        # Return the sentiment
        return sentiments[0]['signal']