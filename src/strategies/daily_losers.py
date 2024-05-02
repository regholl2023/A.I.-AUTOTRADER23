import os
import time
import json
import pandas as pd
from tqdm import tqdm

from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from src.alpaca import AlpacaAPI
from src.sources.yahoo import Yahoo
from src.sources.slack import Slack
from src.sources.marketaux import MarketAux
from src.sources.article_extractor import ArticleExtractor
from src.sources.openai import OpenAiAPI

from dotenv import dotenv_values
config = dotenv_values(".env")

# Load environment variables
if not os.getenv("PRODUCTION") or not config['PRODUCTION']:
    config = dotenv_values(".env")
    production = config['PRODUCTION']
else:
    production = os.getenv('PRODUCTION')


class DailyLosers():
    '''
    Daily Losers Strategy
    The strategy is to buy the stocks that are losers that are oversold based on RSI and Bollinger Bands
    and have a bullish sentiment based on news articles and OpenAI sentiment analysis
    '''
    def __init__(self):
        # Initialize the Alpaca API, Yahoo Finance, Slack, MarketAux, ArticleExtractor, and OpenAI
        self.alpaca = AlpacaAPI()
        self.yahoo = Yahoo()
        self.slack = Slack()
        self.marketaux = MarketAux()
        self.article_extractor = ArticleExtractor()
        self.openai = OpenAiAPI()
        
    ########################################################
    # Define the get_buy_opportunities function
    ########################################################
    def get_buy_opportunities(self):
        """
        Get the buy opportunities based on the market losers
        The strategy is to buy the stocks that are losers that are oversold based on RSI and Bollinger Bands
        and have a bullish sentiment based on news articles and OpenAI sentiment analysis
        return: DataFrame of buy opportunities
        """
        # Get the market losers
        market_losers = self.get_ticker_info()

        # Filter the losers based on indicators
        buy_criteria = ((market_losers[['rsi14', 'rsi30', 'rsi50', 'rsi200']] <= 30).any(axis=1)) | \
                          ((market_losers[['bblo14', 'bblo30', 'bblo50', 'bblo200']] == 1).any(axis=1))
        
        buy_filtered_df = market_losers[buy_criteria]
        filtered_list = buy_filtered_df['Symbol'].tolist()

        news_filtered_list = []
        # Filter the losers based on news sentiment from MarketAux API and OpenAI sentiment analysis
        for symbol in filtered_list:
            if self.get_asset_news_articles_sentiment(symbol):
                news_filtered_list.append(symbol)
        
        # Return the filtered DataFrame
        return buy_filtered_df[buy_filtered_df['Symbol'].isin(news_filtered_list)]
    
    ########################################################
    # Define the get_asset_news_articles_sentiment function
    ########################################################
    def get_asset_news_articles_sentiment(self, symbol):
        """
        Get the sentiment of the news articles for the given symbol using MarketAux API and OpenAI sentiment analysis
        :param symbol: Stock symbol
        return: True if the sentiment is bullish
        return: False if the sentiment is bearish
        """
        # Get the news articles for the given symbol
        data = self.marketaux.get_news_articles(symbol=symbol, limit=3)

        bulls = 0
        bears = 0
        
        # If there are no news articles, return False
        if data.empty:
            return False
        else:
            time.sleep(1)
            # Iterate through the news articles and get the sentiment
            for url in data['url']:
                if url == None or url == '':
                    continue    
                # Get the article extracted from the URL
                extracted_article = self.article_extractor.extract_article(url)
                # Get the sentiment of the article
                sentiment = self.openai.get_market_sentiment(title=extracted_article['title'], article=extracted_article['article'])
                # Increment the bulls and bears based on the sentiment
                if sentiment == 'BULLISH':
                    bulls += 1
                elif sentiment == 'BEARISH':
                    bears += 1
            # Return True if the sentiment is bullish
            if bulls > bears:
                return True
            # Return False if the sentiment is bearish
        return False

    ########################################################
    # Define the get_market_losers function
    ########################################################
    def get_market_losers(self):
        """
        Get the market losers from Yahoo Finance
        The function returns the top 100 market losers
        return: List of market losers
        """
        raw_symbols = self.yahoo.get_symbols('https://finance.yahoo.com/losers?offset=0&count=100', 'stock', 100)
        clean_symbols = []
        # Check if the stock is fractionable
        # If the stock is fractionable, add it to the clean symbols list
        for symbol in raw_symbols:
            try:
                # If the symbol has a '-' in it, continue to the next symbol (NOT A STOCK TICKER)
                if '-' in symbol:
                    continue
                if 'Close' in symbol:
                    continue
                asset = self.alpaca.get_asset(symbol)
                if asset.fractionable == True:
                    clean_symbols.append(symbol)
                else:
                    continue
            except Exception as e:
                if not production:
                    print(e)
                else:
                    self.slack.send_message(channel='#app-development', message=f"Error: {e}", username='@messages_from_api')
                continue
        # Return the clean symbols list of market losers that are fractionable
        return clean_symbols

    ########################################################
    # Define the liquidate_positions_for_capital function
    ########################################################
    def liquidate_positions_for_capital(self):
        """
        Liquidate the positions to make cash 10% of the portfolio
        The strategy is to sell the top 25% of performing stocks evenly to make cash 10% of total portfolio
        return: True if the function is successful
        return: False if the market is closed or there are no stocks to sell
        """
        current_positions = self.alpaca.get_current_positions()

        # If no current positions or market is closed, exit the function by returning False
        if current_positions.iloc[0]['asset'] == 'Cash' or self.alpaca.is_market_open() == False:
            if not self.alpaca.is_market_open():
                print("Market is closed")
            else:
                print("No current positions")
            return False

        cash_row = current_positions[current_positions['asset'] == 'Cash']
        total_holdings = current_positions['market_value'].sum()

        if cash_row['market_value'].values[0] / total_holdings < 0.1:
            # Remove the cash row
            curpositions = current_positions[current_positions['asset'] != 'Cash']
            # Sort the positions by profit percentage
            curpositions = curpositions.sort_values(by='profit_pct', ascending=False) 

            # Sell the top 25% of performing stocks evenly to make cash 10% of total portfolio
            top_performers = curpositions.iloc[:int(len(curpositions) // 2)]
            top_performers_market_value = top_performers['market_value'].sum()
            cash_needed = total_holdings * 0.1 - cash_row['market_value'].values[0]

            # Sell the top performers to make cash 10% of the portfolio
            for index, row in top_performers.iterrows():
                print(f"Selling {row['asset']} to make cash 10% portfolio cash requirement")
                # Calculate the amount to sell in USD
                amount_to_sell = int((row['market_value'] / top_performers_market_value) * cash_needed)
                
                # If the amount to sell is 0, continue to the next stock
                if amount_to_sell == 0:
                    continue

                # Market sell the stock
                try:
                    self.alpaca.market_sell(symbol=row['asset'], notional=amount_to_sell)
                except Exception as e:
                    if not production:
                        print(e)
                    else:
                        self.slack.send_message(channel='#app-development', message=f"Error Liquidating:\n {e}", username='@messages_from_api')
                    continue

    ########################################################
    # Define the sell_orders_from_sell_criteria function
    ########################################################
    def sell_orders_from_sell_criteria(self):
        """
        Sell the stocks based on the sell criteria
        The strategy is to sell the stocks that are overbought based on RSI and Bollinger Bands
        return: True if the function is successful
        return: False if the market is closed or there are no stocks to sell
        """
        # Get the current positions
        current_positions = self.alpaca.get_current_positions()

        # If no current positions, exit the function by returning False
        if current_positions.iloc[0]['asset'] == 'Cash':
            print("No current positions")
            return
        
        # Get the tickers from the filtered market losers via the get_ticker_info function
        current_positions_hist = self.get_ticker_info(current_positions[current_positions['asset'] != 'Cash']['asset'].tolist())
        # Find the stocks to sell based on the sell criteria
        sell_criteria = ((current_positions_hist[['rsi14', 'rsi30', 'rsi50', 'rsi200']] >= 70).any(axis=1)) | \
                            ((current_positions_hist[['bbhi14', 'bbhi30', 'bbhi50', 'bbhi200']] == 1).any(axis=1))
        # Filter the stocks to sell
        sell_filtered = current_positions_hist[sell_criteria]

        # Make sure the symbols are in a list
        symbols = sell_filtered['Symbol'].tolist()
        
        # Check if the market is open or if there are no symbols to sell
        if not symbols or self.alpaca.is_market_open() == False:
            # Return False if the market is closed or there are no symbols to sell
            if not symbols:
                print("No stocks to sell")
            else:
                print("Market is closed")
            return
        
        # Iterate through the symbols and sell the stocks
        for symbol in symbols:
            # Get the quantity of the stock
            qty = current_positions[current_positions['asset'] == symbol]['qty'].values[0]
            # Submit a market sell order
            try:
                self.alpaca.market_sell(symbol=symbol, qty=qty)
            except Exception as e:  
                if not production:
                    print(e)
                else:
                    self.slack.send_message(channel='#app-development', message=f"Error Selling:\n {e}", username='@messages_from_api')
                continue

    ########################################################
    # Define the buy_orders function
    ########################################################
    def buy_orders(self):
        """
        Buy the stocks based on the market losers
        The strategy is to buy the stocks that are losers that are oversold based on RSI and Bollinger Bands
        return: True if the function is successful
        return: False if market is closed
        """
        
        # Get the tickers from the get_ticker_info function and convert symbols to a list
        tickers = self.get_buy_opportunities()['Symbol'].tolist()

        # Get the current positions and available cash
        df_current_positions = self.alpaca.get_current_positions()
        available_cash = df_current_positions[df_current_positions['asset'] == 'Cash']['qty'].values[0]

        # Calculate the notional value for each stock
        # Divide the available cash by the number of tickers
        # This is the amount to buy for each stock
        # First few days will create large positions, but will be rebalanced in the future (hopefully :D)
        notional = available_cash / len(tickers)

        # Iterate through the tickers and buy the stocks
        for ticker in tickers:
            # Market buy the stock
            try:
                self.alpaca.market_buy(symbol=ticker, notional=notional)
            except Exception as e:
                if not production:
                    print(e)
                else:
                    self.slack.send_message(channel='#app-development', message=f"Error Buying, {e}", username='@messages_from_api')
                continue

    ########################################################
    # Define the get_ticker_info function
    ########################################################
    def get_ticker_info(self, sym=None):
        """
        Get the ticker information for the given symbol or the market losers
        :param sym: Stock symbol
        Add RSI and Bollinger Bands Indicators to the stock data
        If sym is None, get the market losers
        Otherwise, get the ticker information for the given symbol
        return: DataFrame of ticker information
        """
        # Get the market losers if sym is None
        if sym is None:
            tickers = self.get_market_losers()
        # Otherwise, set tickers to the symbol
        else:
            # If the sym is a list, set sym to tickers
            if isinstance(sym, list):
                tickers = sym
            # Otherwise, set tickers to a list with the symbol
            else:
                tickers = [sym]

        df_tickers = []
        for i, symbol in tqdm(
            enumerate(tickers),
            desc="Processing {loser_count} tickers for trading signals".format(loser_count=len(tickers)),
        ):
            if symbol == '' or symbol == None:
                continue
            try:
                # Get the stock data
                ticker_history = self.alpaca.get_stock_data(symbol)
                # Initialize the RSI and Bollinger Bands Indicators                
                for n in [14, 30, 50, 200]:
                    # Initialize RSI Indicator
                    ticker_history["rsi" + str(n)] = RSIIndicator(
                        close=ticker_history["Close"], window=n, fillna=True
                    ).rsi()
                    # Initialize Bollinger Bands High Indicator
                    ticker_history["bbhi" + str(n)] = BollingerBands(
                        close=ticker_history["Close"], window=n, window_dev=2, fillna=True
                    ).bollinger_hband_indicator()
                    # Initialize Bollinger Bands Low Indicator
                    ticker_history["bblo" + str(n)] = BollingerBands(
                        close=ticker_history["Close"], window=n, window_dev=2, fillna=True
                    ).bollinger_lband_indicator()
                # Get the last 30 days of data
                df_tickers_temp = ticker_history.iloc[-1:, -30:].reset_index(drop=True)
                # Append the dataframe to the list
                df_tickers.append(df_tickers_temp)
            except:
                KeyError
            pass
        
        # Concatenate the dataframes
        df_tickers = [x for x in df_tickers if not x.empty]
        df_tickers = pd.concat(df_tickers)


        return df_tickers