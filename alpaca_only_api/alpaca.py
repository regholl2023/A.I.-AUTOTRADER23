import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from alpaca.data import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.enums import MarketType
from alpaca.data.requests import MarketMoversRequest, StockBarsRequest

from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from tabulate import tabulate

# Load environment variables
if not os.getenv("PRODUCTION"):
    from dotenv import dotenv_values
    config = dotenv_values(".env")
    alpaca_key_id = config['ALPACA_KEY_ID']
    alpaca_secret_key = config['ALPACA_SECRET_KEY']
    alpaca_paper = config['ALPACA_PAPER']
else:
    alpaca_key_id = os.getenv('ALPACA_KEY_ID')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
    alpaca_paper = os.getenv('ALPACA_PAPER')

# Define the AlpacaAPI class
class AlpacaAPI:
    def __init__(self):
        # Set the Alpaca API key and secret key
        self.alpaca_key_id = alpaca_key_id
        self.alpaca_secret_key = alpaca_secret_key
    
    ########################################################
    # Define the get_ticker_info function
    ########################################################
    def get_ticker_info(self, sym=None):
        # Get the market losers if sym is None
        if sym is None:
            tickers = self.get_market_losers()
        # Otherwise, set tickers to the symbol
        else:
            tickers = [sym]

        df_tickers = []
        for i, symbol in tqdm(
            enumerate(tickers),
            desc="Processing {loser_count} tickers for trading signals".format(loser_count=len(tickers)),
        ):
            try:
                # Get the stock data
                ticker_history = self.get_stock_data(symbol)
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
            except Exception as e:
                print(e)
            pass
        
        # Concatenate the dataframes
        df_tickers = [x for x in df_tickers if not x.empty]
        df_tickers = pd.concat(df_tickers)

        # Return the ticker if sym is not None
        if sym is not None:
            return df_tickers

        # Filter the losers based on indicators
        buy_criteria = ((df_tickers[['rsi14', 'rsi30', 'rsi50', 'rsi200']] <= 30).any(axis=1)) | \
                          ((df_tickers[['bblo14', 'bblo30', 'bblo50', 'bblo200']] == 1).any(axis=1))
        
        # Filter and return the tickers based on the buy criteria
        return df_tickers[buy_criteria]

    ########################################################
    # Define the get_stock_data function
    ########################################################
    def get_stock_data(self, symbol):
        # Create a StockHistoricalDataClient object
        client = StockHistoricalDataClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)

        dt = datetime.now()
        # Create a StockBarsRequest object
        history_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime(dt.year - 1, dt.month, dt.day),
            end=datetime(dt.year, dt.month, dt.day),
        )
        bar_df = client.get_stock_bars(history_request)
        # Reset the index
        bar_df = bar_df.df.reset_index()
        # Rename the columns
        bar_df.rename(columns={'symbol': 'Symbol', 'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        # Drop the uneeded columns
        bar_df.drop(columns=['Volume', 'trade_count', 'vwap'], inplace=True)
        return bar_df
    
    ########################################################
    # Define the get_market_losers function
    ########################################################
    def get_market_losers(self):
        # Create a TradingClient object
        client = ScreenerClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)
        # Get the losers
        params = MarketMoversRequest(
            top=50,
            market_type=MarketType.STOCKS,
        )
        losers = client.get_market_movers(params)
        df_losers = []
        # Iterate through the losers and append the symbol to the list
        for sym in losers.losers:
            df_losers.append(sym.symbol)
        # Return the losers
        return df_losers
    
    ########################################################
    # Market buy function
    ########################################################
    def market_buy(self, symbol, notional):
        # Create a TradingClient object
        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key, paper=alpaca_paper)
        # Create a MarketOrderRequest object
        order_data = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
        # Try to submit the order
        try: 
            # Submit the order to the Alpaca API
            client.submit_order(order_data)
        # If there is an exception, print the exception
        except Exception as e:
            print(e)
            return False
        # Return True if the order is successful
        return True
    
    ########################################################
    # Market sell function
    ########################################################
    def market_sell(self, symbol, qty):
        # Create a TradingClient object
        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key, paper=alpaca_paper)
        # Create a MarketOrderRequest object
        order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        # Try to submit the order
        try: 
            # Submit the order to the Alpaca API
            client.submit_order(order_data)
        # If there is an exception, print the exception
        except Exception as e:
            print(e)
            return False
        # Return True if the order is successful
        return True
    