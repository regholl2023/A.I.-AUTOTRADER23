import os
import json
import pandas as pd

import requests

from datetime import datetime

from alpaca.common.exceptions import APIError

from alpaca.data import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.requests import StockBarsRequest

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
production = os.getenv('PRODUCTION')
alpaca_key_id = os.getenv('ALPACA_KEY_ID')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
alpaca_paper = os.getenv('ALPACA_PAPER')

# Define the AlpacaAPI class
class AlpacaAPI:
    """
    Alpaca API class to interact with Alpaca API
    """
    def __init__(self):
        pass

    def get_latest_news(self, symbol, limit=3):
        url = "https://data.alpaca.markets/v1beta1/news?symbols={}".format(symbol)
        headers = {
            "Apca-Api-Key-Id": alpaca_key_id,
            "Apca-Api-Secret-Key": alpaca_secret_key
        }
        response = requests.get(url, headers=headers)

        response_json = json.loads(response.text)

        print(json.dumps(response_json, indent=2))

    ########################################################
    # Define the submit_order function
    ########################################################
    def stop_limit_order(self, symbol, stop_price, limit_price, qty=None, notional=None, side='buy', time_in_force='day'):
        """
        Submit a stop limit order to Alpaca API
        :param symbol: Stock symbol
        :param stop_price: Price to buy or sell
        :param limit_price: Price to buy or sell
        :param qty: Quantity to buy or sell
        :param notional: Amount to buy or sell in USD
        :param side: buy or sell, default is buy
        :param time_in_force: day or gtc, default is day
        :return: True if the order is successful
        :return: False if the order is unsuccessful
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key, paper=alpaca_paper)
        # Create a StopLimitOrderRequest object
        order_data = StopLimitOrderRequest(
            symbol=symbol,
            qty=qty if qty else None,
            notional=round(notional, 2) if notional else None,
            stop_price=stop_price,
            limit_price=limit_price,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force= TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
        )
        # Try to submit the order
        try:
            # Submit the order to the Alpaca API
            client.submit_order(order_data)
        # If there is an exception, print the exception
        except APIError as e:
            formatted_text = json.dumps(e, indent = 2)
            raise Exception(f"stop_limit_order Error:\n {formatted_text}")

    ########################################################
    # Define the stop_order function
    ########################################################
    def stop_order(self, symbol, stop_price, qty=None, notional=None, side='buy', time_in_force='day'):
        """
        Submit a stop order to Alpaca API
        :param symbol: Stock symbol
        :param stop_price: Price to buy or sell
        :param qty: Quantity to buy or sell
        :param notional: Amount to buy or sell in USD
        :param side: buy or sell, default is buy
        :param time_in_force: day or gtc, default is day
        :return: True if the order is successful
        :return: False if the order is unsuccessful
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key, paper=alpaca_paper)
        # Create a StopOrderRequest object
        order_data = StopOrderRequest(
            symbol=symbol,
            qty=qty if qty else None,
            notional=round(notional, 2) if notional else None,
            stop_price=stop_price,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force= TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
        )
        # Try to submit the order
        try:
            # Submit the order to the Alpaca API
            client.submit_order(order_data)
        # If there is an exception, print the exception
        except APIError as e:
            formatted_text = json.dumps(e, indent = 2)
            raise Exception(f"stop_order Error:\n {formatted_text}")
    
    ########################################################
    # Define the limit_order function
    ########################################################
    def limit_order(self, symbol, limit_price, qty=None, notional=None, side='buy', time_in_force='day'):
        """
        Submit a limit order to Alpaca API
        :param symbol: Stock symbol
        :param limit_price: Price to buy or sell
        :param qty: Quantity to buy or sell
        :param notional: Amount to buy or sell in USD
        :param side: buy or sell, default is buy
        :param time_in_force: day or gtc, default is day
        :return: True if the order is successful
        :return: False if the order is unsuccessful
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key, paper=alpaca_paper)
        # Create a LimitOrderRequest object
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty if qty else None,
            notional=round(notional, 2) if notional else None,
            limit_price=limit_price,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force= TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
        )
        # Try to submit the order
        try:
            # Submit the order to the Alpaca API
            client.submit_order(order_data)
        # If there is an exception, print the exception
        except APIError as e:
            formatted_text = json.dumps(e, indent = 2)
            raise Exception(f"limit_order Error:\n {formatted_text}")

    ########################################################
    # Market Order function
    ########################################################
    def market_order(self, symbol, notional=None, qty=None, side='buy', time_in_force='day'):
        """
        Submit a market order to Alpaca API
        :param symbol: Stock symbol
        :param notional: Amount to buy or sell in USD
        :param qty: Quantity to buy or sell
        :param side: buy or sell, default is buy
        :param time_in_force: day or gtc, default is day
        :return: True if the order is successful
        :return: False if the order is unsuccessful
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key, paper=alpaca_paper)
        # Create a MarketOrderRequest object
        order_data = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2) if notional else None,
                qty=qty if qty else None,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC
            )
        # Try to submit the order
        try: 
            # Submit the order to the Alpaca API
            client.submit_order(order_data)
        # If there is an exception, print the exception
        except APIError as e:
            formatted_text = json.dumps(e, indent = 2)
            raise Exception(f"market_order Error Buying:\n {formatted_text}")
    
    ########################################################
    # Define the get_current_positions function
    ########################################################
    def get_current_positions(self):
        """
        Get the current positions from Alpaca API
        Modify the DataFrame to include the cash position
        return: DataFrame of current positions
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key)
        # Check if there are no positions, return an DataFrame with 0 values
        if client.get_all_positions() == []:
            investments = pd.DataFrame({
            'asset': '',
            'current_price': 0,
            'qty': 0,
            'market_value': 0,
            'profit_dol': 0,
            'profit_pct': 0
        }, index=[0])
        # Otherwise, get the current positions, and return the DataFrame
        else:
            investments = pd.DataFrame({
                'asset': [x.symbol for x in client.get_all_positions()],
                'current_price': [x.current_price for x in client.get_all_positions()],
                'qty': [x.qty for x in client.get_all_positions()],
                'market_value': [x.market_value for x in client.get_all_positions()],
                'profit_dol': [x.unrealized_pl for x in client.get_all_positions()],
                'profit_pct': [x.unrealized_plpc for x in client.get_all_positions()]
            })
        # Set the cash position for easy calculation
        cash = pd.DataFrame({
            'asset': 'Cash',
            'current_price': client.get_account().cash,
            'qty': client.get_account().cash,
            'market_value': client.get_account().cash,
            'profit_dol': 0,
            'profit_pct': 0
        }, index=[0])  # Need to set index=[0] since passing scalars in df

        # Concatenate the investments and cash DataFrames
        assets = pd.concat([investments, cash], ignore_index=True)

        # Drop the empty row if the asset is empty
        if assets['asset'][0] == '':
            assets = assets.drop(0)
            assets.reset_index(drop=True, inplace=True)

        # Format the DataFrame
        float_fmt = ['current_price', 'qty', 'market_value', 'profit_dol', 'profit_pct']
        str_fmt = ['asset']
        # Convert the columns to the correct data types
        for col in float_fmt:
            assets[col] = assets[col].astype(float)
        # Convert the columns to the correct data types
        for col in str_fmt:
            assets[col] = assets[col].astype(str)

        rounding_2 = ['market_value', 'profit_dol']
        rounding_4 = ['profit_pct']
        # Round the columns to 2 decimal places
        assets[rounding_2] = assets[rounding_2].apply(lambda x: pd.Series.round(x, 2))
        # Round the columns to 4 decimal places
        assets[rounding_4] = assets[rounding_4].apply(lambda x: pd.Series.round(x, 4))
        # Calculate the portfolio total
        asset_sum = assets['market_value'].sum()
        # Set the portfolio percentage
        assets['portfolio_pct'] = assets['market_value'] / asset_sum
        # Return the DataFrame
        return assets
    
    ########################################################
    # Define the get_asset function
    ########################################################
    def get_asset(self, symbol):
        """
        Get the asset information from Alpaca API
        :param symbol: Stock symbol
        return: Asset information
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key)
        try:
            # Get the asset information
            asset = client.get_asset(symbol)
        except APIError as e:
            formatted_text = json.dumps(e, indent = 2)
            raise Exception(f"get_asset() Error Getting Asset:\n {formatted_text}")
        # Return the asset information
        return asset

    ########################################################
    # Define the is_market_open function
    ########################################################
    def is_market_open(self):
        """
        Check if the market is open
        return: True if the market is open
        return: False if the market is closed
        """
        # Create a TradingClient object
        client = TradingClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key)
        # Get the clock
        clock = client.get_clock()
        # Return True if the market is open
        return clock.is_open

    ########################################################
    # Define the get_stock_data function
    ########################################################
    def get_stock_data(self, symbol, timeframe='day', start_date=datetime(datetime.now().year - 1, datetime.now().month, datetime.now().day), end_date=datetime(datetime.now().year, datetime.now().month, datetime.now().day)):
        """
        Get the stock data from Alpaca API
        :param symbol: Stock symbol
        :param timeframe: Timeframe for the stock data, default is day
        :param start_date: Start date for the stock data, default is 1 year ago
        :param end_date: End date for the stock data, default is today
        return: DataFrame of stock data
        """
        # Create a StockHistoricalDataClient object
        client = StockHistoricalDataClient(api_key=alpaca_key_id, secret_key=alpaca_secret_key)

        match timeframe:
            case 'day':
                timeframe = TimeFrame.Day
            case 'minute':
                timeframe = TimeFrame.Minute
            case 'hour':
                timeframe = TimeFrame.Hour
            case 'week':
                timeframe = TimeFrame.Week
            case 'month':
                timeframe = TimeFrame.Month
            case _:
                timeframe = TimeFrame.Day            

        # Create a StockBarsRequest object
        history_request = StockBarsRequest(
            symbol_or_symbols=symbol,                       # Single symbol
            timeframe=timeframe,                        # 1 day timeframe
            start=start_date,  # 1 year ago
            end=end_date,      # Today
        )

        try:
            bar_df = client.get_stock_bars(history_request)
            # Reset the index
            bar_df = bar_df.df.reset_index()
            # Rename the columns
            bar_df.rename(columns={'symbol': 'Symbol', 'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            # Return the DataFrame
            return bar_df
        except APIError as e:
            formatted_text = json.dumps(e, indent = 2)
            raise Exception(f"get_stock_data Error Getting Stock Data:\n {formatted_text}")