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
    # Define the liquidate_positions_for_capital function
    ########################################################
    def liquidate_positions_for_capital(self):
        """
        Liquidate the positions to make cash 10% of the portfolio
        The strategy is to sell the top 25% of performing stocks evenly to make cash 10% of total portfolio
        return: True if the function is successful
        return: False if the market is closed or there are no stocks to sell
        """
        current_positions = self.get_current_positions()

        # If no current positions or market is closed, exit the function by returning False
        if current_positions.iloc[0]['asset'] == 'Cash' or self.is_market_open() == False:
            if not self.is_market_open():
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

                # Try to sell the stock
                try:
                    # Submit a market sell order
                    self.market_sell(symbol=row['asset'], notional=amount_to_sell)
                # If there is an exception, print the exception
                except Exception as e:
                    print(e)
                    continue
                # Print the message if the order is successful
                else:
                    print(f"Sold {amount_to_sell} of {row['asset']} at market price")

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
        current_positions = self.get_current_positions()

        # If no current positions, exit the function by returning False
        if current_positions.iloc[0]['asset'] == 'Cash':
            print("No current positions")
            return False
        
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
        if not symbols or self.is_market_open() == False:
            # Return False if the market is closed or there are no symbols to sell
            if not symbols:
                print("No stocks to sell")
            else:
                print("Market is closed")
            return False
        
        # Iterate through the symbols and sell the stocks
        for symbol in symbols:
            try:
                # Get the quantity of the stock
                qty = current_positions[current_positions['asset'] == symbol]['qty'].values[0]
                # Submit a market sell order
                self.market_sell(symbol=symbol, qty=qty)
            # If there is an exception, print the exception
            except Exception as e:
                print(e)
                continue
            # Print the message if the order is successful
            else:
                print(f"Sold {qty} of {symbol} at market price")
        
        return True


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
        # Check if the market is open
        if not self.is_market_open():
            print("Market is closed")
            # Return False if the market is closed
            return False
        
        # Get the tickers from the get_ticker_info function and convert symbols to a list
        tickers = self.get_ticker_info()['Symbol'].tolist()

        # Get the current positions and available cash
        df_current_positions = self.get_current_positions()
        available_cash = df_current_positions[df_current_positions['asset'] == 'Cash']['qty'].values[0]

        # Calculate the notional value for each stock
        # Divide the available cash by the number of tickers
        # This is the amount to buy for each stock
        # 10% of the available cash is kept for future rebalancing
        # First few days will create large positions, but will be rebalanced in the future (hopefully :D)
        notional = (available_cash - (available_cash * 0.1)) / len(tickers)

        # Iterate through the tickers and buy the stocks
        for ticker in tickers:
            try:
                # Submit a market buy order
                self.market_buy(symbol=ticker, notional=notional)
            # If there is an exception, print the exception
            except Exception as e:
                print(e)
                continue
            # Print the message if the order is successful
            else:
                print(f"Bought {ticker} at ${notional}")

        return True

    ########################################################
    # Define the get_current_positions function
    ########################################################
    def get_current_positions(self):
        """
        Get the current positions from Alpaca API, including cash
        Probably a better way to do this, but so far this is the cleanest way to work with this stategy
        return: DataFrame of current positions
        """
        # Create a TradingClient object
        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)
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
    # Define the is_market_open function
    ########################################################
    def is_market_open(self):
        """
        Check if the market is open
        return: True if the market is open
        return: False if the market is closed
        """
        # Create a TradingClient object
        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)
        # Get the clock
        clock = client.get_clock()
        # Return True if the market is open
        return clock.is_open
    
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
        
        buy_filtered_df = df_tickers[buy_criteria]

        self.buy_tickers = buy_filtered_df['Symbol'].tolist()
        
        # Filter and return the tickers based on the buy criteria
        return buy_filtered_df

    ########################################################
    # Define the get_stock_data function
    ########################################################
    def get_stock_data(self, symbol):
        """
        Get the stock data for a given symbol from Alpaca API
        :param symbol: Stock symbol
        Pull data from StockHistoricalDataClient and clean and rename the columns
        so that it can be used for technical analysis
        return: DataFrame of stock data
        """
        # Create a StockHistoricalDataClient object
        client = StockHistoricalDataClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)

        dt = datetime.now()
        # Create a StockBarsRequest object
        history_request = StockBarsRequest(
            symbol_or_symbols=symbol,                       # Single symbol
            timeframe=TimeFrame.Day,                        # 1 day timeframe
            start=datetime(dt.year - 1, dt.month, dt.day),  # 1 year ago
            end=datetime(dt.year, dt.month, dt.day),        # Today
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
        """
        Get the market losers from Alpaca API ScreenerClient
        The function returns the top 50 market losers
        Which can be used to filter the stocks based on the losers
        The strategy is to buy the stocks that are losers that are oversold based on RSI and Bollinger Bands
        return: List of market losers
        """
        # Create a TradingClient object
        client = ScreenerClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)
        # Get the losers
        params = MarketMoversRequest(
            top=50,
            market_type=MarketType.STOCKS,
        )
        losers = client.get_market_movers(params)
        losers_list = []
        # Iterate through the losers and append the symbol to the list
        for sym in losers.losers:
            losers_list.append(sym.symbol)
        # Return list of losers
        return losers_list
    
    ########################################################
    # Market buy function
    ########################################################
    def market_buy(self, symbol, notional=None, qty=None):
        """
        Submit a market buy order to Alpaca API
        :param symbol: Stock symbol
        :param notional: Amount to buy in USD
        :return: True if the order is successful
        :return: False if the order is unsuccessful
        """
        # Create a TradingClient object
        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key, paper=alpaca_paper)
        # Create a MarketOrderRequest object
        order_data = MarketOrderRequest(
                symbol=symbol,
                notional=notional if notional else None,
                qty=qty if qty else None,
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
    def market_sell(self, symbol, qty=None, notional=None):
        """
        Submit a market sell order to Alpaca API
        :param symbol: Stock symbol
        :param qty: Quantity to sell
        :return: True if the order is successful
        :return: False if the order is unsuccessful
        """
        # Create a TradingClient object
        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key, paper=alpaca_paper)
        # Create a MarketOrderRequest object
        order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty if qty else None,
                notional=notional if notional else None,
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
    