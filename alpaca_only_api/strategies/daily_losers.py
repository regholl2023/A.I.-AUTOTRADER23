class DailyLosers:
        ########################################################
    # Define the raw_get_daily_info function
    ########################################################
    def raw_get_daily_info(self, site):
        """
        Get the daily information from a given site
        :param site: Site URL
        Use the requests_html library to get the tables from the site
        return: DataFrame of daily information
        """
        # Create a HTMLSession object
        session = HTMLSession()
        response = session.get(site)
        # Get the tables from the site
        tables = pd.read_html(response.html.raw_html)
        df = tables[0].copy()
        df.columns = tables[0].columns
        # Close the session
        session.close()
        return df
    ########################################################
    # Define the get_market_losers function
    ########################################################
    def get_market_losers(self):
        """
        Get the market losers from Yahoo Finance
        The function returns the top 100 market losers
        return: List of market losers
        """
        df_stock = self.raw_get_daily_info('https://finance.yahoo.com/losers?offset=0&count=100')
        df_stock["asset_type"] = "stock"
        df_stock = df_stock.head(100)

        df_opportunities = pd.concat([df_stock], axis=0).reset_index(drop=True)
        raw_symbols = list(df_opportunities['Symbol'])

        client = TradingClient(api_key=self.alpaca_key_id, secret_key=self.alpaca_secret_key)

        clean_symbols = []
        # Check if the stock is fractionable
        # If the stock is fractionable, add it to the clean symbols list
        for symbol in raw_symbols:
            try:
                # If the symbol has a '-' in it, continue to the next symbol (NOT A STOCK TICKER)
                if '-' in symbol:
                    continue
                asset = client.get_asset(symbol)
                if asset.fractionable == True:
                    clean_symbols.append(symbol)
                else:
                    continue
            except APIError as e:
                if not production:
                    print(e)
                else:
                    formatted_text = json.dumps(e, indent = 2)
                    self.send_slack_message(f"Error Getting Asset: {formatted_text}")
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

                if not self.market_sell(symbol=row['asset'], notional=amount_to_sell):
                    continue
                else:
                    self.send_slack_message(f"Liquidated {row['asset']} to make cash 10% portfolio cash requirement")

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