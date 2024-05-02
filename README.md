# Alpaca Only API

## THIS PROJECT IS UNDER DEVELOPMENT

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/alpaca-only-api/blob/main/LICENSE)

## Description

The Alpaca Only API is a Python project that provides a way to automate trades using only data provided by the AlpacaAPI other than the market losers, which are pulled from yahoo finance. This is currently in development and is in no way to be trusted to be a functioning bot. As it currently stands this script should be once a day and it picks the losers from previous day to filter buy picks using ta indicators. Also pulls articles from https://www.marketaux.com and using openAi to give sentiment analysis of article content, to help pick stocks more likely to rise after losing.

I am currently running this script on Heroku, using Advanced Scheduler to run it at market open. Also I run it at 11AM and then again at 2:30PM but only to look for sell opportunities.

This is all just a test strategy, but many of the functions could be helpful for those just trying out Alpacas Python API

Currently testing functionality in a paper account. 

I will be updating daily and testing using paper account until it is proven to work.



Most of the text in this README is auto generated, and is not a real description. More updates to come

## Features

- Retrieve real-time market data
- Place buy and sell orders
- Manage account information
- Monitor portfolio performance

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/alpaca-only-api.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Import the `alpaca` module:

    ```python
    import alpaca
    ```

2. Initialize the Alpaca API client:

    ```python
    client = alpaca.AlpacaClient(api_key='YOUR_API_KEY', api_secret='YOUR_API_SECRET')
    ```

3. Start using the API methods:

    ```python
    # Get account information
    account_info = client.get_account()

    # Get market data for a specific symbol
    market_data = client.get_market_data(symbol='AAPL')

    # Place a buy order
    order = client.place_order(symbol='AAPL', qty=10, side='buy', type='market')
    ```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/your-username/alpaca-only-api/blob/main/LICENSE).