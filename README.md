# Alpaca Only API

## THIS PROJECT IS UNDER DEVELOPMENT

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/alpaca-only-api/blob/main/LICENSE)

## Description

The Alpaca Only API is a Python project that provides a way to automate trades using only data provided by the AlpacaAPI other than the market losers, which are pulled from yahoo finance. This is currently in development and is in no way to be trusted to be a functioning bot. As it currently stands this script should be once a day and it picks the losers from previous day to filter buy picks using ta indicators. Also pulls articles from https://www.marketaux.com and using openAi to give sentiment analysis of article content, to help pick stocks more likely to rise after losing.

I am currently running this script on Heroku, using Advanced Scheduler to run it at various times. Not set times yet. I am experimenting with run times, so the main.py will change often.

This is all just a test strategy, but many of the functions could be helpful for those just trying out Alpacas Python API

Currently testing functionality in a paper account. 

I will be updating daily and testing using paper account until it is proven to work.

The API's used for this strategy are listed below. Each service offers free or really low price options. But for the script and strategy to function as it stands, you will need API keys from each of them.

- [Alpaca API](https://alpaca.markets/)
- [MarketAux](https://www.marketaux.com/)
- [ArticleExtractor](https://www.articlextractor.com/)
- [OpenAi](https://platform.openai.com)

## ONLY USE THIS SCRIPT ON ALPACA PAPER ACCOUNTS
I just started developing this on 4/28/2024 and have no idea as of yet if this strategy is at all profitable. Alpaca does not have a true sandbox enviroment, so my testing of functionality is running the script during the day. And looking over logs in the evening for errors.

## Features

- Retrieve market data and interact with api from [Alpaca API](https://alpaca.markets/)
- Pulls previous day losers from Yahoo Finance 
- Gets recent articles from [MarketAux](https://www.marketaux.com/) API
- Extracts article text using [ArticleExtractor](https://www.articlextractor.com/) API
- Gets article sentiment using [OpenAi](https://platform.openai.com) API
- Place buy and sell orders based on the strategy buy/sell criterias

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/alpaca-only-api.git
    ```

2. Install the required dependencies:

    ```bash
    poetry install
    ```
    or

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Create a .env file in the root directory with the following values:

    ```properties
    PRODUCTION=False
    ALPACA_KEY_ID=your_alpaca_api_key
    ALPACA_SECRET_KEY=your_alpaca_secret_key
    ALPACA_PAPER=True
    OPENAI_API_KEY=your_open_ai_key
    MARKETAUX_API_KEY=your_market_aux_api_key
    ARTICLEEXTRACT_API_KEY=your_article_extractor_api_key
    SLACK_ACCESS_TOKEN=your_slack_app_api_key
    ```

2. Run the script

    ```bash
    python main.py
    ```
    The main.py file will change daily possibly, depending on what I am testing out. There are 3 main functions for this strategy to operate.

    1. The sell based on criteria function will search through your current positions and look for sell opportunities based on the stratigies criteria.
    ```python
    # Check for sell orders based on sell criteria
    alpaca.sell_orders_from_sell_criteria()
    ```

    2. The liquidate positions for capital function, will sell some quantities of assets to provide enough capital for next buying opportunities. Should be ran before running the buy function.
    ```python
    # Check account for capital and liquidate positions if needed
    alpaca.liquidate_positions_for_capital()
    ```

    3. The buy orders functions, searches for previous day losers and filters through to find and purchase stocks based on ta indicators and article sentiment analysis
    ```python
    # Check for buy orders based on buy criteria
    alpaca.buy_orders()
    ```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/your-username/alpaca-only-api/blob/main/LICENSE).