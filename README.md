# TxCode Alpaca Strategies

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
- Gets recent articles using [MarketAux API](https://www.marketaux.com/) 
- Extracts article text using [ArticleExtractor API](https://www.articlextractor.com/) 
- Gets article sentiment using [OpenAi API](https://platform.openai.com) 
- Place buy and sell orders based on the strategy buy/sell criterias

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/TexasCoding/txcode-alpaca-strategies.git
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
2. I split this project into multiple modules to make future strategies easier to create. Currently I am just working on the DailyLosers class and working out any issues. Below is how my current main.py is setup. This and much more will change in the coming days. Most likely the folder structure/file names/etc.. will change until I have it setup completely.

```python
from txcode_alpaca_strategies.strategies.daily_losers import DailyLosers

from datetime import datetime
import pytz
now = datetime.now(tz=pytz.timezone('US/Eastern'))
print("Current time is: ", now.strftime("%m/%d/%Y, %H:%M:%S"))
current_hour = now.hour
current_minute = now.minute

    def main():
        # Set the Alpaca API key and secret key
        daily_losers = DailyLosers()

        # Check if the current time is before 9:50 AM
        # If it is, check for sell orders, liquidate positions, and check for buy orders
        if current_hour > 9 and current_hour <= 14:
            # Check for sell orders based on sell criteria
            daily_losers.sell_orders_from_sell_criteria()
        # Check if the current time is between 9:50 AM and 3:30 PM
        # If it is, check for only sell orders
        elif current_hour == 15:
            # Check for sell orders based on sell criteria
            daily_losers.sell_orders_from_sell_criteria()
            # Check account for capital and liquidate positions if needed
            daily_losers.liquidate_positions_for_capital()
        else:
            # Check for buy orders based on buy criteria
            daily_losers.buy_orders()
        
if __name__ == "__main__":
    main()
```

2. Run the script

    ```bash
    python main.py
    ```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/your-username/alpaca-only-api/blob/main/LICENSE).
