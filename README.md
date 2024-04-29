# Alpaca Only API

## THIS PROJECT IS UNDER DEVELOPMENT

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/alpaca-only-api/blob/main/LICENSE)

## Description

The Alpaca Only API is a Python project that provides a simple and intuitive interface to interact with the Alpaca API. It allows users to access market data, place trades, and manage their Alpaca accounts programmatically.

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