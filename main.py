from alpaca_only_api.alpaca import *

def main():
    # Set the Alpaca API key and secret key
    alpaca = AlpacaAPI()
    # Get the market losers
    print(alpaca.get_ticker_info(sym="AAPL"))
    

if __name__ == "__main__":
    main()