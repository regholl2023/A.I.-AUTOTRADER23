from alpaca_only_api.alpaca import *

def main():
    # Set the Alpaca API key and secret key
    alpaca = AlpacaAPI()

    # Check for sell orders based on sell criteria
    alpaca.sell_orders_from_sell_criteria()
    # Check account for capital and liquidate positions if needed
    alpaca.liquidate_positions_for_capital()
    # Check for buy orders based on buy criteria
    alpaca.buy_orders()

    alpaca.send_slack_message("Heroku App is running!")
    
if __name__ == "__main__":
    main()