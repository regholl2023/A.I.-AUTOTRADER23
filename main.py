from alpaca_only_api.alpaca import *

from datetime import datetime
import pytz
now = datetime.now(tz=pytz.timezone('US/Eastern'))
print("now time is: ", now.strftime("%m/%d/%Y, %H:%M:%S"))
current_hour = now.hour
current_minute = now.minute

def main():
    # Set the Alpaca API key and secret key
    alpaca = AlpacaAPI()
    
    if current_hour == 9 and current_minute < 50:
        # Check for sell orders based on sell criteria
        alpaca.sell_orders_from_sell_criteria()
        # Check account for capital and liquidate positions if needed
        alpaca.liquidate_positions_for_capital()
        # Check for buy orders based on buy criteria
        alpaca.buy_orders()
    else:
        # Check for sell orders based on sell criteria
        alpaca.sell_orders_from_sell_criteria()
    
if __name__ == "__main__":
    main()