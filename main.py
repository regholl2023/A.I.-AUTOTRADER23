from txcode_alpaca_strategies.strategies.daily_losers import DailyLosers

from datetime import datetime
import pytz

now = datetime.now(tz=pytz.timezone('US/Eastern'))
print("Current time is: ", now.strftime("%m/%d/%Y, %H:%M:%S"))
current_hour = now.hour
current_minute = now.minute
current_day = now.weekday()

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