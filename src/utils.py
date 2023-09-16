#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import matplotlib.dates as mdates


def stock_trading_strategy(stock_price_df, start_date, end_date, initial_cash, investment, buy_threshold=0.05, sell_threshold=0.05, multiplier=1):
    # initialize variables
    num_shares = []
    total_cash = []
    num_shares_purchased = []
    num_shares_sold = []
    batches = []
    action = []
    action_date = []
    stock_price = []
    purchase_sell_action = []
    curr_cash = initial_cash
    # loop through each day in the dataframe
    stock_action_df = stock_price_df[stock_price_df['date'].between(start_date, end_date)]
    initial_price = stock_action_df.iloc[0]['daily_price']
    for i in range(len(stock_action_df)):
        # get the current price and date
        curr_price = stock_action_df.iloc[i]['daily_price']
        curr_date = stock_action_df.iloc[i]['date']
        action_date.append(curr_date)
        stock_price.append(curr_price)

        # check if we should purchase, sell, or pass
        if len(batches) == 0 or curr_price <= batches[-1][0] * (1-buy_threshold):
            if curr_price > initial_price:
                # pass
                action.append('Pass')
                num_shares_purchased.append(0)
                num_shares_sold.append(0)
                total_cash.append(total_cash[-1])
                num_shares.append(num_shares[-1])
            else:
                if len(purchase_sell_action) == 0:
                    pass
                elif purchase_sell_action[-1] == 'Purchase':
                    # increase investment amount at each purchase if the price keeps dropping
                    investment = investment * multiplier
                elif purchase_sell_action[-1] == 'Sell' and purchase_sell_action[-2] == 'Sell':
                    # reset investment amount as price begins to increase
                    investment = investment / multiplier
                else:
                    pass
                # purchase
                num_shares_to_purchase = min(curr_cash, investment) / curr_price
                if len(num_shares) == 0:
                    num_shares.append(num_shares_to_purchase)
                else:
                    num_shares.append(num_shares[-1] + num_shares_to_purchase)
                if len(total_cash) == 0:
                    total_cash.append(initial_cash - min(curr_cash, investment))
                else:
                    total_cash.append(total_cash[-1] - min(curr_cash, investment))
                batches.append((curr_price, num_shares_to_purchase))
                action.append('Purchase')
                purchase_sell_action.append('Purchase')
                num_shares_purchased.append(num_shares_to_purchase)
                num_shares_sold.append(0)
                curr_cash -= min(curr_cash, investment)
        elif curr_price >= batches[-1][0] * (1+sell_threshold):
            # sell
            num_shares_to_sell = batches[-1][1]
            num_shares.append(num_shares[-1] - num_shares_to_sell)
            total_cash.append(total_cash[-1] + num_shares_to_sell * curr_price)
            batches.pop()
            # this batch is sold
            action.append('Sell')
            purchase_sell_action.append('Sell')
            num_shares_purchased.append(0)
            num_shares_sold.append(num_shares_to_sell)
            curr_cash += num_shares_to_sell * curr_price
        else:
            # pass
            action.append('Pass')
            num_shares_purchased.append(0)
            num_shares_sold.append(0)
            total_cash.append(total_cash[-1])
            num_shares.append(num_shares[-1])

        
        # add the results to a new dataframe row
        results_df = pd.DataFrame({
            'date': action_date,
            'daily_price':stock_price,
            'action': action,
            'num_shares_purchased': num_shares_purchased,
            'purchase_value':np.array(stock_price)*np.array(num_shares_purchased),
            'num_shares_sold': num_shares_sold,
            'sell_value':np.array(stock_price)*np.array(num_shares_sold),
            'total_num_shares': num_shares,
            'total_stock_value':np.array(stock_price)*np.array(num_shares),
            'total_cash': total_cash
        })
        
        # concatenate the new row to the overall results dataframe
    return results_df

def stock_trading_strategy_supp_resist(df,
                             targeted_purchase_prices,
                             targeted_sell_prices,
                             initial_cash=10000,
                             investment=1000,
                             buy_threshold=0.05,
                             sell_threshold=0.1,
                            ):
    cash = initial_cash
    stock_units = 0
    purchase_record = []
    total_cash_records = []
    total_num_shares_records = []
    action_records = []
    action_price_records = []
    action_unit_records = []
    
    for index, row in df.iterrows():
        date = row['date']
        price = row['daily_price']
        
        # Check if there's no stock, and then evaluate whether to execute a purchase
        if len(purchase_record) == 0:
            # Check if the price is higher than the maximum targeted purchase price
            if price > max(targeted_purchase_prices):
                action = 'Pass'
                action_price = 0
                action_unit = 0
            else:
                # Find the purchase price
                purchase_price = min([e for e in targeted_purchase_prices if e > price])
                investment = min(cash, investment)
                purchase_units = investment / purchase_price
                purchase_record.append((purchase_price, purchase_units))
                cash -= investment
                stock_units += purchase_units
                action = 'Purchase'
                action_price = purchase_price
                action_unit = purchase_units
        
        else:
            # Now there are stock available, evaluate whether to sell or to purchase
            # compare stock price with the latest purchase batch that is not sold
            if any(e > price for e in targeted_purchase_prices if e < purchase_record[-1][0]*(1-buy_threshold)):
                investment = min(cash, investment)
                if investment > 0:
                    purchase_price = max([e for e in targeted_purchase_prices if e < purchase_record[-1][0]*(1-buy_threshold)])
                    purchase_units = investment / purchase_price
                    purchase_record.append((purchase_price, purchase_units))
                    cash -= investment
                    stock_units += purchase_units
                    action = 'Purchase'
                    action_price = purchase_price
                    action_unit = purchase_units
                else:
                    action = 'Pass'
                    action_price = 0
                    action_unit = 0
            elif any(e < price for e in targeted_sell_prices if e > purchase_record[-1][0]*(1+sell_threshold)):
                if len(purchase_record) > 0:
                    sell_price = min([e for e in targeted_sell_prices if e > purchase_record[-1][0]*(1+sell_threshold)])
                    sell_units = purchase_record[-1][1]

                    cash += sell_units*sell_price
                    stock_units -= sell_units
                    purchase_record.remove(purchase_record[-1])
                    action = 'Sell'
                    action_price = sell_price
                    action_unit = sell_units
            else:
                action = 'Pass'
                action_price = 0
                action_unit = 0
                    
        total_cash_records.append(cash)
        total_num_shares_records.append(stock_units)
        action_records.append(action)
        action_price_records.append(action_price)
        action_unit_records.append(action_unit)

    
    
    # Create a DataFrame to store the records
    records_df = pd.DataFrame({'date': df['date'],
                               'daily_price': df['daily_price'],
                               'action': action_records,
                               'action_price': action_price_records,
                               'action_unit': action_unit_records,
                               'total_cash': total_cash_records,
                               'total_num_shares': total_num_shares_records})
    
    return records_df, purchase_record

def plot_trading_strategy(df_stock, result):
    # plot the daily stock prices as a curve
    plt.figure(figsize=(16, 8))
    plt.plot(df_stock['date'], df_stock['daily_price'], label='Stock Price')


    # plot the purchase actions as red dots
    purchases = result[result['action'] == 'Purchase']
    plt.scatter(purchases['date'], purchases['daily_price'], color='red', label='Purchase')

    for index, row in purchases.iterrows():
        # plt.text(row['date'], row['daily_price'], row['daily_price'], ha='center', va='bottom')
        plt.annotate(f"{row['daily_price']:.2f}", xy=(row['date'], row['daily_price']), 
                     xytext=(row['date'], row['daily_price']*0.98), ha='center', va='top')

    # plot the sell actions as blue dots
    sells = result[result['action'] == 'Sell']
    plt.scatter(sells['date'], sells['daily_price'], color='blue', label='Sell')


    for index, row in sells.iterrows():
        # plt.text(row['date'], row['daily_price'], row['daily_price'], ha='center', va='bottom')
        plt.annotate(f"{row['daily_price']:.2f}", xy=(row['date'], row['daily_price']), 
                     xytext=(row['date'], row['daily_price']*0.98), ha='center', va='top')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Trading Strategy Visualization')
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.show()


def download_stock_df(stock_name, start_date='2018-01-01', end_date=datetime.today().strftime('%Y-%m-%d'), price='Open'):
    '''
    Download the daily stock price from Yahoo Finance as a dataframe
    
    Input:
    stock_name: string, stock name
    start_date: string, 'YYYY-MM-DD' format of date specifying the date range
    end_date: string, 'YYYY-MM-DD' format of date specifying the date range
    price: 'Open' or 'Close', specifying which stock price is output to the result dataframe
    
    Output:
    df_stock_price: dataframe, with two columns: date, daily_price
        where daily_price is the stock open price by default
    '''
    # Download historical data from Yahoo Finance for AAPL
    data = yf.download(stock_name.upper(), start=start_date, end=end_date)
    # Extract the daily closing prices and reset the index
    df_stock_price = data[[price]].reset_index()
    # Rename the columns to match the desired format
    df_stock_price.columns = ['date', 'daily_price']
    # Ensure daily_price is non-negative
    df_stock_price['daily_price'] = df_stock_price['daily_price'].clip(lower=0)
    return df_stock_price



# Create a user function
def user_function():
    stock_name = (input('Enter the stock name:')).upper()
    start_date = input('Enter the strategy start date, in YYYY-MM-DD format (default: %s):' % ((datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d'))) or ((datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d'))
    end_date = input('Enter the strategy end date, in YYYY-MM-DD format (default: %s):' % (datetime.today().strftime('%Y-%m-%d'))) or (datetime.today().strftime('%Y-%m-%d'))
    customerize_ind = input('Would you like to customerize the strategy parameters? Enter Y for yes; otherwise would use default value')
    if customerize_ind.upper() == 'Y':
        initial_cash = float(input('Enter the initial cash (default: %s):' % ('10000')) or (10000))
        investment = float(input('Enter the investment amount for each purchase (default: %s):' % ('1000')) or (1000))
        buy_threshold = int(input('If the stock price drops X percent, make a purchase. Enter X as an integer (default: %s):' % ('5')) or 5) / 100
        sell_threshold = int(input('If the stock price rises X percent, make a sell. Enter X as an integer (default: %s):' % ('5')) or 5) / 100
    else:
        initial_cash = 10000
        investment = 1000
        buy_threshold = 0.05
        sell_threshold = 0.05
    stock_df = download_stock_df(stock_name)
    result = stock_trading_strategy(stock_df, start_date, end_date,initial_cash, investment, buy_threshold=buy_threshold, sell_threshold=sell_threshold, multiplier=1)
    print('Strategy specifics:')
    print('    stock name: %s' %(stock_name))
    print('    strategy start date: %s, strategy end date: %s' %(start_date, end_date))
    print('    initial cash: %s, each investment: %s, buy_threshold: %s, sell_threshold: %s' %(initial_cash, investment, buy_threshold, sell_threshold))
    print('\n')
    print('Final Profit:', ((result.iloc[-1]['total_stock_value'] + result.iloc[-1]['total_cash'])-10000))
    print('Underlying stock price change:', "{0:.02%}".format((result.iloc[-1]['daily_price']-result.iloc[0]['daily_price'])/result.iloc[0]['daily_price']))
    print('Max Profit ever:', (result['total_stock_value']+result['total_cash']).max() - 10000)
    print('Max Loss ever:', (result['total_stock_value']+result['total_cash']).min() - 10000)

    print('Number of Purchase & Sell actions:', result[result['action'].isin(['Purchase','Sell'])].shape[0])
    print('Minimal Cash Reserved: ', result['total_cash'].min())
    print('\n')
    print('Transaction samples:')
    print(result[result['action'].isin(['Purchase','Sell'])].head())
    plot_trading_strategy(stock_df, result)

# create a custom function for the Candlestick chart

def plot_candlestick(df: pd.DataFrame):
    '''
    Input: 
    pd.Dataframe of daily stock data downloaded from Yahoo Finance
    Need to the following columns: date, open, close, high, low, (volume)
    '''
    data = df.copy()
    data.columns = data.columns.str.lower()
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.grid(True, alpha=0.5)

    # Convert dates to mdates
    mdates_data = data.reset_index()
    mdates_data['date'] = mdates.date2num(mdates_data['date'])

    # Plot candlesticks
    for index, row in mdates_data.iterrows():
        open_price = row['open']
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']
        date = row['date']

        if close_price >= open_price:
            color = 'g'  # Green for bullish
            rect_height = close_price - open_price
            bottom = open_price
        else:
            color = 'r'  # Red for bearish
            rect_height = open_price - close_price
            bottom = close_price

        # Plot candlestick body
        ax.add_patch(plt.Rectangle((date - 0.4, bottom), 0.8, rect_height,
                                    facecolor=color, edgecolor='black'))

        # Plot upper wick
        ax.plot([date, date], [high_price, max(open_price, close_price)],
                color='black')

        # Plot lower wick
        ax.plot([date, date], [low_price, min(open_price, close_price)],
                color='black')

    # Customize the appearance and layout of the plot
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.grid()
    
    return ax
