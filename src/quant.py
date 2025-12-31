import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import math
from colorama import Fore, Style
import seaborn as sns
sns.set(style='darkgrid')

from scipy.optimize import fsolve, curve_fit
from sklearn.metrics import r2_score
from math import ceil, floor

from utils import plot_candlestick, exponential_func, get_optimum_clusters

def bottom_buy(df_asset, share_outstanding=1, scenario='declining', return_signal=False):
    '''
    Identify buy signal for a stock touching bottom
    share_outstanding: in the unit of MM
    scenario: 'declining' vs 'ranging'
    '''
    share_outstanding = share_outstanding*10**6
    df_asset['price_mt_cond1'] = (df_asset['close'].pct_change(5)*100 < -8)
    df_asset['price_mt_cond2'] = (df_asset['close'].pct_change(1)*100 > -3)
    df_asset['price_mt_cond3'] = ((df_asset[['open','close']].min(axis=1) - df_asset['low']) > 1.5 * abs(df_asset['open'] - df_asset['close']))
    
    df_asset['vol_cond1'] = (df_asset['volume'] / df_asset['volume'].rolling(5).mean() < 0.6)
    df_asset['vol_cond2'] = (df_asset['volume'].shift(1) / df_asset['volume'].rolling(5).mean().shift(1) < 0.6)
    df_asset['vol_cond3'] = (df_asset['volume'] / share_outstanding < 0.03)
    
    delta = df_asset['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_asset['rsi_14'] = 100 - (100 / (1 + rs))
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs_5 = gain / loss
    df_asset['rsi_5'] = 100 - (100 / (1 + rs_5))
    exp1 = df_asset['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_asset['close'].ewm(span=26, adjust=False).mean()
    df_asset['macd'] = exp1 - exp2
    df_asset['macd_signal'] = df_asset['macd'].ewm(span=9, adjust=False).mean()
    df_asset['macd_diff'] = df_asset['macd'] - df_asset['macd_signal']
    
    df_asset['tech_cond1'] = (df_asset['rsi_14'] < 25)
    df_asset['tech_cond2'] = (df_asset['rsi_5'] > df_asset['rsi_5'].shift(1))
    df_asset['tech_cond3'] = (df_asset['macd_diff'] > df_asset['macd_diff'].shift(1))
    df_asset['tech_cond4'] = (df_asset['macd_diff'].shift(1) < df_asset['macd_diff'].shift(2))

    df_asset['price_mt_cond1_2d'] = df_asset['price_mt_cond1'].rolling(2).max()
    df_asset['price_mt_cond2_2d'] = df_asset['price_mt_cond2'].rolling(2).max()
    df_asset['price_mt_cond3_2d'] = df_asset['price_mt_cond3'].rolling(2).max()
    df_asset['vol_cond1_2d'] = df_asset['vol_cond1'].rolling(2).max()
    df_asset['vol_cond2_2d'] = df_asset['vol_cond2'].rolling(2).max()
    df_asset['vol_cond3_2d'] = df_asset['vol_cond3'].rolling(2).max()
    df_asset['tech_cond1_2d'] = df_asset['tech_cond1'].rolling(2).max()
    df_asset['tech_cond2_2d'] = df_asset['tech_cond2'].rolling(2).max()
    df_asset['tech_cond3_2d'] = df_asset['tech_cond3'].rolling(2).max()
    df_asset['tech_cond4_2d'] = df_asset['tech_cond4'].rolling(2).max()

    if scenario == 'ranging':
        # print("Price drop in 5days > 8%:", price_mt_cond1_2d.iloc[-1])
        print("Price drop today < 3%:", df_asset['price_mt_cond2_2d'].iloc[-1])
        print("Long longer shadow:", df_asset['price_mt_cond3_2d'].iloc[-1])
        # print("Volume contraction:", vol_cond1_2d.iloc[-1])
        # print("Volume contraction for two days:", vol_cond2_2d.iloc[-1])
        # print("Low turnover rate:", vol_cond3_2d.iloc[-1])
        # print("RSI_14 < 25:", tech_cond1_2d.iloc[-1])
        print("RSI_5 bottom:", df_asset['tech_cond2_2d'].iloc[-1])
        print("MACD bottom:", df_asset['tech_cond3_2d'].iloc[-1], df_asset['tech_cond4_2d'].iloc[-1])
        df_asset['buy_signal'] = (
            # df_asset['price_mt_cond1_2d'].fillna(0) *
            df_asset['price_mt_cond2_2d'].fillna(0) *
            df_asset['price_mt_cond3_2d'].fillna(0) *
            # df_asset['df_asset['vol_cond1_2d'].fillna(0) *
            # df_asset['vol_cond2_2d'].fillna(0) *
            # df_asset['vol_cond3_2d'].fillna(0) *
            # df_asset['tech_cond1_2d'].fillna(0) *
            df_asset['tech_cond2_2d'].fillna(0) *
            df_asset['tech_cond3_2d'].fillna(0) *
            df_asset['tech_cond4_2d'].fillna(0)
        )
    else:
        print("Price drop in 5days > 8%:", df_asset['price_mt_cond1_2d'].iloc[-1])
        print("Price drop today < 3%:", df_asset['price_mt_cond2_2d'].iloc[-1])
        print("Long longer shadow:", df_asset['price_mt_cond3_2d'].iloc[-1])
        print("Volume contraction:", df_asset['vol_cond1_2d'].iloc[-1])
        print("Volume contraction for two days:", df_asset['vol_cond2_2d'].iloc[-1])
        print("Low turnover rate:", df_asset['vol_cond3_2d'].iloc[-1])
        print("RSI_14 < 25:", df_asset['tech_cond1_2d'].iloc[-1])
        print("RSI_5 bottom:", df_asset['tech_cond2_2d'].iloc[-1])
        print("MACD bottom:", df_asset['tech_cond3_2d'].iloc[-1])
        df_asset['buy_signal'] = (
            df_asset['price_mt_cond1_2d'].fillna(0) *
            df_asset['price_mt_cond2_2d'].fillna(0) *
            df_asset['price_mt_cond3_2d'].fillna(0) *
            df_asset['vol_cond1_2d'].fillna(0) *
            df_asset['vol_cond2_2d'].fillna(0) *
            df_asset['vol_cond3_2d'].fillna(0) *
            df_asset['tech_cond1_2d'].fillna(0) *
            df_asset['tech_cond2_2d'].fillna(0) *
            df_asset['tech_cond3_2d'].fillna(0)
        )
    print("Buy signal:",
        df_asset['buy_signal'].iloc[-1]
    )
    
    if return_signal:
        return df_asset

def peak_sell(df_asset, return_signal=False):
    '''
    Identify sell signal for a stock touching peak in an uptrend
    '''
    
