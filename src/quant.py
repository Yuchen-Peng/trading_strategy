import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import pytz
from dateutil.relativedelta import relativedelta
import yfinance as yf
import math
from colorama import Fore, Style
import seaborn as sns
sns.set(style='darkgrid')

from scipy.optimize import fsolve, curve_fit
from scipy.stats import norm
from sklearn.metrics import r2_score
from math import ceil, floor

from utils import plot_candlestick, exponential_func, get_optimum_clusters

# =================== #
# 1. Option strategies
# =================== #
# 1.1 Gamma flip point

def bs_gamma(S, K, T, r, sigma):
    """
    Calculate Gamma for an option given stock price S and strike K
    Parameters:
        S: stock price
        K: option strike
        T: annualized days to expiration (expiration - current_date) / 365
        r: risk-free return rate, use 3-month Treasury bill rate
        sigma: implied volatility (IV)
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def total_gamma_exposure(S, options_df, r=0.0368):
    gex = 0.0
    for _, row in options_df.iterrows():
        gamma = bs_gamma(
            S=S,
            K=row["strike"],
            T=row["T"],
            r=r,
            sigma=row["impliedVolatility"]
        )
        exposure = gamma * row["openInterest"] * 100 * S**2

        # assumption: dealer short calls, and long puts
        if row["type"] == "call":
            gex += exposure
        else:
            gex -= exposure
    return gex

def load_option_chain(tk, expiration):
    """
    Load option chain for a pre-specified Ticker object
    Parameter:
        tk: yf.Ticker object
        expiration: date in %Y-%m-%d format
    Return:
        options_df: dataframe
    """
    calls, puts = tk.option_chain(expiration).calls, tk.option_chain(expiration).puts
    now = datetime.today()
    T = (datetime.strptime(expiration, "%Y-%m-%d") - now).days / 365
    calls = calls.assign(
        type="call",
        T=T
    )
    puts = puts.assign(
        type="put",
        T=T
    )
    options_df = pd.concat([calls, puts], ignore_index=True)
    options_df = options_df[
        (options_df["openInterest"] > 0) &
        (options_df["impliedVolatility"] > 0)
    ]
    return options_df

def find_gamma_flip(ticker_name, use_single_expiration=True, exp_dt=None, price_imputed=None, price_range_pct=0.2):
    """
    Find gamma flip point of a stock
    Parameter:
        ticker_name: str to get yf.Ticker(ticker_name)
        use_single_expiration: boolean
        * if True, default to the 3rd Friday of the month / next month
        * if False, use multiple expiration dates available in the option chain (try within 31 days but blocked by yfinance)
        exp_dt: str, YYYY-MM-DD format specific expiration date
        price_imputed: in case a different than close price needs to be manually imputed
        price_range_pct: find the flip_price within this range

    """
    tk = yf.Ticker(ticker_name)
    if use_single_expiration and exp_dt is None:
        today = datetime.today()
        current_month_start = datetime(today.year, today.month, 1)
        next_month_start = datetime(today.year, today.month+1, 1)
        current_third_friday = datetime(today.year, today.month, 1 + (4 - current_month_start.weekday()) % 7 + 14)
        next_third_friday = datetime(today.year, today.month+1, 1 + (4 - next_month_start.weekday()) % 7 + 14)
        if today <= current_third_friday:
            expirations = [current_third_friday.strftime('%Y-%m-%d')]
        else:
            expirations = [next_third_friday.strftime('%Y-%m-%d')]
    elif use_single_expiration and exp_dt is not None:
        expirations = [exp_dt]
    else:
        # Extracting all expiration dates is disabled by yfinance; using options expiring within 31 day
        expirations = [exp_dt for exp_dt in tk.options
                       if datetime.today() < datetime.strptime(exp_dt, '%Y-%m-%d') < datetime.today() + pd.Timedelta(days=31)]

    
    if price_imputed is None:
        S0 = tk.history(period="1d")["Close"].iloc[-1]
    else:
        S0 = price_imputed
    current_gex_value = np.array([total_gamma_exposure(S0, load_option_chain(tk, exp_dt)) for exp_dt in expirations]).sum()
    prices = np.linspace(
        S0 * (1 - price_range_pct),
        S0 * (1 + price_range_pct),
        200
    )
    gex_values = np.array([
        np.array([total_gamma_exposure(S, load_option_chain(tk, exp_dt)) for exp_dt in expirations]).sum()
        for S in prices]
    )
    sign_change = np.where(np.sign(gex_values[:-1]) != np.sign(gex_values[1:]))[0]

    plt.figure(figsize=(12, 7))
    gex_vals_b = gex_values / 1e9
    plt.plot(prices, gex_vals_b, label='Net Gamma Exposure', color='#1f77b4', linewidth=2.5)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    plt.fill_between(prices, gex_vals_b, 0, where=(gex_vals_b >= 0), 
                     color='green', alpha=0.15, label='Positive Gamma (Stabilizing)')
    plt.fill_between(prices, gex_vals_b, 0, where=(gex_vals_b < 0), 
                     color='red', alpha=0.15, label='Negative Gamma (Volatile)')
    plt.axvline(S0, color='grey', linestyle='--', alpha=0.8)
    plt.text(S0, max(gex_vals_b)*0.9, f' Current: ${S0:.2f}', 
             color='grey', fontweight='bold', ha='left')

    if len(sign_change) > 0:
        plt.axvline(prices[sign_change[0]], color='#d62728', linestyle='--', linewidth=2)
        plt.scatter([prices[sign_change[0]]], [0], color='#d62728', s=100, zorder=5, label='Flip Point')
        plt.text(prices[sign_change[0]], 0, f'  Flip: ${prices[sign_change[0]]:.2f}', 
                 color='#d62728', fontweight='bold', va='bottom', ha='left')

    pic_title = f'{ticker_name.upper()} Total Gamma Exposure Profile\n(Assumes: Market Makers Long Call / Short Put)'
    if len(expirations) == 1:
        pic_title += f'\n(Evaluated using options with expiration date {expirations[0]})'
    else:
        pic_title += f'\n(Evaluated using options expiring within 31 days after {datetime.today().strftime('%Y-%m-%d')})'
    plt.title(pic_title, fontsize=16)
    plt.xlabel('Stock Price ($)', fontsize=12)
    plt.ylabel('Total Gamma Exposure ($ Billion / 1% move)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    history_df = tk.history(period='20d')

    if len(sign_change) == 0:
        print("No flip point (too far from current price, or gamma being always positive / negative)")
        print(f"{ticker_name.upper()} current GEX: {current_gex_value/10**9:.2f}B; average turnover: {(history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean()/10**9:.2f}B; 1% current GEX vs average turnover: {current_gex_value / 100 / (history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean():.2%}")
        print(f"{ticker_name.upper()} GEX minimal at {prices[np.argmin(gex_values)]:.2f}")
        print(f"{ticker_name.upper()} minimal GEX: {gex_values.min()/10**9:.2f}B; average turnover: {(history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean()/10**9:.2f}B; 1% minimal GEX vs average turnover: {gex_values.min() / 100 / (history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean():.2%}")
        print(f"{ticker_name.upper()} GEX maximal at {prices[np.argmax(gex_values)]:.2f}")
        print(f"{ticker_name.upper()} maximal GEX: {gex_values.max()/10**9:.2f}B; average turnover: {(history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean()/10**9:.2f}B; 1% maximal GEX vs average turnover: {gex_values.max() / 100 / (history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean():.2%}")
        return None, current_gex_value

    flip_price = prices[sign_change[0]]
    print(f"{ticker_name.upper()} current Price: {S0:.2f}")
    print(f"{ticker_name.upper()} Gamma Flip Point: {flip_price:.2f}")    
    print(f"{ticker_name.upper()} current GEX: {current_gex_value/10**9:.2f}B; average turnover: {(history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean()/10**9:.2f}B; 1% current GEX vs average turnover: {current_gex_value / 100 / (history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean():.2%}")
    print(f"{ticker_name.upper()} GEX minimal at {prices[np.argmin(gex_values)]:.2f}")
    print(f"{ticker_name.upper()} minimal GEX: {gex_values.min()/10**9:.2f}B; average turnover: {(history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean()/10**9:.2f}B; 1% minimal GEX vs average turnover: {gex_values.min() / 100 / (history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean():.2%}")
    print(f"{ticker_name.upper()} GEX maximal at {prices[np.argmax(gex_values)]:.2f}")
    print(f"{ticker_name.upper()} maximal GEX: {gex_values.max()/10**9:.2f}B; average turnover: {(history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean()/10**9:.2f}B; 1% maximal GEX vs average turnover: {gex_values.max() / 100 / (history_df['Volume']*(history_df['High']+history_df['Low']+history_df['Close'])/3).mean():.2%}")
    return flip_price, current_gex_value

# 1.2 Option walls

def get_option_walls(ticker_symbol, target_date):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period='1d')
    if history.empty: return "Data not found"
    spot_price = history['Close'].iloc[-1]
    high_price = history['High'].iloc[-1]
    low_price = history['Low'].iloc[-1]

    chain = ticker.option_chain(target_date)
    
    # --- 1. 定义非对称阈值，排除深度价内 ---
    # Call: 看现价 -2% 到 +10% (阻力区)
    call_min, call_max = spot_price * 0.98, spot_price * 1.10
    # Put: 看现价 -10% 到 +2% (支撑区)
    put_min, put_max = spot_price * 0.9, spot_price * 1.02
    
    calls = chain.calls[(chain.calls['strike'] >= call_min) & (chain.calls['strike'] <= call_max)]
    puts = chain.puts[(chain.puts['strike'] >= put_min) & (chain.puts['strike'] <= put_max)]

    # Put Wall
    put_wall = puts.loc[puts['openInterest'].idxmax()]
    # Call Wall
    call_wall = calls.loc[calls['openInterest'].idxmax()]
    
    # Calculate OI ratio between call and put
    total_put_oi = puts['openInterest'].sum()
    total_call_oi = calls['openInterest'].sum()
    
    print(f"{ticker_symbol.upper()} Put Wall: ${put_wall['strike']} | OI: {int(put_wall['openInterest'])} | OI vs volume: {int(put_wall['openInterest'])*100 / ticker.history(period='20d')['Volume'].mean():.2%}")
    print(f"{ticker_symbol.upper()} Call Wall: ${call_wall['strike']} | OI: {int(call_wall['openInterest'])} | OI vs volume: {int(call_wall['openInterest'])*100 / ticker.history(period='20d')['Volume'].mean():.2%}")
    print(f"{ticker_symbol.upper()} P/C OI Ratio: {total_put_oi / total_call_oi:.2%}")
    
    # --- 2. 绘图布局 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"{ticker_symbol.upper()} Strategic OI Distribution ({target_date})\nSpot: {spot_price:.2f}", fontsize=14)

    # --- 3. Call Subplot (Resistance) ---
    ax1.bar(calls['strike'], calls['openInterest'], color='seagreen', alpha=0.7)
    ax1.set_ylabel("Call Open Interest")
    ax1.axvline(spot_price, color='blue', linestyle='--', alpha=0.5, label=f'Close Price on {datetime.today().strftime('%Y-%m-%d')}')
    ax1.legend()

    # 标注 OTM/轻微 ITM 中的 Top 5 墙
    top_calls = calls.nlargest(5, 'openInterest')
    for _, row in top_calls.iterrows():
        ax1.text(row['strike'], row['openInterest'], f"{int(row['strike'])}", 
                 ha='center', va='bottom', fontweight='bold', color='darkgreen')

    # --- 4. Put Subplot (Support) ---
    ax2.bar(puts['strike'], puts['openInterest'], color='indianred', alpha=0.7)
    ax2.set_ylabel("Put Open Interest")
    ax2.axvline(spot_price, color='blue', linestyle='--', alpha=0.5, label=f'Close Price on {datetime.today().strftime('%Y-%m-%d')}')
    
    # 标注 OTM/轻微 ITM 中的 Top 5 墙
    top_puts = puts.nlargest(5, 'openInterest')
    for _, row in top_puts.iterrows():
        ax2.text(row['strike'], row['openInterest'], f"{int(row['strike'])}", 
                 ha='center', va='bottom', fontweight='bold', color='darkred')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# =================== #
# 2. Bottom strategy
# =================== #

def bottom_buy(stock_name, scenario='declining', volume_period=5, return_signal=False):
    '''
    Identify buy signal for a stock touching bottom
    if current time >= 3:30pm est, include today's data; otherwise just use yesterday's data
    scenario: 'declining' vs 'ranging'
    长下影缩量（跌不动） or 长柱体放量收绿（大量入场）and 短上影线（deepseek推荐）
    '''
    df_asset = yf.download(stock_name.upper(),
                                  start=(datetime.today() - relativedelta(years=3)).strftime('%Y-%m-%d'),
                                  end=datetime.today().strftime('%Y-%m-%d'),
                                  prepost=True,
                                  auto_adjust=True).droplevel(level='Ticker', axis=1)
    df_asset = df_asset.reset_index()
    df_asset.columns = df_asset.columns.str.lower()
    ticker = yf.Ticker(stock_name.upper())
    if datetime.now(pytz.timezone('US/Eastern')).time() > time(15, 30):
        df_new = ticker.history(period='1d').reset_index()
        df_new.columns = df_new.columns.str.lower()
        df_new['date'] = pd.to_datetime(df_new['date']).dt.tz_localize(None)
        df_asset = pd.concat(
                [df_asset[['date', 'open', 'close', 'high', 'low', 'volume']],
                 df_new[['date', 'open', 'close', 'high', 'low', 'volume']]]
            ).reset_index(drop=True)
    share_outstanding = ticker.info.get('floatShares', ticker.info.get('sharesOutstanding'))
    if share_outstanding is None:
        share_outstanding = int(input("Please enter the outstanding share manually:"))

    df_asset['price_mt_cond1'] = (df_asset['close'].pct_change(5)*100 < -8)
    df_asset['price_mt_cond2'] = (df_asset['close'].pct_change(1)*100 > -3)
    df_asset['price_mt_cond3'] = ((df_asset[['open','close']].min(axis=1) - df_asset['low']) > 1.5 * abs(df_asset['open'] - df_asset['close']))
    df_asset['price_mt_cond4'] = ((df_asset['high'] - df_asset[['open','close']].max(axis=1)) < 0.3 * abs(df_asset['open'] - df_asset['close']))
    
    df_asset['vol_cond1'] = (df_asset['volume'] / df_asset['volume'].rolling(volume_period).mean() < 0.85) # 0.6 too strict! try 0.85
    df_asset['vol_cond2'] = (df_asset['volume'].shift(1) / df_asset['volume'].rolling(volume_period).mean().shift(1) < 0.85) # this often not meet
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
    df_asset['price_mt_cond4_2d'] = df_asset['price_mt_cond4'].rolling(2).max()
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
            df_asset['price_mt_cond4_2d'].fillna(0) *
            # df_asset['vol_cond1_2d'].fillna(0) *
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
        print("Long lower shadow:", df_asset['price_mt_cond3_2d'].iloc[-1])
        print("Short upper shadow:", df_asset['price_mt_cond4_2d'].iloc[-1])
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
            df_asset['price_mt_cond4_2d'].fillna(0) *
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
        
# =================== #
# 3. Peak strategy
# =================== #

def peak_sell(stock_name, volume_period=20, return_signal=False):
    '''
    Identify sell signal for a stock touching peak in an uptrend
    if current time >= 3:30pm est, include today's data; otherwise just use yesterday's data
    '''
    df_asset = yf.download(stock_name.upper(),
                                  start=(datetime.today() - relativedelta(years=3)).strftime('%Y-%m-%d'),
                                  end=datetime.today().strftime('%Y-%m-%d'),
                                  prepost=True,
                                  auto_adjust=True).droplevel(level='Ticker', axis=1)
    df_asset = df_asset.reset_index()
    df_asset.columns = df_asset.columns.str.lower()
    ticker = yf.Ticker(stock_name.upper())
    if datetime.now(pytz.timezone('US/Eastern')).time() > time(15, 30):
        df_new = ticker.history(period='1d').reset_index()
        df_new.columns = df_new.columns.str.lower()
        df_new['date'] = pd.to_datetime(df_new['date']).dt.tz_localize(None)
        df_asset = pd.concat(
                [df_asset[['date', 'open', 'close', 'high', 'low', 'volume']],
                 df_new[['date', 'open', 'close', 'high', 'low', 'volume']]]
            ).reset_index(drop=True)
    if 'RSI_raw' in df_asset.columns:
        rsi_14 = df_asset['RSI_raw']
    else:
        delta = df_asset['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs))
    rsi_condition = (rsi_14>80) & (rsi_14.diff().abs()<2) # what if rsi decrease? should we change this?
    
    volume = df_asset['volume']
    volume_ma = volume.rolling(window=volume_period).mean()
    volume_std = volume.rolling(window=volume_period).std()
    volume_spike = volume > (volume_ma + 1.5*volume_std)
    price_change = (df_asset['close'] - df_asset['close'].shift(1)) / df_asset['close'].shift(1) * 100
    volume_condition = volume_spike & (price_change < 2) & (price_change > -1)

    body_size = abs(df_asset['close'] - df_asset['open'])
    upper_shadow = df_asset['high'] - np.maximum(df_asset['close'], df_asset['open'])
    # body_size = abs(df_asset['close'] - df_asset['close'].shift(1))
    # upper_shadow = df_asset['high'] - np.maximum(df_asset['close'], df_asset['close'].shift(1))
    long_upper_shadow = upper_shadow > 2*body_size
    cross_star = (body_size < (df_asset['high']- df_asset['low'])*0.1)
    price_structure_condition = long_upper_shadow | cross_star
    
    sell_signal = rsi_condition.fillna(0) * volume_condition.fillna(0) * long_upper_shadow.fillna(0)

    print("RSI flat:", rsi_condition.iloc[-1])
    print("Volume spike:", volume_spike.iloc[-1])
    print("Price change little:", ((price_change < 2) & (price_change > -1)).iloc[-1])
    print("Volume condition:", volume_condition.iloc[-1])
    print("Long upper shadow:", long_upper_shadow.iloc[-1])
    print("Cross star:", cross_star.iloc[-1])
    print("Price structure condition:", price_structure_condition.iloc[-1])
    print("Sell signal:", sell_signal.iloc[-1])

    if return_signal:
        return {
            "rsi_condition": rsi_condition.fillna(0),
            "volume_condition": volume_condition.fillna(0),
            "price_structure_condition": price_structure_condition.fillna(0),
            "sell_signal": sell_signal,
        }