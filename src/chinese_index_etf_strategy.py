import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from dateutil.relativedelta import relativedelta
import yfinance as yf
import akshare as ak
from colorama import Fore, Style
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objects as go

from scipy.optimize import fsolve, curve_fit
from sklearn.metrics import r2_score
from math import ceil, floor

from utils import plot_candlestick, exponential_func, get_optimum_clusters

BEIJING_TZ = pytz.timezone('Asia/Shanghai')


def etf_regression(etf_code,
                   regression_start='20100101',
                   end=datetime.now(BEIJING_TZ).strftime('%Y%m%d'),
                   detailed=False,
                   adjust="qfq",
                   source="etf",
                   ):
    '''
    Frequent value for regression_start: '20100101', '20200101', '20200320'
    Including maximal drawdown
    '''
    if source == "etf":
        df_etf = ak.fund_etf_hist_em(symbol=etf_code,
        	                   period="daily", 
                               start_date=regression_start,
                               end_date=end,
                               adjust=adjust)
        df_etf.rename(columns={"日期": 'date', "开盘": 'open', "收盘": 'close', "最高": 'high', "最低": 'low', "成交量": 'volume', "成交额": 'amount'}, inplace=True)
    elif source == "index" and etf_code[0:3].upper() == "CSI":
        df_etf = ak.stock_zh_index_hist_csindex(symbol=etf_code[3:].upper(), start_date=regression_start, end_date=end)
        df_etf.rename(columns={"日期": 'date', "开盘": 'open', "收盘": 'close', "最高": 'high', "最低": 'low', "成交量": 'volume', "成交额": 'amount'}, inplace=True)
    elif source == "index" and etf_code[0:2].upper() == "HK":
        # ak.stock_hk_index_daily_em cannot return volume
        df_etf = ak.stock_hk_index_daily_sina(symbol=etf_code[2:].upper())
        df_etf = df_etf[df_etf['date'].between(pd.to_datetime(regression_start).date(), pd.to_datetime(end).date())]
    elif source == "index" and etf_code[0:2].lower() in ["sh", "sz"]:
        df_etf = ak.stock_zh_index_daily_em(symbol=etf_code.lower(), start_date=regression_start, end_date=end)
    else:
        print("Index / ETF source not recognized")
        
    df_etf['date'] = pd.to_datetime(df_etf['date'])
    df_etf['max_price'] = df_etf['high'].cummax()
    df_etf['drawdown'] = (df_etf['low'] - df_etf['max_price']) / df_etf['max_price']
    
    max_drawdown_row = df_etf.loc[df_etf['drawdown'].idxmin()]
    max_drawdown = max_drawdown_row['drawdown']
    
    peak_date = df_etf.loc[:max_drawdown_row.name, 'high'].idxmax()
    peak_price = round(df_etf.loc[peak_date, 'high'],2)
    trough_date = max_drawdown_row['date']
    
    df_etf = df_etf.reset_index()
    df_etf['log_price'] = np.log(df_etf['close'])

    df_reg = df_etf
    df_reg['days_since_start'] = (df_reg['date'] - df_reg['date'].min()).dt.days

    # Initial guess for parameters
    initial_guess = (df_reg.head(1)['close'].unique()[0], np.log10(df_reg.tail(1)['close'].unique()[0] / df_reg.head(1)['close'].unique()[0])/df_reg.tail(1)['days_since_start'].unique()[0])  # You might need to adjust these initial values based on your data

    # Fit the curve
    popt, pcov = curve_fit(exponential_func, df_reg['days_since_start'], df_reg['close'], p0=initial_guess)
    a_fit, b_fit = popt
    # Generate the fitted curve
    fitted_curve = exponential_func(df_reg['days_since_start'], *popt)
    # Construct bands, using residual and std in log space
    log_residuals = df_reg['log_price'] - np.log(fitted_curve)
    sigma = log_residuals.std()
    upper_1 = fitted_curve * np.exp(1 * sigma)
    lower_1 = fitted_curve * np.exp(-1 * sigma)
    upper_2 = fitted_curve * np.exp(2 * sigma)
    lower_2 = fitted_curve * np.exp(-2 * sigma)
    
    if detailed:
        print(f"Annual expected return rate: {(10**(b_fit*365)-1):.2%}")
        print(f"Maximal Drawdown: {max_drawdown:.2%}")
        print(f"Peak date: {peak_date} with price {peak_price}")
        print(f"Trough date: {trough_date.date()} with price {round(max_drawdown_row['low'],2)}")
        print("R2:", round(r2_score(np.log10(df_reg['close']).values.reshape(-1, 1), np.log10(fitted_curve)),4))

        # For fitted_curve that starts above the actual, the drawdown is not practical as we won't buy there
        # Start from the first place where fitted_curve is below the actual, then becomes larger than actual
        index_start = np.where(df_reg['close'].values > fitted_curve)[0][0]
        print(f"Maximal Drawdown from regression price relative to regression price: {((fitted_curve[index_start:] - df_reg['close'].values[index_start:]) / fitted_curve[index_start:]).max():.2%}")
        # Plot the original data and the fitted curve
        plt.figure(figsize=(10, 6))
        plt.scatter(df_reg['date'], df_reg['close'], label='Daily Close Price')
        plt.plot(df_reg['date'], fitted_curve, 'r-', label='Fitted Curve')
        plt.plot(df_reg['date'], upper_1, 'm--', label='Upper 1σ')
        plt.plot(df_reg['date'], lower_1, 'm--', label='Lower 1σ')
        plt.plot(df_reg['date'], upper_2, 'y--', label='Upper 2σ')
        plt.plot(df_reg['date'], lower_2, 'y--', label='Lower 2σ')
        plt.fill_between(df_reg['date'], lower_1, upper_1, color='gray', alpha=0.3)
        plt.fill_between(df_reg['date'], lower_2, upper_2, color='gray', alpha=0.15)
        plt.xlabel('Days Since Start')
        plt.ylabel('Price')
        plt.title('Exponential Curve Fitting for ' + etf_code.upper(), fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.show()
    return {
        'fitted': round(fitted_curve.iloc[-1],2),
        'upper_1': round(upper_1.iloc[-1],2),
        'lower_1': round(lower_1.iloc[-1],2),
        'upper_2': round(upper_2.iloc[-1],2),
        'lower_2': round(lower_2.iloc[-1],2),
    }


class etf_strategy:
    """
    class to print and plot etf trading strategy
    strategy: allowed values are 'longterm', 'daily', 'test'
    """
    def __init__(
        self,
        etf_code: str,
        start: datetime = (datetime.now(BEIJING_TZ) - relativedelta(years=3)).strftime('%Y%m%d'),
        end: datetime = datetime.now(BEIJING_TZ).strftime('%Y%m%d'),
        saturation_point: float = 0.05,
        impute: bool = False,
        strategy: str = 'longterm',
        adjust: bool = "qfq", # Use adjust="qfq" as default to adjust historical price
        source="etf",
        ):
        self.etf_code = etf_code
        self.strategy = strategy
        self.start = start
        self.end = end
        self.adjust = adjust

        print(f"Downloading data for {self.etf_code.upper()} from {start} to {end} using akshare...")

        if source == "etf":
            self.df = ak.fund_etf_hist_em(
            	symbol=self.etf_code,
            	start_date=self.start,
            	end_date=self.end,
            	period='daily',  # Default to daily,
            	adjust=self.adjust
            )
            # Rename columns to match the yfinance dataframe structure
            # akshare column names might be in Chinese or different.
            # Common akshare columns for ETF historical data might include:
            # '日期' (date), '开盘' (Open), '收盘' (close), '最高' (high), '最低' (low), '成交量' (Volume)
            self.df.rename(columns={"日期": 'date', "开盘": 'open', "收盘": 'close', "最高": 'high', "最低": 'low', "成交量": 'volume', "成交额": 'amount'}, inplace=True)
        elif source == "index" and etf_code[0:3].upper() == "CSI":
            self.df = ak.stock_zh_index_hist_csindex(
            	symbol=self.etf_code[3:].upper(),
            	start_date=self.start,
            	end_date=self.end
            )
            self.df.rename(columns={"日期": 'date', "开盘": 'open', "收盘": 'close', "最高": 'high', "最低": 'low', "成交量": 'volume', "成交额": 'amount'}, inplace=True)
        elif source == "index" and etf_code[0:2].upper() == "HK":
            # ak.stock_hk_index_daily_em cannot return volume
            self.df = ak.stock_hk_index_daily_sina(
                symbol=self.etf_code[2:].upper(),
            )
            self.df = self.df[self.df['date'].between(pd.to_datetime(self.start).date(), pd.to_datetime(self.end).date())]

        elif source == "index" and etf_code[0:2].lower() in ["sh", "sz"]:
            self.df = ak.stock_zh_index_daily_em(
            	symbol=self.etf_code.lower(),
            	start_date=self.start,
            	end_date=self.end
            )
        else:
            print("Index / ETF source not recognized")
        self.df['date'] = pd.to_datetime(self.df['date']) # Convert 'date' column to datetime objects
        self.df = self.df[['date', 'open', 'high', 'low', 'close', 'volume']] # Keep only necessary columns

        today_str = datetime.now(BEIJING_TZ).strftime('%Y%m%d')
        latest_data = self.df.tail(1)
        if not latest_data.empty:
            self.ticker_close_price = latest_data.iloc[-1]['close']
        else:
            print(f"Could not retrieve real-time price for {self.etf_code.upper()}. Using last available close price.")
            self.ticker_close_price = self.df['close'].iloc[-1]

        self.calculate_ema()
        self.create_bb()
        self.create_weekly()
        self.calculate_rolling_vwap()
        self.calculate_rsi()
        self.create_macd()
        if self.strategy == 'daily':
            self.low_centers, self.high_centers = self.support_and_resistance(saturation_point)

    def calculate_ema(self):
        '''
        Calculate 5 Day EMA, 12 Day EMA & 26 Day EMA (for MACD), and 50 Day EMA & 200 Day EMA
        '''
        self.df['5 Day EMA'] = self.df['close'].ewm(span=5, adjust=False).mean()
        self.df['12 Day EMA'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['26 Day EMA'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['50 Day EMA'] = self.df['close'].ewm(span=50, adjust=False).mean()
        self.df['200 Day EMA'] = self.df['close'].ewm(span=200, adjust=False).mean()

    def create_bb(self):
        '''
        Create 5 Day MA; 20 Day MA, 50 Day MA, and their corresponding Bollinger Bands
        '''
        self.df['5 Day MA'] = self.df['close'].rolling(window=5).mean()
        self.df['20 Day MA'] = self.df['close'].rolling(window=20).mean()
        self.daily_std20 = self.df['close'].rolling(window=20).std()
        self.df['Upper Band - 20MA'] = self.df['20 Day MA'] + (self.daily_std20 * 2)
        self.df['lower Band - 20MA'] = self.df['20 Day MA'] - (self.daily_std20 * 2)
        self.df['50 Day MA'] = self.df['close'].rolling(window=50).mean()
        self.daily_std50 = self.df['close'].rolling(window=50).std()
        self.df['Upper Band - 50MA'] = self.df['50 Day MA'] + (self.daily_std50 * 2.5)
        self.df['lower Band - 50MA'] = self.df['50 Day MA'] - (self.daily_std50 * 2.5)
        self.df['120 Day MA'] = self.df['close'].rolling(window=120).mean()
        self.df['200 Day MA'] = self.df['close'].rolling(window=200).mean()

    def create_weekly(self):
        # Calculate weekly MA
        # self.weekly = self.df[['date', 'close']].set_index('date')['close'].resample("W-FRI").last()
        self.weekly = self.df[['date', 'close']].set_index('date')['close'].resample("W-FRI").ffill()
        self.weekly_ma10 = self.weekly.rolling(10).mean()
        self.weekly_ma30 = self.weekly.rolling(30).mean()
        self.weekly_ma40 = self.weekly.rolling(40).mean()
        
        # Weekly Bollinger Bands (20-week)
        self.weekly_ma20 = self.weekly.rolling(20).mean()
        self.weekly_std20 = self.weekly.rolling(20).std()
        self.weekly_bb_upper = self.weekly_ma20 + 2 * self.weekly_std20
        self.weekly_bb_lower = self.weekly_ma20 - 2 * self.weekly_std20

    def calculate_rolling_vwap(self):
        tp = (self.df.set_index('date')['high'] + self.df.set_index('date')['low'] + self.df.set_index('date')['close']) / 3
        pv = tp*self.df.set_index('date')['volume']
        # periods = [5, 10, 20, 50, 200]
        rolling_pv = pv.rolling(window=5).sum()
        rolling_volume = self.df.set_index('date')['volume'].rolling(window=5).sum()
        self.vwap_5d = rolling_pv / rolling_volume
        rolling_pv = pv.rolling(window=10).sum()
        rolling_volume = self.df.set_index('date')['volume'].rolling(window=10).sum()
        self.vwap_10d = rolling_pv / rolling_volume
        rolling_pv = pv.rolling(window=20).sum()
        rolling_volume = self.df.set_index('date')['volume'].rolling(window=20).sum()
        self.vwap_20d = rolling_pv / rolling_volume
        rolling_pv = pv.rolling(window=50).sum()
        rolling_volume = self.df.set_index('date')['volume'].rolling(window=50).sum()
        self.vwap_50d = rolling_pv / rolling_volume     
        rolling_pv = pv.rolling(window=200).sum()
        rolling_volume = self.df.set_index('date')['volume'].rolling(window=200).sum()
        self.vwap_200d = rolling_pv / rolling_volume
        # calculate band for vwap
        log_vwap_20d = np.log(self.vwap_20d)
        rolling_std_20d = log_vwap_20d.rolling(window=20).std()
        self.vwap_20d_upper_1 = self.vwap_20d * np.exp(1*rolling_std_20d)
        self.vwap_20d_lower_1 = self.vwap_20d * np.exp(-1*rolling_std_20d)
        self.vwap_20d_upper_2 = self.vwap_20d * np.exp(2*rolling_std_20d)
        self.vwap_20d_lower_2 = self.vwap_20d * np.exp(-2*rolling_std_20d)
        # Just in case 10d needed
        log_vwap_10d = np.log(self.vwap_10d)
        rolling_std_10d = log_vwap_10d.rolling(window=10).std()
        self.vwap_10d_upper_1 = self.vwap_10d * np.exp(1*rolling_std_10d)
        self.vwap_10d_lower_1 = self.vwap_10d * np.exp(-1*rolling_std_10d)
        self.vwap_10d_upper_2 = self.vwap_10d * np.exp(2*rolling_std_10d)
        self.vwap_10d_lower_2 = self.vwap_10d * np.exp(-2*rolling_std_10d)
        
    def calculate_rsi(self):
        '''
        Calculate RSI
        '''
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def create_macd(self):
        '''
        Create MACD using 12 Day EMA and 26 Day EMA
        '''
        # Calculate MACD line
        self.df['MACD'] = self.df['12 Day EMA'] - self.df['26 Day EMA']
        # Calculate signal line (9-period EMA of MACD line)
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_diff'] = self.df['MACD'] - self.df['MACD_signal']

    def support_and_resistance(self, saturation_point):
        '''
        Calculate support and resistance using KMean
        '''
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(months=6)]
        else:
            df_plot = self.df
        low_clusters = get_optimum_clusters(df_plot[(df_plot['low']!=df_plot['open'])&(df_plot['low']!=df_plot['close'])][['date',"low"]].set_index('date'), saturation_point)
        # low_clusters = get_optimum_clusters(df_plot[['date',"low"]].set_index('date'))
        low_centers = low_clusters.cluster_centers_
        low_centers = np.sort(low_centers, axis=0)

        high_clusters = get_optimum_clusters(df_plot[(df_plot['high']!=df_plot['open'])&(df_plot['high']!=df_plot['close'])][['date',"high"]].set_index('date'), saturation_point)
        # high_clusters = get_optimum_clusters(df_plot[['date',"high"]].set_index('date'))
        high_centers = high_clusters.cluster_centers_
        high_centers = np.sort(high_centers, axis=0)
        return low_centers, high_centers

    def calculate_anchored_vwap(self, start_date='2024-09-24', plot=True):
        anchored_df = self.df[self.df['date'] >= start_date]
        anchored_df.set_index('date', inplace=True)
        # Ensure the DataFrame index is a datetime object for proper comparison
        if not isinstance(anchored_df.index, pd.DatetimeIndex):
            anchored_df.index = pd.to_datetime(anchored_df.index)
        anchored_df['tp'] = (anchored_df['high'] + anchored_df['low'] + anchored_df['close']) / 3
        anchored_df['pv'] = anchored_df['tp'] * anchored_df['volume']
        cumulative_pv = anchored_df['pv'].cumsum()
        cumulative_volume = anchored_df['volume'].cumsum()
        anchored_vwap_series = cumulative_pv / cumulative_volume
        print(f'Latest anchored VWAP since {start_date} is {anchored_vwap_series.iloc[-1]}')
        # anchored vwap band
        log_anchored_vwap = np.log(anchored_vwap_series)
        anchored_vwap_std = log_anchored_vwap.expanding().std()
        anchored_vwap_upper_1 = anchored_vwap_series * np.exp(1*anchored_vwap_std)
        anchored_vwap_lower_1 = anchored_vwap_series * np.exp(-1*anchored_vwap_std)
        anchored_vwap_upper_2 = anchored_vwap_series * np.exp(2*anchored_vwap_std)
        anchored_vwap_lower_2 = anchored_vwap_series * np.exp(-2*anchored_vwap_std)
        
        # directly plot
        if plot:
            ax = plot_candlestick(anchored_df.reset_index(), figsize=(32,8))
            ax.plot(anchored_df.index, anchored_df['close'], ls='--', label='Daily close price')
            ax.plot(anchored_df.index, anchored_vwap_series, ls='--', label='AWAP')
            ax.fill_between(anchored_df.index, anchored_vwap_upper_1, anchored_vwap_lower_1, color='gray', alpha=0.3)
            ax.fill_between(anchored_df.index, anchored_vwap_upper_2, anchored_vwap_lower_2, color='gray', alpha=0.15)
            ax.set_ylabel('Price')
            ax.set_title(f'{self.etf_code.upper()}: daily price vs anchored VWAP since {start_date}')
            ax.grid(True, alpha=0.5)
            ax2 = ax.twinx()
            ax2.bar(anchored_df.index, anchored_df['volume'], alpha=0.3, color='orange', label='Daily Volume')
            ax2.set_ylabel('Volume')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax.legend()
            plt.show()

    def atr_drawdown_thresholds(self, period: int = 14) -> float:
        df = self.df.set_index('date').tail(100).sort_index()
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift()).abs()
        low_close_prev = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        # atr = tr.rolling(window=period, min_periods=period).mean()  # SMA version
        atr = tr.ewm(span=period, adjust=False).mean()            # EMA version
        self.atr_static = atr.iloc[-2]  # latest ATR, not including today's dynamic prices
        previous_day = self.df[self.df['date']<datetime.now(BEIJING_TZ).strftime('%Y%m%d')]['date'].max()
        last_closing = self.df[self.df['date']==previous_day]['close'].values[0]
        print(f"Last closing price: {last_closing}; ATR: {self.atr_static}")
        p_breakout = float(input('Enter the breakout price:'))
        print(f"Did the last closing price stay above breakout price - 0.5*ATR? {last_closing > p_breakout - 0.5*self.atr_static}")
        
    def print_info(self):
        '''
        Print out the information needed
        '''
        previous_day = self.df[self.df['date']<datetime.now(BEIJING_TZ).strftime('%Y%m%d')]['date'].max()
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(months=6)]
        else:
            df_plot = self.df
        new_price = self.ticker_close_price # Use the stored latest price
        if self.strategy == 'daily':
            try:
                support = max([e[0] for e in self.low_centers if e < new_price])
            except:
                print('Break all support; record min etf price')
                support = df_plot['low'].min()
            try:
                resistance = min([e[0] for e in self.high_centers if e > new_price])
            except:
                print('Break all resistance; record max etf price')
                resistance = df_plot['high'].max()
            print('* Current etf price:', round(new_price,2), '~ up', ceil(resistance*100)/100.0, ', down', floor(support*100)/100)
        elif self.strategy == 'longterm':
            print('* Current etf price:', round(new_price,2))
        print('* Recent high:', round(df_plot['high'].max(),2))
        print('* Current etf price is at ' + str(100*round(new_price/df_plot['high'].max(),4)) + '% of recent high')
        print("Latest 5 Day MA:", round(self.df[self.df['date']==previous_day]['5 Day MA'].item(), 2))
        print("Latest 5 Day EMA:", round(self.df[self.df['date']==previous_day]['5 Day EMA'].item(), 2))
        print("Latest 20 Day MA:", round(self.df[self.df['date']==previous_day]['20 Day MA'].item(), 2))
        print("Latest lower Bollinger Band, 20MA:", round(self.df[self.df['date']==previous_day]['lower Band - 20MA'].item(), 2))
        print("Latest higher Bollinger Band, 20MA:", round(self.df[self.df['date']==previous_day]['Upper Band - 20MA'].item(), 2))
        print("Latest 50 Day MA:", round(self.df[self.df['date']==previous_day]['50 Day MA'].item(), 2))
        print("Latest lower Bollinger Band, 50MA:", round(self.df[self.df['date']==previous_day]['lower Band - 50MA'].item(), 2))
        print("Latest higher Bollinger Band, 50MA:", round(self.df[self.df['date']==previous_day]['Upper Band - 50MA'].item(), 2))
        print("Latest 50 Day EMA:", round(self.df[self.df['date']==previous_day]['50 Day EMA'].item(), 2))
        print("Latest 120 Day MA:", round(self.df[self.df['date']==previous_day]['120 Day MA'].item(), 2))
        print("Latest 200 Day MA:", round(self.df[self.df['date']==previous_day]['200 Day MA'].item(), 2))
        print("Latest 200 Day EMA:", round(self.df[self.df['date']==previous_day]['200 Day EMA'].item(), 2))
        
        print("Latest 10 Week MA:", round(self.weekly_ma10.iloc[-1], 2))
        print("Latest 20 Week MA:", round(self.weekly_ma20.iloc[-1], 2))
        print("Latest 30 Week MA:", round(self.weekly_ma30.iloc[-1], 2))
        print("Latest 40 Week MA:", round(self.weekly_ma40.iloc[-1], 2))
        print("Latest Lower Weekly Bollinger Band, 20MA:", round(self.weekly_bb_lower.iloc[-1], 2))
        print("Latest Higher Weekly Bollinger Band, 20MA:", round(self.weekly_bb_upper.iloc[-1], 2))
        print()

        latest_rsi = round(self.df[self.df['date']==previous_day]['RSI'].item(), 2)
        if latest_rsi > 70:
            print("Latest RSI:", Fore.RED + str(latest_rsi), Style.RESET_ALL)
        elif latest_rsi < 30:
            print("Latest RSI:", Fore.GREEN + str(latest_rsi), Style.RESET_ALL)
        else:
            print("Latest RSI:", latest_rsi, Style.RESET_ALL)

        latest_macd = round(self.df[self.df['date']==previous_day]['MACD_diff'].item(), 4)
        if latest_macd < 0:
            print("Latest MACD Divergence:", Fore.RED + str(latest_macd), Style.RESET_ALL)
        elif latest_macd > 0:
            print("Latest MACD Divergence:", Fore.GREEN + str(latest_macd), Style.RESET_ALL)
        else:
            print("Latest MACD Divergence:", Fore.BLACK + str(latest_macd), Style.RESET_ALL)

        self.break_point_solution()

        if self.strategy == 'daily':
            print(self.low_centers)
            print(self.high_centers)

    def break_point_solution(self):
        '''
        Solve for the break point solution price for breaking the current MA/BB
        '''
        # Download data for breakpoint calculation using akshare
        df = self.df.tail(300)

        # Define the expression whose roots we want to find
        # fsolve is not satisfying; provide analytical solution

        last_4day_price = df['close'][-5:-1]
        last_19day_price = df['close'][-20:-1]
        last_49day_price = df['close'][-50:-1]
        last_119day_price = df['close'][-120:-1]
        last_199day_price = df['close'][-200:-1]
        
        if (last_19day_price.sum() - 4*last_4day_price.sum())/3 > 0:
            print('5MA crosses 20MA at', round((last_19day_price.sum() - 4*last_4day_price.sum())/3, 2))
        if (2*last_49day_price.sum() - 5*last_19day_price.sum())/3 > 0:
            print('20MA crosses 50MA at', round((2*last_49day_price.sum() - 5*last_19day_price.sum())/3, 2))
        if (last_199day_price.sum() - 4*last_49day_price.sum())/3 > 0:
            print('50MA crosses 200MA at', round((last_199day_price.sum() - 4*last_49day_price.sum())/3, 2))

        a1 = np.sum(last_19day_price)
        a2 = np.sum(last_19day_price**2)
        p_ma = np.mean(last_19day_price)
        # These formulas look like they are derived from specific assumptions
        # and might need re-derivation if the underlying data distribution/model changes
        # with akshare data. Keeping them as is based on your provided code.
        p_ubb = (562*a1 + np.sqrt((562*a1)**2 - 4*(5339*(99*a1**2-1600*a2))))/5339/2
        p_lbb = (562*a1 - np.sqrt((562*a1)**2 - 4*(5339*(99*a1**2-1600*a2))))/5339/2

        print('5MA break point:', round(np.mean(df['close'][-5:-1]), 2))
        print('20MA break point:', round(p_ma,2))
        print('20MA lower Bollinger Band break point:', round(p_lbb,2))
        print('20MA Upper Bollinger Band break point:', round(p_ubb,2))

        a1 = np.sum(last_49day_price)
        a2 = np.sum(last_49day_price**2)
        p_ma = np.mean(last_49day_price)
        p_ubb = (4177*a1 + np.sqrt((4177*a1)**2 - 4*(102336.5*(361.5*a1**2-15625*a2))))/102336.5/2
        p_lbb = (4177*a1 - np.sqrt((4177*a1)**2 - 4*(102336.5*(361.5*a1**2-15625*a2))))/102336.5/2
        
        print('50MA break point:', round(p_ma,2))
        print('50MA lower Bollinger Band break point:', round(p_lbb,2))
        print('50MA Upper Bollinger Band break point:', round(p_ubb,2))

        print('120MA break point:', round(np.mean(last_119day_price),2))
        print('200MA break point:', round(np.mean(last_199day_price),2))

    def plot_daily_chart(self, interactive_plot):
        '''
        Plot the daily etf trading charts
        Including:
        * SMA
        * EMA
        * Bollinger Bands
        * Support and Resistance
        * MACD
        '''
        # Start from month-6 to have full Bollingerband
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)]
        else:
            df_plot = self.df
        if self.strategy == 'daily':
            ax = plot_candlestick(df_plot, figsize=(32,16))
            ax.set_title(self.etf_code.upper(), fontsize=32)
            for low in self.low_centers[:]:
                ax.axhline(low[0], color='green', ls='--', label=f'Support at {round(low[0],4)}')
            for high in self.high_centers[:]:
                ax.axhline(high[0], color='red', ls='--', label=f'Resistance at {round(high[0],4)}')
            ax.plot(df_plot['date'], df_plot['20 Day MA'], ls='--', label='20 Day Moving Average')
            ax.plot(df_plot['date'], df_plot['Upper Band - 20MA'], ls='--', label='Upper Bollinger Band, 20MA')
            ax.plot(df_plot['date'], df_plot['lower Band - 20MA'], ls='--', label='lower Bollinger Band, 20MA')
            ax.fill_between(df_plot['date'], df_plot['Upper Band - 20MA'], df_plot['lower Band - 20MA'], color='gray', alpha=0.3) # Fill the area between the bands
            ax.plot(df_plot['date'], df_plot['50 Day MA'], ls='--', label='50 Day Moving Average')
            ax.plot(df_plot['date'], df_plot['Upper Band - 50MA'], ls='--', label='Upper Bollinger Band, 50MA')
            ax.plot(df_plot['date'], df_plot['lower Band - 50MA'], ls='--', label='lower Bollinger Band, 50MA')
            ax.fill_between(df_plot['date'], df_plot['Upper Band - 50MA'], df_plot['lower Band - 50MA'], color='gray', alpha=0.15) # Fill the area between the bands

            ax.legend(loc='upper left', fontsize=16)

        elif self.strategy == 'longterm' and interactive_plot:
            figsize=(12, 8)
            fig = go.Figure(layout=dict(width=figsize[0]*80, height=figsize[1]*80))
            
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['close'], name='Daily Price'))
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['50 Day MA'], mode='lines', line=dict(dash='dash'), name='50 Day MA'))
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['lower Band - 50MA'], mode='lines', line=dict(dash='dash'), name='lower Band'))
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Upper Band - 50MA'], mode='lines', line=dict(dash='dash'), name='Upper Band', fill='tonexty', fillcolor='rgba(128,128,128,0.3)'))
            
            fig.update_layout(title='Interactive Plot of Daily etf Price for ' + self.etf_code.upper(),
                              xaxis_title='date',
                              yaxis_title='Daily Price',
                              hovermode='closest')
            
            # Add ability to select a single data point
            fig.update_traces(marker=dict(size=10, opacity=0.8),
                              selector=dict(mode='markers'))
            
            fig.show()

        elif self.strategy == 'longterm' and not interactive_plot:
            figsize=(24, 12)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
            ax1.grid(True, alpha=0.5)
            ax1.plot(df_plot['date'], df_plot['close'], label='Daily close Price')
            ax1.plot(df_plot['date'], df_plot['50 Day MA'], ls='--', label='50 Day Moving Average')
            ax1.plot(df_plot['date'], df_plot['Upper Band - 50MA'], ls='--', label='Upper Bollinger Band, 50MA')
            ax1.plot(df_plot['date'], df_plot['lower Band - 50MA'], ls='--', label='lower Bollinger Band, 50MA')
            ax1.fill_between(df_plot['date'], df_plot['Upper Band - 50MA'], df_plot['lower Band - 50MA'], color='gray', alpha=0.3) # Fill the area between the bands
            ax1.set_title('Daily etf price for ' + self.etf_code.upper(), fontsize=32)
            ax1.legend(fontsize=16)
            
            # Plot MACD and signal line, color bars based on MACD above/below signal line
            ax2.plot(df_plot['date'], df_plot['MACD'], label='MACD', color='red')
            ax2.plot(df_plot['date'], df_plot['MACD_signal'], label='Signal Line', linestyle='--', color='blue')
            bar_colors = ['green' if macd > signal else 'red' for macd, signal in zip(df_plot['MACD'], df_plot['MACD_signal'])]
            ax3 = ax2.twinx()
            ax3.bar(df_plot['date'], df_plot['MACD'] - df_plot['MACD_signal'], width=1.5,  alpha=0.3, align='center', color=bar_colors)        
            ax2.set_title('MACD')
            ax2.legend()
            fig.suptitle('MACD Analysis', fontsize=16)
            fig.subplots_adjust(hspace=0.05)  # Reduce vertical space between subplots
            plt.xticks(rotation=45)  
            
            plt.tight_layout()
            plt.show()
            
    def plot_weekly_chart(self):
        '''
        Plot the weekly etf trading charts
        Including:
        * 10 - 40 weekly MA
        * 20 weekly Bollinger Bands
        '''
        plt.figure(figsize=(8,4.8))
        plt.plot(self.weekly[~self.weekly_ma20.isna()].index, self.weekly[~self.weekly_ma20.isna()], label="Weekly Close Price", color="black", linewidth=1)
        plt.plot(self.weekly_ma10[~self.weekly_ma20.isna()].index, self.weekly_ma10[~self.weekly_ma20.isna()], label="10-week MA (~50-day)", ls='--', color="blue", linewidth=1)
        plt.plot(self.weekly_ma30.index, self.weekly_ma30, label="30-week MA (~150-day)", ls='--', color="orange", linewidth=1)
        plt.plot(self.weekly_ma40.index, self.weekly_ma40, label="40-week MA (~200-day)", ls='--', color="red", linewidth=1)
        
        plt.plot(self.weekly_ma20.index, self.weekly_ma20, label="20-week MA (BB mid)", ls='--', color="green", linewidth=1)
        plt.fill_between(self.weekly_bb_upper.index, self.weekly_bb_lower, self.weekly_bb_upper, color="gray", alpha=0.2, label="20-week Bollinger Band")
        
        plt.title(f'Weekly ETF price for {self.etf_code.upper()}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45) 
        plt.tight_layout()
        plt.show()
        
    def plot_daily_vwap(self):
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(months=9)]
        else:
            df_plot = self.df
        ax = plot_candlestick(df_plot, figsize=(32,8))
        print(f"Latest 5D VWAP: {self.vwap_5d.iloc[-1]}")
        print(f"Latest 10D VWAP: {self.vwap_10d.iloc[-1]}")
        print(f"Latest 20D VWAP: {self.vwap_20d.iloc[-1]}")
        print(f"Latest 50D VWAP: {self.vwap_50d.iloc[-1]}")
        print(f"Latest 200D VWAP: {self.vwap_200d.iloc[-1]}")
        ax.plot(self.vwap_5d[~self.vwap_200d.isna()].index, self.vwap_5d[~self.vwap_200d.isna()], label='5D VWAP')
        ax.plot(self.vwap_10d[~self.vwap_200d.isna()].index, self.vwap_10d[~self.vwap_200d.isna()], label='10D VWAP')
        ax.plot(self.vwap_20d[~self.vwap_200d.isna()].index, self.vwap_20d[~self.vwap_200d.isna()], label='20D VWAP')
        ax.plot(self.vwap_50d[~self.vwap_200d.isna()].index, self.vwap_50d[~self.vwap_200d.isna()], label='50D VWAP')
        ax.plot(self.vwap_200d.index, self.vwap_200d, label='200D VWAP')
        ax.fill_between(self.vwap_20d[~self.vwap_200d.isna()].index, self.vwap_20d_upper_1[~self.vwap_200d.isna()], self.vwap_20d_lower_2[~self.vwap_200d.isna()], color='gray', alpha=0.3)
        ax.fill_between(self.vwap_20d[~self.vwap_200d.isna()].index, self.vwap_20d_upper_2[~self.vwap_200d.isna()], self.vwap_20d_lower_2[~self.vwap_200d.isna()], color='gray', alpha=0.15)
        ax.set_title(f'{self.etf_code.upper()}: daily price vs daily VWAPs')
        ax.grid(True, alpha=0.5)
        ax2 = ax.twinx()
        ax2.bar(df_plot['date'], df_plot['volume'], alpha=0.3, color='orange', label='Daily Volume')
        ax2.set_ylabel('Volume')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.legend()
        plt.show()
        
    def output(self, interactive_plot: bool = False, weekly_chart: bool = False):    
        '''        
        Call print_info and plot_chart to output result
        '''
        self.print_info()
        self.plot_daily_chart(interactive_plot)
        if weekly_chart:
            self.plot_weekly_chart()

    def return_result(self):
        return self.df

    def latest_metric(self, realtime=True, imputed_value=None, print_result=True):
        '''
        Pulling latest metrics of RSI and MACD, using latest realtime etf price
        If realtime=False: impute latest close price or provided value instead
        '''
        if realtime:
            new_price = self.ticker_close_price
        elif imputed_value is None:
            new_price = self.df.tail(1)['close'].item() # Use .item() to get scalar
        else:
            new_price = imputed_value

        # Make sure 'date' column is datetime before operations
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Create a DataFrame for the new day's data
        # Ensure the date for today is a datetime object
        today_date = datetime.now(BEIJING_TZ)
        new_day_df = pd.DataFrame({
            'date': [today_date],
            'close': [new_price]
        })

        # Concatenate and re-calculate
        df_check = pd.concat([self.df, new_day_df], ignore_index=True)

        df_check['12 Day EMA'] = df_check['close'].ewm(span=12, adjust=False).mean()
        df_check['26 Day EMA'] = df_check['close'].ewm(span=26, adjust=False).mean()
        df_check['MACD'] = df_check['12 Day EMA'] - df_check['26 Day EMA']
        df_check['MACD_signal'] = df_check['MACD'].ewm(span=9, adjust=False).mean()
        delta = df_check['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_check['RSI'] = 100 - (100 / (1 + rs))
        latest_rsi = round(df_check.tail(1)['RSI'].item(), 2)
        self.curr_rsi = latest_rsi
        latest_macd = round(df_check.tail(1)['MACD'].item() - df_check.tail(1)['MACD_signal'].item(), 4)
        self.curr_macd = latest_macd
        if print_result:
            if latest_rsi > 70:
                print("Current RSI:", Fore.RED + str(latest_rsi), Style.RESET_ALL)
            elif latest_rsi < 30:
                print("Current RSI:", Fore.GREEN + str(latest_rsi), Style.RESET_ALL)
            else:
                print("Current RSI:", latest_rsi, Style.RESET_ALL)
            if latest_macd < 0:
                print("Current MACD Divergence:", Fore.RED + str(latest_macd), Style.RESET_ALL)
            elif latest_macd > 0:
                print("Current MACD Divergence:", Fore.GREEN + str(latest_macd), Style.RESET_ALL)
            else:
                print("Current MACD Divergence:", Fore.BLACK + str(latest_macd), Style.RESET_ALL)

    def infer_metric(self, realtime=True, imputed_value=None, print_result=True):
        '''
        Assuming the current etf price holds for another day, what would be MACD or RSI?
        If realtime=False: impute latest close price or provided value instead
        '''
        if realtime:
            new_price = self.ticker_close_price
        elif imputed_value is None:
            new_price = self.df.tail(1)['close'].item()
        else:
            new_price = imputed_value

        # Make sure 'date' column is datetime before operations
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Create DataFrame for two new days (today and tomorrow)
        today_date = datetime.now(BEIJING_TZ)
        tomorrow_date = datetime.now(BEIJING_TZ) + relativedelta(days=1)
        two_days_df = pd.DataFrame({
            'date': [today_date, tomorrow_date],
            'close': [new_price, new_price] # Assume price holds for tomorrow
        })

        # Concatenate and re-calculate
        df_check = pd.concat([self.df, two_days_df], ignore_index=True)

        df_check['12 Day EMA'] = df_check['close'].ewm(span=12, adjust=False).mean()
        df_check['26 Day EMA'] = df_check['close'].ewm(span=26, adjust=False).mean()
        df_check['MACD'] = df_check['12 Day EMA'] - df_check['26 Day EMA']
        df_check['MACD_signal'] = df_check['MACD'].ewm(span=9, adjust=False).mean()
        delta = df_check['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_check['RSI'] = 100 - (100 / (1 + rs))
        latest_rsi = round(df_check.tail(1)['RSI'].item(), 2)
        self.infer_rsi = latest_rsi
        latest_macd = round(df_check.tail(1)['MACD'].item() - df_check.tail(1)['MACD_signal'].item(), 4)
        self.infer_macd = latest_macd
        if print_result:
            if latest_rsi > 70:
                print("Tomorrow inferred RSI:", Fore.RED + str(latest_rsi), Style.RESET_ALL)
            elif latest_rsi < 30:
                print("Tomorrow inferred RSI:", Fore.GREEN + str(latest_rsi), Style.RESET_ALL)
            else:
                print("Tomorrow inferred RSI:", latest_rsi, Style.RESET_ALL)
            if latest_macd < 0:
                print("Tomorrow inferred MACD Divergence:", Fore.RED + str(latest_macd), Style.RESET_ALL)
            elif latest_macd > 0:
                print("Tomorrow inferred MACD Divergence:", Fore.GREEN + str(latest_macd), Style.RESET_ALL)
            else:
                print("Tomorrow inferred MACD Divergence:", Fore.BLACK + str(latest_macd), Style.RESET_ALL)