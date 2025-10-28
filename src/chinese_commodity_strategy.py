import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

class commodity_strategy:
    """
    class to print and plot commodity trading strategy
    strategy: allowed values are 'longterm', 'daily', 'test'
    """
    def __init__(
        self,
        commodity_code: str,
        start: datetime = (datetime.today() - relativedelta(years=3)).strftime('%Y-%m-%d'),
        end: datetime = datetime.today().strftime('%Y-%m-%d'),
        saturation_point: float = 0.05,
        impute: bool = False,
        strategy: str = 'longterm',
        ):
        self.commodity_code = commodity_code
        self.strategy = strategy

        print(f"Downloading data for {self.commodity_code} from {start} to {end} using akshare...")
        self.df = ak.spot_hist_sge(
            symbol=self.commodity_code,
        )
        self.df['date'] = pd.to_datetime(self.df['date']) # Convert 'date' column to datetime objects
        self.df = self.df[self.df['date'].between(pd.to_datetime(start), pd.to_datetime(end))]

        # For the latest price, call ak.spot_quotations_sge
        today_str = datetime.today().strftime('%Y%m%d')
        latest_data = ak.spot_quotations_sge(
            symbol=self.commodity_code,
        )
        if not latest_data.empty:
            self.ticker_close_price = latest_data[latest_data['时间'] == latest_data['时间'].max()]['现价']
        else:
            print(f"Could not retrieve real-time price for {self.commodity_code}. Using last available close price.")
            self.ticker_close_price = self.df['close'].iloc[-1]


        self.calculate_ema()
        self.create_bb()
        self.create_weekly()
        self.calculate_rsi()
        self.create_macd()
        if self.strategy == 'daily':
            self.low_centers, self.high_centers = self.support_and_resistance(saturation_point)

    def calculate_ema(self):
        '''
        Calculate 12 Day EMA & 26 Day EMA (for MACD), and 50 Day EMA & 200 Day EMA
        '''
        self.df['12 Day EMA'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['26 Day EMA'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['50 Day EMA'] = self.df['close'].ewm(span=50, adjust=False).mean()
        self.df['200 Day EMA'] = self.df['close'].ewm(span=200, adjust=False).mean()

    def create_bb(self):
        '''
        Create 20 Day MA, 50 Day MA, and their corresponding Bollinger Bands
        '''
        self.df['20 Day MA'] = self.df['close'].rolling(window=20).mean()
        self.df['20 Day STD'] = self.df['close'].rolling(window=20).std()
        self.df['Upper Band - 20MA'] = self.df['20 Day MA'] + (self.df['20 Day STD'] * 2)
        self.df['lower Band - 20MA'] = self.df['20 Day MA'] - (self.df['20 Day STD'] * 2)
        self.df['50 Day MA'] = self.df['close'].rolling(window=50).mean()
        self.df['50 Day STD'] = self.df['close'].rolling(window=50).std()
        self.df['Upper Band - 50MA'] = self.df['50 Day MA'] + (self.df['50 Day STD'] * 2.5)
        self.df['lower Band - 50MA'] = self.df['50 Day MA'] - (self.df['50 Day STD'] * 2.5)
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

        # Calculate weekly summary, so candlestick plot can be generated at weekly level
        self.weekly_summary = (self.df[['date', 'open', 'high', 'low', 'close']]
                               .set_index('date').resample('W-FRI')
                               .agg({
                                   'open': 'first',
                                   'high': 'max',
                                   'low': 'min',
                                   'close': 'last'
                                   })
                              )
        
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
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)]
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

    def print_info(self):
        '''
        Print out the information needed
        '''
        previous_day = self.df[self.df['date']<datetime.today().strftime('%Y%m%d')]['date'].max()
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)]
        else:
            df_plot = self.df
        new_price = self.ticker_close_price # Use the stored latest price
        if self.strategy == 'daily':
            try:
                support = max([e[0] for e in self.low_centers if e < new_price])
            except:
                print('Break all support; record min commodity price')
                support = df_plot['low'].min()
            try:
                resistance = min([e[0] for e in self.high_centers if e > new_price])
            except:
                print('Break all resistance; record max commodity price')
                resistance = df_plot['high'].max()
            print('* Current commodity price:', round(new_price,2), '~ up', ceil(resistance*100)/100.0, ', down', floor(support*100)/100)
        elif self.strategy == 'longterm':
            print('* Current commodity price:', round(new_price,2))
        print('* Recent high:', round(df_plot['high'].max(),2))
        print('* Current commodity price is at ' + str(100*round(new_price/df_plot['high'].max(),4)) + '% of recent high')
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

        latest_macd = round(self.df[self.df['date']==previous_day]['MACD_diff'].item(), 2)
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
        df = ak.spot_hist_sge(
            symbol=self.commodity_code,
        )
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].between(datetime.today() - relativedelta(days=300), datetime.today())]

        # Define the expression whose roots we want to find
        # fsolve is not satisfying; provide analytical solution

        last_19day_price = df['close'][-20:-1]
        last_49day_price = df['close'][-50:-1]
        last_119day_price = df['close'][-120:-1]
        last_199day_price = df['close'][-200:-1]

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

    def plot_chart(self, interactive_plot):
        '''
        Plot the commodity trading charts
        Including:
        * SMA
        * EMA
        * Bollinger Bands
        * Support and Resistance
        * MACD
        '''
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)]
        else:
            df_plot = self.df
        if self.strategy == 'daily':
            ax = plot_candlestick(df_plot, figsize=(32,16))
            ax.set_title(self.commodity_code, fontsize=32)
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
            
            fig.update_layout(title='Interactive Plot of Daily commodity Price for ' + self.commodity_code,
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
            ax1.set_title('Daily commodity price for ' + self.commodity_code, fontsize=32)
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
            
    def plot_weekly_chart(self, candlestick=False):
        '''
        Plot the weekly etf trading charts
        Including:
        * 10 - 40 weekly MA
        * 20 weekly Bollinger Bands
        '''
        if candlestick:
            df_plot = self.weekly_summary[~self.weekly_ma20.isna()].reset_index()
            ax = plot_candlestick(df_plot, figsize=(8,4.8))
        else:
            fig, ax = plt.subplots(figsize=(8,4.8))
            ax.grid(True, alpha=0.5)
            ax.plot(self.weekly[~self.weekly_ma20.isna()].index, self.weekly[~self.weekly_ma20.isna()], label="Weekly Close Price", color="black", linewidth=1)
        ax.plot(self.weekly_ma10[~self.weekly_ma20.isna()].index, self.weekly_ma10[~self.weekly_ma20.isna()], label="10-week MA (~50-day)", ls='--', color="blue", linewidth=1)
        ax.plot(self.weekly_ma30.index, self.weekly_ma30, label="30-week MA (~150-day)", ls='--', color="orange", linewidth=1)
        ax.plot(self.weekly_ma40.index, self.weekly_ma40, label="40-week MA (~200-day)", ls='--', color="red", linewidth=1)
        ax.plot(self.weekly_ma20.index, self.weekly_ma20, label="20-week MA (BB mid)", ls='--', color="green", linewidth=1)
        ax.fill_between(self.weekly_bb_upper.index, self.weekly_bb_lower, self.weekly_bb_upper, color="gray", alpha=0.2, label="20-week Bollinger Band")
        
        ax.set_title(f'Weekly commodity price for {self.commodity_code.upper()}')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        
    def output(self, interactive_plot: bool = False):    
        '''        
        Call print_info and plot_chart to output result
        '''
        self.print_info()
        self.plot_chart(interactive_plot)

    def return_result(self):
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)]
        else:
            df_plot = self.df
        return df_plot