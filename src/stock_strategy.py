import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
from colorama import Fore, Style
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objects as go

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import fsolve
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from math import ceil, floor

from utils import plot_candlestick

def stock_correlation(
    stock_name1: str,
    stock_name2: str,
    start: datetime = (datetime.today() - relativedelta(years=3)).strftime('%Y-%m-%d'),
    end: datetime = datetime.today().strftime('%Y-%m-%d'),
    ):
    '''
    For two corrlated stocks, if one has very short history, we can use this to fit its historical price 
    Return: dataframe of the second stock_name
    '''
    df = yf.download(stock_name1.upper(), start=start, end=end)
    df = df.reset_index()
    df.columns = df.columns.str.lower()

    df_2 = yf.download(stock_name2.upper(), start=start, end=end)
    df_2 = df_2.reset_index()
    df_2.columns = df_2.columns.str.lower()

    print(datetime.today().strftime('%Y-%m-%d'))

    df_merged = pd.merge(df[['date','open','high', 'low','close']],
                  df_2[['date','open','high', 'low','close']],
                  on='date')
    s1 = df_merged['close_x']
    s2 = df_merged['close_y']
    X_b = np.c_[np.ones((len(s1), 1)), s1]  # Add a bias term (intercept) to feature matrix
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(s2)
    # print("Correlation and R2:", s1.corr(s2), r2_score(s2, intercept+slope*s1))
    # print("Regression coeffs:", theta_best[0], theta_best[1])
    df['close'] = theta_best[0] + theta_best[1]*df['close']

    s1 = df_merged['open_x']
    s2 = df_merged['open_y']
    X_b = np.c_[np.ones((len(s1), 1)), s1]  # Add a bias term (intercept) to feature matrix
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(s2)
    # print("Correlation and R2:", s1.corr(s2), r2_score(s2, intercept+slope*s1))
    # print("Regression coeffs:", theta_best[0], theta_best[1])
    df['open'] = theta_best[0] + theta_best[1]*df['open']

    s1 = df_merged['low_x']
    s2 = df_merged['low_y']
    X_b = np.c_[np.ones((len(s1), 1)), s1]  # Add a bias term (intercept) to feature matrix
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(s2)
    # print("Correlation and R2:", s1.corr(s2), r2_score(s2, intercept+slope*s1))
    # print("Regression coeffs:", theta_best[0], theta_best[1])
    df['low'] = theta_best[0] + theta_best[1]*df['low']

    s1 = df_merged['high_x']
    s2 = df_merged['high_y']
    X_b = np.c_[np.ones((len(s1), 1)), s1]  # Add a bias term (intercept) to feature matrix
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(s2)
    # print("Correlation and R2:", s1.corr(s2), r2_score(s2, intercept+slope*s1))
    # print("Regression coeffs:", theta_best[0], theta_best[1])
    df['high'] = theta_best[0] + theta_best[1]*df['high']
    return df

def get_optimum_clusters(
    df: pd.DataFrame,
    saturation_point: float):
    '''
    This can be put into utils since it's a function
    param df: dataframe
    param saturation_point: The amount of difference we are willing to detect
    return: clusters with optimum K centers

    This method uses elbow method to find the optimum number of K clusters
    We initialize different K-means with 1..10 centers and compare the inertias
    If the difference is no more than saturation_point, we choose that as K and move on
    '''
    wcss = []
    k_models = []

    size = min(11, len(df.index))
    for i in range(1, size):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss)-1
    for i in range(0, len(wcss)-1):
        diff = abs(wcss[i+1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    # knee = KneeLocator(range(1, size), wcss, curve='convex', direction='decreasing')
    # optimum_k = knee.knee

    # print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = k_models[optimum_k]
    return optimum_clusters

class stock_strategy:
    """
    class to print and plot stock trading strategy
    strategy: allowed values are 'longterm', 'daily', 'test'
    """
    def __init__(
        self,
        stock_name: str,
        start: datetime = (datetime.today() - relativedelta(years=3)).strftime('%Y-%m-%d'),
        end: datetime = datetime.today().strftime('%Y-%m-%d'),
        saturation_point: float = 0.05,
        impute: bool = False,
        strategy: str = 'longterm',
        ):
        self.stock_name = stock_name
        self.strategy = strategy
        if self.stock_name == 'fbtc' and impute:
            self.df = stock_correlation(stock_name1='BTC-USD', stock_name2='fbtc')
        else:    
            self.df = yf.download(self.stock_name.upper(),
                start=start,
                end=end)
            self.df = self.df.reset_index()
            self.df.columns = self.df.columns.str.lower()
        self.ticker = yf.Ticker(self.stock_name.upper()).history(period='1d')
        self.calculate_ema()
        self.create_bb()
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
        self.df['Lower Band - 20MA'] = self.df['20 Day MA'] - (self.df['20 Day STD'] * 2)
        self.df['50 Day MA'] = self.df['close'].rolling(window=50).mean()
        self.df['50 Day STD'] = self.df['close'].rolling(window=50).std()
        self.df['Upper Band - 50MA'] = self.df['50 Day MA'] + (self.df['50 Day STD'] * 2.5)
        self.df['Lower Band - 50MA'] = self.df['50 Day MA'] - (self.df['50 Day STD'] * 2.5)

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
        previous_day = self.df[self.df['date']<datetime.today().strftime('%Y-%m-%d')]['date'].max()
        if self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)].shape[0] > 0:
            df_plot = self.df[self.df['date'] >= self.df['date'].min() + relativedelta(years=1)]
        else:
            df_plot = self.df
        close = df_plot[df_plot['date']==previous_day]['close'].item()
        if self.strategy == 'daily':
            try:
                support = max([e[0] for e in self.low_centers if e < close])
            except:
                print('Break all support; record min stock price')
                support = df_plot['low'].min()
            try:
                resistance = min([e[0] for e in self.high_centers if e > close])
            except:
                print('Break all resistance; record max stock price')
                resistance = df_plot['high'].max()
            print('* previous stock price closing', round(close,2), '~ up', ceil(resistance*100)/100.0, ', down', floor(support*100)/100)
        elif self.strategy == 'longterm':
            print('* previous stock price closing', round(close,2))
        print("Latest 20 Day MA:", round(self.df[self.df['date']==previous_day]['20 Day MA'].item(), 2))
        print("Latest Lower Bollinger Band, 20MA:", round(self.df[self.df['date']==previous_day]['Lower Band - 20MA'].item(), 2))
        print("Latest Higher Bollinger Band, 20MA:", round(self.df[self.df['date']==previous_day]['Upper Band - 20MA'].item(), 2))
        print("Latest 50 Day MA:", round(self.df[self.df['date']==previous_day]['50 Day MA'].item(), 2))
        print("Latest Lower Bollinger Band, 50MA:", round(self.df[self.df['date']==previous_day]['Lower Band - 50MA'].item(), 2))
        print("Latest Higher Bollinger Band, 50MA:", round(self.df[self.df['date']==previous_day]['Upper Band - 50MA'].item(), 2))
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
        df = yf.download(self.stock_name.upper(),
                     start=(datetime.today() - relativedelta(days=200)).strftime('%Y-%m-%d'),
                     end=datetime.today().strftime('%Y-%m-%d')
                     )
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        # df = df[['close']]
        
        # Define the expression whose roots we want to find
        # fsolve is not satisfying; provide analytical solution
        
        last_19day_price = df['close'][-20:-1]
        last_49day_price = df['close'][-50:-1]
        last_199day_price = df['close'][-200:-1]

        # last_19day_price = df[df['date'].between(
        #     (datetime.today()-relativedelta(days=19)).strftime('%Y-%m-%d'),
        #     (datetime.today()-relativedelta(days=1)).strftime('%Y-%m-%d'))]['close']
        # last_49day_price = df[df['date'].between(
        #     (datetime.today()-relativedelta(days=49)).strftime('%Y-%m-%d'),
        #     (datetime.today()-relativedelta(days=1)).strftime('%Y-%m-%d'))]['close']
        
        # func_20MA = lambda price : (np.sum(last_19day_price) + price)/20 - price
        # func_50MA = lambda price : (np.sum(last_49day_price) + price)/50 - price
        # func_20MA_UBB = lambda price : (np.sum(last_19day_price) + price)/20 + 2*np.sqrt((np.sum(last_19day_price**2) + price**2 - (np.sum(last_19day_price) + price)**2/20)/19) - price
        # func_20MA_LBB = lambda price : (np.sum(last_19day_price) + price)/20 - 2*np.sqrt((np.sum(last_19day_price**2) + price**2 - (np.sum(last_19day_price) + price)**2/20)/19) - price
        # func_50MA_UBB = lambda price : (np.sum(last_49day_price) + price)/50 + 2.5*np.sqrt((np.sum(last_49day_price**2) + price**2 - (np.sum(last_49day_price) + price)**2/50)/49) - price
        # func_50MA_LBB = lambda price : (np.sum(last_49day_price) + price)/50 - 2.5*np.sqrt((np.sum(last_49day_price**2) + price**2 - (np.sum(last_49day_price) + price)**2/50)/49) - price

        # func_20MA = lambda price : np.mean(np.append(last_19day_price, price)) - price
        # func_50MA = lambda price : np.mean(np.append(last_49day_price, price)) - price
        # func_20MA_UBB = lambda price : np.mean(np.append(last_19day_price, price)) + 2 * np.std(np.append(last_19day_price, price), ddof=1) - price
        # func_20MA_LBB = lambda price : np.mean(np.append(last_19day_price, price)) - 2 * np.std(np.append(last_19day_price, price), ddof=1) - price
        # func_50MA_UBB = lambda price : np.mean(np.append(last_49day_price, price)) + 2.5 * np.std(np.append(last_49day_price, price), ddof=1) - price
        # func_50MA_LBB = lambda price : np.mean(np.append(last_49day_price, price)) - 2.5 * np.std(np.append(last_49day_price, price), ddof=1) - price

        # price_initial_guess = df['close'].tolist()[-1]
        # price_solution = fsolve(func_20MA, price_initial_guess)
        # price_solution = fsolve(func_50MA, price_initial_guess)
        # price_solution = fsolve(func_20MA_UBB, price_initial_guess)
        # price_solution = fsolve(func_20MA_LBB, price_initial_guess)
        # price_solution = fsolve(func_50MA_UBB, price_initial_guess)
        # price_solution = fsolve(func_50MA_LBB, price_initial_guess)

        if (2*last_49day_price.sum() - 5*last_19day_price.sum())/3 > 0:
            print('20MA crosses 50MA at', round((2*last_49day_price.sum() - 5*last_19day_price.sum())/3, 2))
        if (last_199day_price.sum() - 4*last_49day_price.sum())/3 > 0:
            print('50MA crosses 200MA at', round((last_199day_price.sum() - 4*last_49day_price.sum())/3, 2))

        a1 = np.sum(last_19day_price)
        a2 = np.sum(last_19day_price**2)
        p_ma = np.mean(last_19day_price)
        p_ubb = (562*a1 + np.sqrt((562*a1)**2 - 4*(5339*(99*a1**2-1600*a2))))/5339/2
        p_lbb = (562*a1 - np.sqrt((562*a1)**2 - 4*(5339*(99*a1**2-1600*a2))))/5339/2
        
        print('20MA break point:', round(p_ma,2))
        print('20MA Lower Bollinger Band break point:', round(p_lbb,2))
        print('20MA Upper Bollinger Band break point:', round(p_ubb,2))

        a1 = np.sum(last_49day_price)
        a2 = np.sum(last_49day_price**2)
        p_ma = np.mean(last_49day_price)
        p_ubb = (4177*a1 + np.sqrt((4177*a1)**2 - 4*(102336.5*(361.5*a1**2-15625*a2))))/102336.5/2
        p_lbb = (4177*a1 - np.sqrt((4177*a1)**2 - 4*(102336.5*(361.5*a1**2-15625*a2))))/102336.5/2
        
        print('50MA break point:', round(p_ma,2))
        print('50MA Lower Bollinger Band break point:', round(p_lbb,2))
        print('50MA Upper Bollinger Band break point:', round(p_ubb,2))

    def plot_chart(self, interactive_plot):
        '''
        Plot the stock trading charts
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
            ax = plot_candlestick(df_plot, figsize=(32,8))
            ax.set_title(self.stock_name.upper())
            for low in self.low_centers[:]:
                ax.axhline(low[0], color='green', ls='--')
            for high in self.high_centers[:]:
                ax.axhline(high[0], color='red', ls='--')
            ax.plot(df_plot['date'], df_plot['20 Day MA'], ls='--', label='20 Day Moving Average')
            ax.plot(df_plot['date'], df_plot['Upper Band - 20MA'], ls='--', label='Upper Bollinger Band, 20MA')
            ax.plot(df_plot['date'], df_plot['Lower Band - 20MA'], ls='--', label='Lower Bollinger Band, 20MA')
            ax.fill_between(df_plot['date'], df_plot['Upper Band - 20MA'], df_plot['Lower Band - 20MA'], color='gray', alpha=0.3) # Fill the area between the bands
            ax.plot(df_plot['date'], df_plot['50 Day MA'], ls='--', label='50 Day Moving Average')
            ax.plot(df_plot['date'], df_plot['Upper Band - 50MA'], ls='--', label='Upper Bollinger Band, 50MA')
            ax.plot(df_plot['date'], df_plot['Lower Band - 50MA'], ls='--', label='Lower Bollinger Band, 50MA')
            ax.fill_between(df_plot['date'], df_plot['Upper Band - 50MA'], df_plot['Lower Band - 50MA'], color='gray', alpha=0.15) # Fill the area between the bands

            ax.legend()

        elif self.strategy == 'longterm' and interactive_plot:
            figsize=(12, 8)
            fig = go.Figure(layout=dict(width=figsize[0]*80, height=figsize[1]*80))
            
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['close'], name='Daily Price'))
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['50 Day MA'], mode='lines', line=dict(dash='dash'), name='50 Day MA'))
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Lower Band - 50MA'], mode='lines', line=dict(dash='dash'), name='Lower Band'))
            fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['Upper Band - 50MA'], mode='lines', line=dict(dash='dash'), name='Upper Band', fill='tonexty', fillcolor='rgba(128,128,128,0.3)'))
            
            # for low in low_centers[:]:
            #     fig.add_trace(go.Scatter(x=df['date'], y=[low[0]]*len(df['date']), mode='lines', line=dict(color='green', dash='dash'), name='Support at ' + str(low[0])))
            # for high in high_centers[:]:
            #     fig.add_trace(go.Scatter(x=df['date'], y=[high[0]]*len(df['date']), mode='lines', line=dict(color='red', dash='dash'), name='Resistance at ' + str(high[0])))        
            fig.update_layout(title='Interactive Plot of Daily Stock Price for ' + self.stock_name.upper(),
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
            ax1.plot(df_plot['date'], df_plot['close'], label='Daily Close Price')
            ax1.plot(df_plot['date'], df_plot['50 Day MA'], ls='--', label='50 Day Moving Average')
            ax1.plot(df_plot['date'], df_plot['Upper Band - 50MA'], ls='--', label='Upper Bollinger Band, 50MA')
            ax1.plot(df_plot['date'], df_plot['Lower Band - 50MA'], ls='--', label='Lower Bollinger Band, 50MA')
            ax1.fill_between(df_plot['date'], df_plot['Upper Band - 50MA'], df_plot['Lower Band - 50MA'], color='gray', alpha=0.3) # Fill the area between the bands
            ax1.set_title('Daily stock price for ' + self.stock_name.upper())
            ax1.legend()
                
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

    def latest_metric(self, realtime=True, imputed_value=None, print_result=True):
        '''
        Pulling latest metrics of RSI and MACD, using latest realtime stock price
        If realtime=False: impute latest close price or provided value instead
        '''
        if realtime:
            new_price = self.ticker['Close'].tolist()
        elif imputed_value is None:
            new_price = self.df.tail(1)['close'].tolist()
        else:
            new_price = [imputed_value]

        df_check = pd.concat([
            self.df,
            pd.DataFrame(
                {
                'date':[datetime.today().strftime('%Y-%m-%d')],
                'close':new_price}
                )
            ], ignore_index=True)
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
        latest_macd = round(df_check.tail(1)['MACD'].item() - df_check.tail(1)['MACD_signal'].item(), 2)
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
        Assuming the current stock price holds for another day, what would be MACD or RSI?
        If realtime=False: impute latest close price or provided value instead
        '''
        if realtime:
            new_price = self.ticker['Close'].tolist()
        elif imputed_value is None:
            new_price = self.df.tail(1)['close'].tolist()
        else:
            new_price = [imputed_value]

        df_check = pd.concat([
            self.df,
            pd.DataFrame(
                {
                'date':[datetime.today().strftime('%Y-%m-%d'), (datetime.today() + relativedelta(days=1)).strftime('%Y-%m-%d')],
                'close':new_price*2}
                )
            ], ignore_index=True)
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
        latest_macd = round(df_check.tail(1)['MACD'].item() - df_check.tail(1)['MACD_signal'].item(), 2)
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
