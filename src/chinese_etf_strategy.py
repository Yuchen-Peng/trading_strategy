import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import akshare as ak
import yfinance as yf
from colorama import Fore, Style
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objects as go

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import fsolve, curve_fit
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from math import ceil, floor

from utils import plot_candlestick

