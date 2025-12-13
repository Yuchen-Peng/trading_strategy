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

def 