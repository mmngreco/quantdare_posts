import os
os.chdir('calcular_retornos')

# load packages
import requests as req
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
%matplotlib inline

# ploting setup
plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc('font', family='Times New Roman', size=15)
set_matplotlib_formats('png', 'png', quality=80)
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = 'cm'
plt.rcParams['axes.grid'] = True
kw_save = dict(bbox_iches='tight', transparent=True)

# asset information
asset_info = '''
Banco do Brasil S.A. (BBAS3.SA)
Sao Paolo - Sao Paolo Delayed Price. Currency in BRL.
Source: https://finance.yahoo.com/quote/BBAS3.SA
'''

# useful functions
# ================

# Total return function
def total_return(prices):
    """Retuns the return between the first and last value of the DataFrame.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame

    Returns
    -------
    total_return : float or pandas.Series
        Depending on the input passed returns a float or a pandas.Series.
    """
    return prices.iloc[-1] / prices.iloc[0] - 1


def total_return_from_returns(returns):
    """Retuns the return between the first and last value of the DataFrame.

    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame

    Returns
    -------
    total_return : float or pandas.Series
        Depending on the input passed returns a float or a pandas.Series.
    """
    return (returns + 1).prod() - 1


def plot_this(df, title, figsize=None, ylabel='',
             output_file='imgs/fig_rets_approach1.png', bottom_adj=0.25,
             txt_ymin=-0.4, bar=False):
    if bar:
        ax = df.plot.bar(title=title, figsize=figsize)
    else:
        ax = df.plot(title=title, figsize=figsize)
    sns.despine()
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.text(0, txt_ymin, asset_info, transform=ax.transAxes, fontsize=9)
    plt.gcf().subplots_adjust(bottom=bottom_adj)
    plt.savefig(output_file, **kw_save)

# Geting data
# ===========
today = '20170926'  # to make static this script.
tckr = 'BBAS3.SA'  # Banco do Brasil SA
# download data
data = pdr.get_data_yahoo(tckr, 2014, today)
data = data.asfreq('B')  # add frequency needed for some pandas functionalities releated with offsets
data.columns = data.columns.map(lambda col: col.lower())
data.head()  # first values
data.tail()  # last values

# what about NaNs
data.isnull().sum()
data.ffill(inplace=True)  # to avoid problems with NaNs.

# using close prices
prices = data.close.copy()
# we convert to DataFrame to make easy store more series.
results_storage = prices.to_frame().copy()

# plotting
plot_this(prices, title='Prices of %s' % tckr, ylabel='Prices in BRL',
          txt_ymin=-0.2, bottom_adj=0.15, output_file='imgs/fig_prices.png',)

# extract some date information
results_storage['year'] = prices.index.year
results_storage['month'] = prices.index.month
results_storage['day'] = prices.index.day
results_storage['week_day'] = prices.index.dayofweek
results_storage['week_day_name'] = prices.index.strftime('%A')
results_storage.tail(10)

# ================
# Trailing Returns
# ================

# Approach 1: starting from prices
# ================================
approach1 = results_storage.groupby(['year', 'month'], )['close'].apply(total_return)
approach1.tail(10)

# ploting
# -------
plot_this(approach1, bar=True, title='Trailing returns: Approach 1',
          ylabel='Returns (parts per unit)', txt_ymin=-0.4, bottom_adj=0.25,
          output_file='imgs/fig_rets_approach1.png')

# Nota bene: What means approach 1:
# means that we are selecting all available prices INSIDE a month and then we
# calculate the total return with that prices.
select_idx = (2017, 8)
idx_approach1 = results_storage.groupby(['year', 'month'])['close'].groups[select_idx]
last_group = results_storage.loc[idx_approach1]
last_group.head()
last_group.tail()

# example of the calculation
total_return(last_group.close)
approach1.loc[select_idx]

# Approach 2: starting from daily returns
# =======================================
r = prices.pct_change()
approach2 = r.groupby((r.index.year, r.index.month))\
             .apply(total_return_from_returns)

approach2.tail(10)

plot_this(approach2, bar=True, title='Trailing returns: Approach 2',
          ylabel='Returns (parts per unit)', txt_ymin=-0.4, bottom_adj=0.25,
          output_file='imgs/fig_rets_approach2.png')

# However, this approximation is almost correct since we have started from 2014
# prices, therefore it is not possible to calculate the return of the first
# month of that first available year.
# This same situation occurs in the last month available (current), as no data
# are available for the last day of the month, the return is also not
# comparable with the rest.



# Approach 3: Now yes, the definitive approach
# ============================================
# ... what mean approach3, first trimm prices according to end of month this
# then with that prices calculate returns. So we are calculating returns between
# different months and mostly in differents days whichs always are the last
# bussiness day of the month.

# with "asfreq" we decimate the prices, then group by year and month, so we
# have all the prices at the end of the working month available in the DataFrame.
# Finally we calculate the return of this new series with "pct_change".
approach3 = results_storage.asfreq('BM')\
                           .set_index(['year', 'month'])\
                           .close\
                           .pct_change()
approach3.tail(10)
plot_this(approach3, bar=True, title='Trailing returns: Approach 3',
          ylabel='Returns (parts per unit)', txt_ymin=-0.4, bottom_adj=0.25,
          output_file='imgs/fig_rets_approach3.png')


# Comparing all approaches
# ========================
all_approaches = pd.concat([approach1, approach2, approach3], axis=1,
                           keys=['approach1', 'approach2', 'approach3'])

plot_this(all_approaches, title='Comparing all approaches',
          output_file='imgs/all_approaches.png', bar=True,
          ylabel='Returns (parts per unit)', figsize=(15,8), bottom_adj=0.2,
          txt_ymin=-0.3)



# 
