# coding: utf-8
import os
cd = os.getcwd(); cd
os.listdir(cd)
sys.path.append(os.path.join(cd, 'marzo_14'))
get_ipython().run_line_magic('matplotlib', 'qt5')
from tools import *
from hurst_ets.hurst import *
from hurst_ets.utils import *

coin_sym_list = 'BTC ETH DASH LTC'.split()
prices = get_symbol_close(coin_sym_list)
prices = prices.loc['2014':'2017']  # select the same period
prices = prices.where(prices != 0., np.nan)  # convert zeros into NaN's
prices.head()
prices.tail()

# plot price series
kw = dict(figsize=(13,8), grid=True, subplots=True, layout=(2,2), linewidth=1)
axs = prices.plot(**kw)
[ax.set_ylabel('In dollars ($)') for ax in axs.flat[::2]]
# plt.suptitle('Close Price by Currency', y=0.95);

# table hurst with entire series
hurst_methods = [
    prices.apply(lambda x: hurst_exponent(x.dropna(), method='DSOD')).to_frame().rename(columns={0: 'DSOD'}),
    prices.apply(lambda x: hurst_exponent(x.dropna(), method='RS')).to_frame().rename(columns={0: 'RS'}),
    prices.apply(lambda x: hurst_exponent(x.dropna(), method='DMA')).to_frame().rename(columns={0: 'DMA'}),
]
hurst_table = pd.concat(hurst_methods, axis=1)

# the same methodology as shown in the paper.
roll_days = 300
roll_prices = prices.rolling(roll_days)
summary = [
    roll_prices.apply(lambda s: hurst_exponent(s, method='RS')),
    roll_prices.apply(lambda s: hurst_exponent(s, method='DSOD')),
    ]
roll_hurst = pd.concat(summary, axis=1, keys=['RS', 'DSOD'])
roll_hurst = roll_hurst.swaplevel(axis=1)\
                       .asfreq(res.index.inferred_freq)\
                       .sort_index(axis=1, level=0)

roll_hurst[::50].tail()
roll_hurst[::50].mean()
roll_hurst[::50].median()
roll_hurst[::50].mean(level=0, axis=1)

fig, axs = plt.subplots(2,2)
for ax, coin in zip(axs.flat, coin_sym_list):
    roll_hurst[coin][::50].plot(ax=ax)
    ax.set_title(coin)
    ax.set_yticks(np.arange(11)/10)
    ax.grid(True, linestyle='--')
    ax.axhline(0.5, linestyle='--', color='k', alpha=0.5)


# rather decimate the series we aliasing the seies.
ma_roll_hurst = roll_hurst.rolling(50).mean()
ma_roll_hurst.tail()
ma_roll_hurst.plot(**kw)

# whats happens if we increase the windows size
roll_days = prices.count(0).min() - 50 # days
prices.rolling(roll_days).apply(lambda x: x.shape[0])
axs = roll_hurst.plot(**kw)
[ax.axhline(0.5, linestyle='--', color='k') for ax in axs.flat];
plt.suptitle('%s-Days Rolling Mean of Rolling Hurst Over Close Prices' % roll_days, y=0.95);


