# How to annualize properly

## Returns

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import urllib
import datetime

import numpy as np
import pandas as pd


def get_symbol_close(symbol):

    if isinstance(symbol, list):
        out = [get_symbol_close(s) for s in symbol]
        return pd.concat(out, axis=1)

    start = datetime.datetime.today()
    period1 = int((start - datetime.timedelta(days=365)).timestamp())
    period2 = int((start - datetime.timedelta(days=0)).timestamp())

    _symbol = urllib.parse.quote_plus(symbol)

    url_fmt = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?"
        "symbol={symbol}"
        "&period1={period1}"
        "&period2={period2}"
        "&interval=1d"
        # "&includePrePost=true"
        # "&events=div%7Csplit%7Cearn"
        # "&lang=en-US"
        # "&region=US"
        # "&crumb=E7fq3l8OFQo"
        # "&corsDomain=finance.yahoo.com"
    )

    url = url_fmt.format(symbol=_symbol, period1=period1, period2=period2)
    res = urllib.request.urlopen(url)
    res_json = res.read()
    res_dict = pd.io.json.loads(res_json)

    # to pandas
    data = res_dict["chart"]["result"][0]
    index = np.array(data["timestamp"], dtype="datetime64[s]")
    values = data["indicators"]["adjclose"][0]["adjclose"]
    data = pd.DataFrame(values, index=index, columns=[symbol])

    return data


def total_return(prices, relative=True):

    t0 = df.first_valid_index()
    tf = df.last_valid_index()

    if relative:
        tot_ret = df.loc[tf] / df.loc[t0] - 1
    else:
        tot_ret = df.loc[tf] - df.loc[t0]

    return tot_ret


df = get_symbol_close(["SPY"])
rets = df.pct_change()
rets_avg = rets.copy()
df.iloc[[0,-1]]
rets_avg.loc[1:] = (total_return(df).iloc[0] + 1) ** (261 / df.shape[0]) - 1
(rets_avg.fillna(0)+1).cumprod()
(rets.fillna(0)+1).cumprod()

nrows = rets.shape[0]
angles = 2 * pi * np.arange(nrows) / nrows
plt.ion()

ax = plt.subplot(111, polar=True)
plt.xticks(
    angles[::28], rets.index[::28].strftime("%Y-%m-%d"), color='grey', size=8
)
ax.set_rlabel_position(0)
line_ret = ax.plot(
    angles,
    rets.values.squeeze(),
    linewidth=1,
    linestyle='solid',
    label="SPY daily returns",
    color='#1f77b4',
)
line_ret_avg = ax.plot(
    angles,
    rets_avg.values.squeeze(),
    linewidth=1,
    linestyle='solid',
    label="SPY avgerage daily returns",
    color='red',
)
ax.scatter(
    angles[0],
    rets.values.squeeze()[0],
    marker="^",
    label="Start",
    color=line_ret[0].get_color(),
)
ax.scatter(
    angles[-1],
    rets.values.squeeze()[-1],
    marker="o",
    label="End",
    color=line_ret[0].get_color(),
)
plt.legend()
