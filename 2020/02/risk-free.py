import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import urllib
import datetime


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


def interpolate(returns, annualized_total_ret, eps, to_log=True):
    """
    Linear interpolation between the original returns
    and the annualized series.

    Parameters
    ----------
    returns : pandas.DataFrame
    annualized_total_ret : float
    eps : float
    to_log : bool
        If True is passed, converts inputs into logarithmic. Otherwise, use
        the inputs as if they are logarithmic.

    Returns
    -------
    interpolated : pd.DataFrame
    """
    if to_log:
        returns = returns.log()
        annualized_total_ret = np.log(1 + annualized_total_ret)
        _ret_interpolation = returns * (1 - eps) + eps * annualized_total_ret
        return _ret_interpolation.r2r(to='arithmetic')

    _ret_interpolation = returns * (1 - eps) + eps * annualized_total_ret
    return _ret_interpolation


def cum_returns(returns):
    returns = returns.copy()
    returns.iloc[0] = 0.0
    return returns.add(1.0).cumprod().mul(100)


def price2returns(price, periods):
    return price.pct_change(periods)


def price2cum_returns(price, periods):
    ret = price2returns(price, periods)
    cret = returns2cum_returns(ret, price.iloc[0])
    return cret


def returns(price, periods):
    return price.pct_change(periods)


def returns2cum_returns(returns, x0=100):
    x = returns.copy()
    x.iloc[0] = 0.0
    cum_ret = x.add(1.0).cumprod().mul(x0)
    return cum_ret


def total_returns(returns):
    tot_ret = returns.iloc[-1] / returns.iloc[0] - 1
    return tot_ret


def annualize(x, factor, kind="returns"):
    if kind == "returns":
        ann = x.add(1).pow(factor).sub(1)
    elif kind == "volatility":
        ann = x * factor

    return ann


if __name__ == "__main__":
    plt.ion()
    close = get_symbol_close("ES=F").ffill()
    # close = get_symbol_close("GOOG").ffill()
    # close = get_symbol_close("GOOG").ffill()
    n = close.shape[0]
    rets = price2returns(close, 1)
    mu = rets.mean().values
    sigma = rets.std().values

    x0 = close.iloc[0, 0]
    dt = np.sqrt(1 / n)
    np.random.seed(1)

    w = np.random.normal(0, np.sqrt(dt), size=(n, 500))
    x = np.exp((mu - (sigma ** 2) / 2) * dt + sigma * w)
    x[0, :] = 1
    x = returns2cum_returns(x, x0)

    sims = pd.DataFrame(x, index=close.index)

    ax = sims.plot(color="grey", alpha=0.4, legend=False)
    close.plot(ax=ax, color="red")

    total_returns(sims)
    sims_rets = price2returns(sims, 1)
    sims_rets.mean().mean(), mu
    sims_rets.var().mean(), sigma
