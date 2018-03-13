from scipy import signal
import numpy as np
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, HTML


def hurst_dsod(x):
    """Estimate Hurst exponent on data timeseries.

    The estimation is based on the discrete second order derivative. Consists on
    get two different noise of the original series and calculate the standard
    deviation and calculate the slope of two point with that values.
    source: https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc

    Parameters
    ----------
    x : numpy array
        time series to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given time series.

    References
    ----------
    Istas, J.; G. Lang (1994), “Quadratic variations and estimation of the local
    Hölder index of data Gaussian process,” Ann. Inst. Poincaré, 33, pp. 407–436.


    Notes
    -----
    This hurst_ets is data literal traduction of wfbmesti.m of waveleet toolbox from
    matlab.
    """
    y = np.cumsum(np.diff(x, axis=0), axis=0)

    # second order derivative
    b1 = [1, -2, 1]
    y1 = signal.lfilter(b1, 1, y, axis=0)
    y1 = y1[len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    b2 = [1,  0, -2, 0, 1]
    y2 = signal.lfilter(b2, 1, y, axis=0)
    y2 = y2[len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=0)
    s2 = np.mean(y2 ** 2, axis=0)

    h = 0.5 * np.log2(s2 / s1)

    # Ensure range
    if h > 1:
        h = 1
    elif h < 0:
        h = 0
    return h


def get_prices(coin_symbol):
    """Get close price.

    Given a symbol crytocurrency retrieve last 2k close prices in USD.

    Parameters
    ----------
    coin_symbol : str

    Returns
    -------
    price_close : pandas.DataFrame
    """
    endpoint = "https://min-api.cryptocompare.com/data/histoday"
    params = dict(fsym=coin_symbol, tsym="USD",limit=2000, aggregate=1)
    out = requests.get(endpoint, params=params).json()['Data']
    data = pd.DataFrame(out).set_index('time')\
                            .loc[:, ['close']]\
                            .rename(columns=dict(close=coin_symbol))
    return data


def get_symbol_close(coin_symbol_list):
    """Get symbol close.

    Given a list of cryptocurrencies symbols retrieve close prices.

    Parameters
    ----------
    coin_symbol_list : list

    Returns
    -------
    price_close : pandas.DataFrame
    """
    d = [get_prices(coin_sym) for coin_sym in coin_symbol_list]
    out = pd.concat(d, axis=1)
    out.index = out.index.map(datetime.utcfromtimestamp)
    return out.asfreq(out.index.inferred_freq)


def peak_detection(array, delta, position=None):
    """Detects peaks.

    Parameters
    ----------
    array : numpy.array
        Series where to find peaks.
    delta : float
        Threshold to determine peaks, in the same units that array.
    position : numpy.array
        Position stored instead of the original position when finds a peak.

    Returns
    -------
    maxtab : numpy.array
        Has two columns, the first one are indexes of the each max peak, the
        other one is the value.
    mintab : numpy.array
        Has two columns, the first one are indexes of the each max peak, the
        other one is the value.

    Notes
    -----
    This funciton originally find it on:
    https://gist.github.com/mmngreco/5c7abbf2c73a9e95eeb832720bdcc156.
    It was Converted from MATLAB script at http://billauer.co.il/peakdet.html.

    """
    maxtab = []
    mintab = []

    if position is None:
        position = np.arange(len(array), dtype='int64')
    if len(array) != len(position):
        raise ValueError('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        raise ValueError('Input argument delta must be a scalar')
    if delta <= 0:
        raise ValueError('Input argument delta must be positive')

    maximum = -np.Inf
    minimum = np.Inf
    minimum_pos = None
    maximum_pos = None

    lookformax = True

    for i, arr_i in enumerate(array):
        if arr_i > maximum:
            maximum = arr_i
            maximum_pos = position[i]
        if arr_i < minimum:
            minimum = arr_i
            minimum_pos = position[i]

        if lookformax:
            if arr_i < (maximum-delta):
                maxtab.append((maximum_pos, maximum))
                minimum = arr_i
                minimum_pos = position[i]
                lookformax = False
        else:
            if arr_i > (minimum+delta):
                mintab.append((minimum_pos, minimum))
                maximum = arr_i
                maximum_pos = position[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def filter_peaks(series, delta=0.1, which='max'):
    """Get indexes from desired peaks.

    Parameters
    ----------
    series : pandas.Series or numpy.array.
    delta :
    which : str, {'max', 'min', 'both'}

    Returns
    -------
    filtered_series : the same as input
    """
    if which == 'max':
        return filter_maxpeaks(series, delta)
    elif which == 'min':
        return filter_minpeaks(series, delta)
    elif which == 'both':
        return filter_maxpeaks(series, delta), filter_minpeaks(series, delta)
    else:
        raise ValueError('Unexpected value of which, allowed values: max , min , both.')
    return series.iloc[indxs]


def filter_maxpeaks(series, delta=0.1):
    arr = series.__array__()
    indxs = peak_detection(arr, delta)[0][:, 0]
    return series.iloc[indxs]


def filter_minpeaks(series, delta=0.1):
    arr = series.__array__()
    indxs = peak_detection(arr, delta)[1][:, 0]
    return series.iloc[indxs]



def plot_with_peaks(data, delta=0.1, ax=None, figsize=None, **kw):
    if data.ndim == 1:
        if ax is None:
            ax = data.plot(figsize=figsize, **kw)
        else:
            data.plot(**kw, ax=ax)
        maxpeaks, minpeaks = filter_peaks(data, delta=delta, which='both')
        data.loc[maxpeaks.index].plot(ax=ax, linestyle='', marker='x', color='r', label='max_peak')
        data.loc[minpeaks.index].plot(ax=ax, linestyle='', marker='x', color='g', label='min_peak')
        return ax
    if data.ndim == 2:
        cols = data.shape[1]
        ncols = int(np.ceil(np.sqrt(cols)))
        fig, axs = plt.subplots(ncols, ncols, figsize=figsize)
        for ax, ic in zip(axs.flatten(), range(cols)):
            _data = data.iloc[:, ic]
            plot_with_peaks(_data, delta=delta, ax=ax)
            ax.set_title('%s' % data.columns[ic])
        return axs



def multi_column_df_display(list_dfs, cols=3):
    html_table = "<table style='width:100%; border:0px'>{content}</table>"
    html_row = "<tr style='border:0px'>{content}</tr>"
    html_cell = "<td style='width:{width}%;vertical-align:top;border:0px'>{{content}}</td>"
    html_cell = html_cell.format(width=100/cols)
    cells = [html_cell.format(content=df.to_html()) for df in list_dfs]
    cells += (cols - (len(list_dfs) % cols)) * [html_cell.format(content="")]
    rows = [html_row.format(content="".join(cells[i:i+cols])) for i in range(0, len(cells), cols)]
    display(HTML(html_table.format(content="".join(rows))))
