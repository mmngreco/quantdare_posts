"""
How to use TensorFlow to obtain derivatives of price option.
For this post I used tensorflow v2 which has a nicer API.

https://en.wikipedia.org/wiki/Greeks_(finance)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint

DTYPE = tf.float32
SEED = 3232


def initialize_variables(
    S0=100,
    strike=110,
    time_to_expiry=2,
    implied_vol=0.2,
    riskfree=0.03,
    to_tf=False
):
    if to_tf:
        S0 = tf.Variable(S0, dtype=DTYPE)
        strike = tf.Variable(strike, dtype=DTYPE)
        time_to_expiry = tf.Variable(time_to_expiry, dtype=DTYPE)
        implied_vol = tf.Variable(implied_vol, dtype=DTYPE)
        riskfree = tf.Variable(riskfree, dtype=DTYPE)

    out = dict(
            S0=S0,
            strike=strike,
            time_to_expiry=time_to_expiry,
            implied_vol=implied_vol,
            riskfree=riskfree,
        )

    return out


# ============================================================================
# Black-Scholes


@tf.function
def pricer_blackScholes(S0, strike, time_to_expiry, implied_vol, riskfree):
    """pricer_blackScholes.

    Parameters
    ----------
    S0 : float
    strike : float
    time_to_expiry : float
    implied_vol : float
    riskfree : float

    Returns
    -------
    npv : float
        Net present value.

    Examples
    --------
    >>> kw = initialize_variables(to_tf=True)
    >>> pricer_blackScholes(**kw)

    Notes
    -----
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula
    """
    S       = S0
    K       = strike
    dt      = time_to_expiry
    dt_sqrt = tf.sqrt(dt)
    sigma   = implied_vol
    r       = riskfree
    Phi     = tf.compat.v1.distributions.Normal(0., 1.).cdf

    d1 = (tf.math.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * dt_sqrt)
    d2 = d1 - sigma * dt_sqrt

    npv =  S * Phi(d1) - K * tf.exp(-r * dt) * Phi(d2)
    return npv


def calculate_blackScholes():
    """calculate_blackScholes.

    Returns
    -------
    out : dict
        npv : net presetn value
        dv : First order derivates
        d2v : Second order derivates
        d3v : Third order derivates

    Example
    -------
    >>> calculate_blackScholes()
    """
    variables = initialize_variables(to_tf=True)

    with tf.GradientTape() as g1:
        npv = pricer_blackScholes(**variables)
    dv = g1.gradient(npv, variables)

    dv = {k: v.numpy() for k,v in dv.items()}  # get the value
    return dict(npv=npv.numpy(), dv=dv)


# ============================================================================
# Monte Carlo


@tf.function
def brownian(S0, dt, sigma, mu, dw):
    """Generates a brownian motion.

    Parameters
    ----------
    S0 : float
        Initial value of Spot.
    dt : float
        Time step.
    sigma : float
        Volatility.
    mu : float
        Mean, in black Scholes frame it's the risk free rate.
    dw : numpy.array
        Random variable.

    Returns
    -------
    out : numpy.array

    Examples
    --------
    >>> nsims = 10
    >>> nobs = 400
    >>> v = initialize_variables(to_tf=True)
    >>> S0 = v["S0"]
    >>> dw = tf.random.normal((nsims, nobs), seed=SEED)
    >>> dt = v["time_to_expiry"] / dw.shape[1]
    >>> sigma = v["implied_vol"]
    >>> r = v["riskfree"]
    >>> paths = np.transpose(brownian(S0, dt, sigma, r, dw))
    >>> # to pandas
    >>> df = pd.DataFrame(paths)
    >>> df.mean()
    >>> df.plot()
    """
    dt_sqrt = tf.math.sqrt(dt)
    shock = sigma * dt_sqrt * dw
    drift = (mu - (sigma ** 2) / 2)
    bm = tf.math.exp(drift * dt + shock)
    out = S0 * tf.math.cumprod(bm, axis=1)
    return out


@tf.function
def pricer_montecarlo(S0, strike, time_to_expiry, implied_vol, riskfree, dw):
    """pricer_montecarlo.

    Parameters
    ----------
    S0 :
    strike :
    time_to_expiry :
    implied_vol :
    riskfree :
    dw :

    Returns
    -------
    npv : float
        Net present value.

    Examples
    --------
    """
    sigma = implied_vol
    T = time_to_expiry
    r = riskfree
    K = strike
    dt = T / dw.shape[1]

    st = brownian(S0, dt, sigma, r, dw)
    payout = tf.math.maximum(st[:, -1] - K, 0)
    npv = tf.exp(-r * T) * tf.reduce_mean(payout)

    return npv


def calculate_montecarlo(greeks=True):
    """calculate_montecarlo.

    Returns
    -------
    out : dict
        npv : Net present value
        dv : First order derivatives
        d2v : Second order derivatives

    Examples
    --------
    >>> calculate_montecarlo()
    """
    nsims = 10000000
    nobs = 2
    dw = tf.random.normal((nsims, nobs), seed=SEED)
    v = initialize_variables(to_tf=True)

    out = dict()

    if greeks:
        with tf.GradientTape() as g2:
            with tf.GradientTape() as g1:
                npv = pricer_montecarlo(**v, dw=dw)
            dv = g1.gradient(npv, v)
        d2v = g2.gradient(dv, v)

        out["dv"] = dv
        out["d2v"] = d2v
    else:
        npv = St(**v, dw=dw)

    out["npv"] = npv
    return out


def test_brownian():
    nsims = 100000000
    nobs = 2
    dw = tf.random.normal((nsims, nobs), seed=SEED)
    v = initialize_variables()
    S0 = v["S0"]
    t = v["time_to_expiry"]
    dt = t / dw.shape[1]
    sigma = v["implied_vol"]
    r = v["riskfree"]
    paths = brownian(S0, dt, sigma, r, dw)

    obtained = np.mean(paths[:, -1])
    expected = np.exp(r * t) * kw["S0"]

    tol = 0.01
    assert abs(obtained - expected) < tol, "%.3f , %.3f , %.3f" % (obtained, expected, obtained-expected)
