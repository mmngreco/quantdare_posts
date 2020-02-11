import sympy as sp
from scipy import stats
import numpy as np
import tensorflow as tf

W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def blackScholes(S0, strike, time_to_expiry, implied_vol, riskfree_rate):

    S = S0
    K = strike
    dt = time_to_expiry
    dt_sqrt = np.sqrt(dt)
    sigma = implied_vol
    r = riskfree_rate
    Phi = stats.norm.cdf
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * dt) / (sigma * dt_sqrt)
    d_2 = d_1 - sigma * dt_sqrt
    out =  S * Phi(d_1) - K * np.exp(-r * dt) * Phi(d_2)

    return out


def blackScholes_py(S0, strike, time_to_expiry, implied_vol, riskfree_rate):

    S = S0
    K = strike
    dt = time_to_expiry
    dt_sqrt = np.sqrt(dt)
    sigma = implied_vol
    r = riskfree_rate
    Phi = stats.norm.cdf
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * dt) / (sigma * dt_sqrt)
    d_2 = d_1 - sigma * dt_sqrt
    out =  S * Phi(d_1) - K * np.exp(-r * dt) * Phi(d_2)

    return out

"""
Performance Profiling
%timeit blackScholes_py(100., 110., 4., 0.2, 0.03)
%timeit blackScholes(100., 110., 4., 0.2, 0.03)
"""

def blackScholes_tf_pricer(enable_greeks=True):
    # Build the static computational graph

    S = tf.Variable(dtype=tf.float32)
    K = tf.Variable(dtype=tf.float32)
    dt = tf.Variable(dtype=tf.float32)
    sigma = tf.Variable(dtype=tf.float32)
    r = tf.Variable(dtype=tf.float32)

    Phi = tf.distributions.Normal(0., 1.).cdf

    d_1 = (tf.log(S / K) + (r + sigma ** 2 / 2) * dt) / (sigma * tf.sqrt(dt))
    d_2 = d_1 - sigma * tf.sqrt(dt)

    npv =  S * Phi(d_1) - K * tf.exp(-r * dt) * Phi(d_2)
    target_calc = [npv]

    if enable_greeks:
        greeks = tf.gradients(npv, [S, sigma, r, K, dt])
        dS_2ndOrder = tf.gradients(greeks[0], [S, sigma, r, K, dt])
        # Calculate mixed 2nd order greeks for S (esp. gamma, vanna) and sigma (esp. volga)
        dsigma_2ndOrder = tf.gradients(greeks[1], [S, sigma, r, K, dt])
        dr_2ndOrder = tf.gradients(greeks[2], [S, sigma, r, K, dt])
        dK_2ndOrder = tf.gradients(greeks[3], [S, sigma, r, K, dt])
        dT_2ndOrder = tf.gradients(greeks[4], [S, sigma, r, K, dt])

        target_calc += [
                greeks,
                dS_2ndOrder,
                dsigma_2ndOrder,
                dr_2ndOrder,
                dK_2ndOrder,
                dT_2ndOrder
            ]

    # Function to feed in the input and calculate the computational graph
    def execute_graph(S_0, strike, time_to_expiry, implied_vol, riskfree_rate):
        with tf.Session() as sess:
            res = sess.run(target_calc,
                           {
                               S: S_0,
                               K : strike,
                               r : riskfree_rate,
                               sigma: implied_vol,
                               dt: time_to_expiry})
        return res
    return execute_graph

