# Have you tried calculate derivatives using TF

We will lear how to implement a simple funtion using TensorFlow 2 and how
to obtain the derivatives from it. We will implement a Black-Scholes model
for princing a call option and then we are going to obtain the greeks.

[Matthias Groncki](https://ipythonquant.wordpress.com/2018/05/22/tensorflow-meets-quantitative-finance-pricing-exotic-options-with-monte-carlo-simulations-in-tensorflow/) wrote a very interesting post
about how to obtain [the greeks](https://en.wikipedia.org/wiki/Greeks_(finance)
of a princing option using tensorflow which inspired me to write this post. So,
I took the same example and make some updates to use TensorFlow 2.

### Requirements

* [Python](https://www.python.org/)
* [TesorFlow](https://www.tensorflow.org/api_docs/python/tf)
* [Black-Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
* [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing)

## Black-Scholes pricing formula

We are going to implement the Black-Scholes formula for pricing options. In
this example we focus on the call option.

The version 2 of TensorFlow has many enhacements, specially on the python
API which is easier to write code than before.

```python
@tf.function
def pricer_blackScholes(S0, strike, time_to_expiry, implied_vol, riskfree):
    """Prices call option.

    Parameters
    ----------
    S0 : float
        Spot price.
    strike : float
        Strike price.
    time_to_expiry : float
        Time to maturity.
    implied_vol : float
        Volatility.
    riskfree : float
        Risk free rate.

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
```

As we can see the above code is the implementation of a call option in terms of
the Black-Scholes framework. A very cool improvement is the
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)
decorator which create a callable graph for us.

## Calculating Derivatives

In previous versions of TensorFlow we need to use `tf.gradient` which requires
create a session and plenty of annoing stuff. Now, we make the same writting
something like this:

```python
with tf.GradientTape() as g1:
	npv = pricer_blackScholes(**variables)
dv = g1.gradient(npv, variables)  # first order derivatives
```

and if we want higher order derivatives we can get it done adding a
new `tf.GradientTape`:

```python
with tf.GradientTape() as g2:
	with tf.GradientTape() as g1:
		npv = pricer_blackScholes(**variables)
	dv = g1.gradient(npv, variables)
d2v = g2.gradient(dv, variables)
```

Using that, our code can be written as follows:

```python
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

    return dict(npv=npv, dv=dv)
```

The previous function returns:

```python
>>> calculate_blackScholes()
{'dv': {'S0': 0.5066145,
        'implied_vol': 56.411205,
        'riskfree': 81.843216,
        'strike': -0.37201464,
        'time_to_expiry': 4.048208},
 'npv': 9.739834}
```

Where:
the net present value is 9.74.
$\frac{}{}$

