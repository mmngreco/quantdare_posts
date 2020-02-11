# Have you tried calculate derivatives using TF

We will lear how to implement a simple funtion using TensorFlow 2 and how
to obtain the derivatives from it. We will implement a Black-Scholes model
for princing a call option and then we are going to obtain the greeks.

[Matthias Groncki](https://ipythonquant.wordpress.com/2018/05/22/tensorflow-meets-quantitative-finance-pricing-exotic-options-with-monte-carlo-simulations-in-tensorflow/) wrote a very
interesting post about how to obtain
[the greeks](https://en.wikipedia.org/wiki/Greeks_(finance))
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

In previous versions of TensorFlow we need to use [`tf.gradient`]() which requires
create a session and plenty of annoing stuff. Now, all this process is done 
using [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape)
which is simplier. We make get it done writting something like :

```python
with tf.GradientTape() as g1:
	npv = pricer_blackScholes(**variables)
dv = g1.gradient(npv, variables)  # first order derivatives
```

ok, but what if we want higher order derivatives? The answer is easy, we only
have to add a new `tf.GradientTape`:

```python
with tf.GradientTape() as g2:
	with tf.GradientTape() as g1:
		npv = pricer_blackScholes(**variables)
	dv = g1.gradient(npv, variables)
d2v = g2.gradient(dv, variables)
```

### Black-Scholes method

We use the well-known
[Black-Scholes](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
 to estimate the price of the call. Our code can be written as follows:

```python
@tf.function
def pricer_blackScholes(S0, strike, time_to_expiry, implied_vol, riskfree):
    """pricer_blackScholes.

    Parameters
    ----------
    S0 : tensorflow.Variable
        Underlying spot price.
    strike : tensorflow.Variable
        Strike price.
    time_to_expiry : tensorflow.Variable
        Time to expiry.
    implied_vol : tensorflow.Variable
        Volatility.
    riskfree : tensorflow.Variable
        Risk free rate.

    Returns
    -------
    npv : tensorflow.Tensor
        Net present value.

    Examples
    --------
    >>> kw = initialize_variables(to_tf=True)
    >>> pricer_blackScholes(**kw)
    <tf.Tensor: id=120, shape=(), dtype=float32, numpy=9.739834>

    Notes
    -----
    Formula: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula
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

To get the net present value (NPV) and the greeks (derivatives) we can write a
function that wraps all process. It's optional of course but very useful.

```python
@tf.function
def pricer_blackScholes(S0, strike, time_to_expiry, implied_vol, riskfree):
    """Calculates NPV and greeks using Black-Scholes model.

    Parameters
    ----------
    S0 : tensorflow.Variable
        Underlying spot price.
    strike : tensorflow.Variable
        Strike price.
    time_to_expiry : tensorflow.Variable
        Time to expiry.
    implied_vol : tensorflow.Variable
        Volatility.
    riskfree : tensorflow.Variable
        Risk free rate.

    Returns
    -------
    npv : tensorflow.Tensor
        Net present value.

    Examples
    --------
    >>> kw = initialize_variables(to_tf=True)
    >>> pricer_blackScholes(**kw)
    <tf.Tensor: id=120, shape=(), dtype=float32, numpy=9.739834>

    Notes
    -----
    Formula: https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black%E2%80%93Scholes_formula
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

* `npv` : the net present value is 9.74.
* `S0` = $\frac{\partial v}{\partial S}$
* `implied_vol` = $\frac{\partial v}{\partial \sigma}$
* `strike` = $\frac{\partial v}{\partial K}$
* `time_to_expiry` = $\frac{\partial v}{\partial \tau}$

We have seen how to implement a TensorFlow function and how to get the
derivatives from it. Now, we are going to see another example using the
Monte Carlo method.

### Monte Carlo method

The Monte Carlo method is very useful when we don't have the close formula or
it's very complex. We are going to implement the Monte Carlo pricing function,
to this task I decided implement `brownian` function too used inside of
`pricer_montecarlo`.

```python
@tf.function
def pricer_montecarlo(S0, strike, time_to_expiry, implied_vol, riskfree, dw):
    """Monte Carlo pricing method.

    Parameters
    ----------
    S0 : tensorflow.Variable
        Underlying spot price.
    strike : tensorflow.Variable
        Strike price.
    time_to_expiry : tensorflow.Variable
        Time to expiry.
    implied_vol : tensorflow.Variable
        Volatility.
    riskfree : tensorflow.Variable
        Risk free rate.
    dw : tensorflow.Variable
        Normal random variable.

    Returns
    -------
    npv : tensorflow.Variable
        Net present value.

    Examples
    --------
    >>> nsims = 10
    >>> nobs = 100
    >>> dw = tf.random.normal((nsims, nobs), seed=3232)
    >>> v = initialize_variables(to_tf=True)
    >>> npv = pricer_montecarlo(**v, dw=dw)
    >>> npv
    <tf.Tensor: id=646, shape=(), dtype=float32, numpy=28.780073>
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


@tf.function
def brownian(S0, dt, sigma, mu, dw):
    """Generates a brownian motion.

    Parameters
    ----------
    S0 : tensorflow.Variable
        Initial value of Spot.
    dt : tensorflow.Variable
        Time step.
    sigma : tensorflow.Variable
        Volatility.
    mu : tensorflow.Variable
        Mean, in black Scholes frame it's the risk free rate.
    dw : tensorflow.Variable
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
    """
    dt_sqrt = tf.math.sqrt(dt)
    shock = sigma * dt_sqrt * dw
    drift = (mu - (sigma ** 2) / 2)
    bm = tf.math.exp(drift * dt + shock)
    out = S0 * tf.math.cumprod(bm, axis=1)
    return out
```

Now, we are ready to calculate the NPV and the greeks under this frame.

```python
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
    >>> out = calculate_montecarlo()
    >>> pprint(out)
    {'dv': {'S0': 0.5065364,
            'implied_vol': 56.45906,
            'riskfree': 81.81441,
            'strike': -0.37188327,
            'time_to_expiry': 4.050169},
     'npv': 9.746445}
    """
    nsims = 10000000
    nobs = 2
    dw = tf.random.normal((nsims, nobs), seed=SEED)
    v = initialize_variables(to_tf=True)

    out = dict()

    with tf.GradientTape() as g1:
        npv = pricer_montecarlo(**v, dw=dw).numpy()
    dv = g1.gradient(npv, v)

    out["dv"] = {k: v.numpy() for k, v in dv.items()}
    return out
```

The output:

```python
>>> out = calculate_montecarlo()
>>> pprint(out)
{'dv': {'S0': 0.5065364,
        'implied_vol': 56.45906,
        'riskfree': 81.81441,
        'strike': -0.37188327,
        'time_to_expiry': 4.050169},
 'npv': 9.746445}
```


### Comparison

We are taking a look at the results of both methods:

| Variable                             | Black-Scholes | Montecarlo  |
| :----------------------------------: | :-----------: | :---------: |
| npv                                  | 9.746445      | 9.739834    |
| $\frac{\partial v}{\partial S}$      | 0.5065364     | 0.5066145   |
| $\frac{\partial v}{\partial \sigma}$ | 56.45906      | 56.411205   |
| $\frac{\partial v}{\partial r}$      | 81.81441      | 81.843216   |
| $\frac{\partial v}{\partial K}$      | -0.37188327   | -0.37201464 |
| $\frac{\partial v}{\partial \tau}$   | 4.050169      | 4.048208    |


As we can see, we can get similar results with both methods. There is room for
improvements, for example: We can increase the number of simulations into
Monte Carlo method. However, the results are reasonably close between them and
very easy to implement, which is more important.

Have you used this before?  would be great if you can tell your use case in the
comments.

Keep coding!

