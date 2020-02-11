"""
pip install git+https://github.com/slaypni/fastdtw
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance
from fastdtw import fastdtw

x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
y = np.array([[2,2], [3,3], [4,4]])

distance, path = fastdtw(x, y, dist=euclidean)
pd.DataFrame(np.array(path)).plot()

print(distance)
print(path)

x = np.arange(10)
y = np.arange(10)

x = np.random.randn(10000)
y = np.random.randn(10000)

distance, path = fastdtw(x, y, dist=distance.euclidean)
distance, path = fastdtw(x, y, dist=distance.chebyshev)

path_arr = np.array(path)

pd.DataFrame([x, y, x[path_arr[:, 0]], y[path_arr[:, 1]]]).T.plot()

pd.DataFrame(path_arr)
pd.DataFrame([x,y]).T.plot()

