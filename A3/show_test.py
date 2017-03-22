import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data2D.npy")

plt.figure()
# plt.plot(r_X[0],r_X[1])
plt.scatter(data[:, 0], data[:, 1], marker='x', color='m', label='1', s=30)
plt.show()