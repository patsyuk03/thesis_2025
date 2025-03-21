import numpy as np
import os


xi_mean = np.zeros(6)
xi_cov = 5*np.identity(6)

print(xi_mean)

print(xi_cov)

xi_mean_cov = np.vstack((xi_mean, xi_cov))

print(xi_mean_cov)

print(xi_mean_cov[0])
print(xi_mean_cov[1:])