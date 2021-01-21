import numpy as np
import math

def gen_covariance_matrix(variance_vector):
    q = np.eye(variance_vector.shape[0])
    for i, variance in enumerate(variance_vector):
        q[i][i] = variance;
    return q

