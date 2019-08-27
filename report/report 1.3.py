import numpy as np


matrix = np.empty(shape = (256, 8))

for i in range(256):
    j = 0
    k = '{:08b}'.format(i)
    t = list(k)
    for j in range(8):
        matrix[i][j] = t[j]



row_rand_array = np.arange(matrix.shape[0])
np.random.shuffle(row_rand_array)
row_rand = matrix[row_rand_array[0:100]]

def compute_distances_no_loops(A, B):
    m = np.shape(A)[0]
    n = np.shape(B)[0]
    M = np.dot(A, B.T)
    H = np.tile(np.matrix(np.square(A).sum(axis=1)).T,(1,n))
    K = np.tile(np.matrix(np.square(B).sum(axis=1)),(m,1))
    return np.sqrt(-2 * M + H + K)

distances_matrix = compute_distances_no_loops(row_rand, matrix)
v = distances_matrix.sum(axis = 0)
k = np.argmin(v)

