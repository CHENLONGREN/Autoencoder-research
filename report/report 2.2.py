import numpy as np
import report3

matrix = np.loadtxt(open("D:/1.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix.tolist()
matrix = np.array(list_matrix)

matrix16 = [[0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1]]
matrix16 = np.array(matrix16)
distances_matrix = report3.compute_distances_no_loops(matrix, matrix16)
distances_matrix = np.array(distances_matrix)

temp = np.zeros(shape=(50))
sum = np.zeros(shape=(120))
t = 0
for i in range(15):
    for j in range(i + 1, 16):
        for k in range(50):
            if(distances_matrix[k][i] >= distances_matrix[k][j]):
                temp[k] = distances_matrix[k][j]
            else:
                temp[k] = distances_matrix[k][i]
        sum[t] = np.sum(temp)
        t += 1

pp = np.argmin(sum)
print(sum[pp])













