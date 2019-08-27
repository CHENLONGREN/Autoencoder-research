import numpy as np

matrix = np.loadtxt(open("C:/Users\chenl\PycharmProjects/research/traning/rating_matrix.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix.tolist()
matrix = np.array(list_matrix)
m = len(matrix)
n = len(matrix[0])

for i in range(m):
    j=0
    for j in range(n):
        if(matrix[i][j] == 0):
            matrix[i][j] = 3

np.savetxt('per_rating_matrix.csv', matrix, delimiter = ',')