import numpy as np

matrix = np.loadtxt(open("C:/Users\chenl\PycharmProjects/research/traning/rating_matrix.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix.tolist()
matrix = np.array(list_matrix)
m = len(matrix)
n = len(matrix[0])

for i in range(m):
    j = 0
    count = 0
    sum = 0
    per = 0
    for j in range(n):
        if(matrix[i][j] != 0):
            sum += matrix[i][j]
            count += 1
    j=0
    per = sum/count
    for j in range(n):
        if (matrix[i][j] == 0):
            matrix[i][j] = per

np.savetxt('user_per_rating_matrix.csv', matrix, delimiter = ',')
