import numpy as np

matrix = np.loadtxt(open("C:/Users\chenl\PycharmProjects/research/traning/rating_matrix.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix.tolist()
matrix = np.array(list_matrix)
m = len(matrix)
n = len(matrix[0])

for i in range(n):
    j = 0
    count = 0
    sum = 0
    per = 0
    for j in range(m):
        if(matrix[j][i] != 0):
            sum += matrix[j][i]
            count += 1
    j=0
    if(count == 0):
        per = 0
    else:
        per = sum / count
    for j in range(m):
        if (matrix[j][i] == 0):
            matrix[j][i] = per

np.savetxt('item_per_rating_matrix.csv', matrix, delimiter = ',')