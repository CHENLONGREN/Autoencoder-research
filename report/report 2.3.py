import numpy as np


matrix = np.loadtxt(open("D:/1.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix.tolist()
matrix = np.array(list_matrix)
g = np.zeros(shape=(50, 50))

p = 4
for i in range(50):
    for j in range(50):
        z = matrix[i] - matrix[j]
        g[i][j] = (p - np.dot(z.T, z))/p

for i in range(50):
    count = 0
    t = list(matrix[i])
    for j in range(50):
        k = list(matrix[j])
        if(t == k):
            count += 1
    p = count/50
    g[i][i] = p

temp1 = np.zeros(shape=(50))
for i in range(50):
    for j in range(50):
        temp1[i] += g[i][j]

# print(np.argmax(temp1))

temp2 = np.zeros(shape=(50))
sum = np.zeros(shape=(49))
t = 0
for i in range(50):
    if(i == 17):
        continue
    for j in range(50):
        if (g[j][17] >= g[j][i]):
            temp2[j] = np.sqrt(np.sum(np.square(matrix[17] - matrix[j])))
        else:
            temp2[j] = np.sqrt(np.sum(np.square(matrix[i] - matrix[j])))
    sum[t] = np.sum(temp2)
    t += 1

pp = np.argmin(sum)
print(sum[pp])

















