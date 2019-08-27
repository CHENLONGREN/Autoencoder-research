import numpy


def matrix_factorisation(R, P, Q, K, steps=100, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


matrix = numpy.loadtxt(open("C:/Users\chenl\PycharmProjects/research/traning/rating_matrix.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix.tolist()
R = numpy.array(list_matrix)
R = numpy.array(R)
m = len(R)
n = len(R[0])

for i in range(m):
    j=0
    for j in range(n):
        if(R[i][j] == 0):
            R[i][j] = numpy.random.randint(1, 5)


N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorisation(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
numpy.savetxt('nmf_rating_matrix.csv', nR, delimiter=',')
