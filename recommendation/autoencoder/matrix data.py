import numpy as np
import pandas as pd

data_ratings = pd.read_table("C://data/ml-1m/ratings.dat", sep = '::', names = ['user_id', 'movie_id', 'rating', 'time'], engine = 'python')
data_users = pd.read_table("C://data/ml-1m/users.dat", sep = '::', names = ['user_id', 'gender', 'age', 'occupation', 'zip'], engine = 'python')
data_movies = pd.read_table("C://data/ml-1m/movies.dat", sep = '::', names = ['movie_id', 'title', 'genres'], engine = 'python')
list_ratings = data_ratings.values.tolist()
list_users = data_users.values.tolist()
list_movies = data_movies.values.tolist()
m = len(list_users)
n = len(list_movies)
l = len(list_ratings)
ratings = np.zeros(shape=(m, n))

# i = 0
# j = 0
t = 0
for i in range(m):
    j = 0
    for j in range(n):
        k = t
        while(list_ratings[k][0] == i+1):
            if(list_ratings[k][1] == j+1):
                ratings[i][j] = list_ratings[k][2]
            k+=1
            if(k >= l):
                break
        # j+=1
    while(list_ratings[t][0] == i+1):
        t+=1
        if (t >= l):
            break
    # i+=1

np.savetxt('rating_matrix.csv', ratings, delimiter = ',')





