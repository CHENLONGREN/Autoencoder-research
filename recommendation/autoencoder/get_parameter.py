import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep
import tensorflow as tf
from autoencoder import AdditiveGaussianAutoencoder


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]


matrix0 = np.loadtxt(open("C:/Users/chenl/PycharmProjects/research/traning/user_per_rating_matrix.csv","rb"),delimiter=",",skiprows=0)
list_matrix = matrix0.tolist()
matrix = np.array(list_matrix)

X_train, X_test = train_test_split(matrix, test_size=0.2, random_state=0)
n_samples = X_train.shape[0]
training_epoch = 100
batch_size = 256

autoencoder = AdditiveGaussianAutoencoder(n_input=3883, n_hidden=10, transfer_function=tf.nn.sigmoid,
                                          optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

for epoch in range(training_epoch):
    avg_cost = 0
    step = 1
    while step * batch_size < n_samples:
        batch_x = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_x)
        avg_cost += cost / batch_size
        step += 1

# batch_test = get_random_block_from_data(X_test, 256)
# encoder_test = autoencoder.transform(batch_test)
# decoder_test = autoencoder.reconstrdata(batch_test)
# np.savetxt('0.csv', batch_test, delimiter = ',')
# np.savetxt('1.csv', encoder_test, delimiter = ',')
# np.savetxt('2.csv', decoder_test, delimiter = ',')
w1 = autoencoder.getWeights1()
b1 = autoencoder.getBiases1()
w2 = autoencoder.getWeights2()
b2 = autoencoder.getBiases2()
# print(w.shape)
# print(b.shape)
np.savetxt('w1.csv', w1, delimiter = ',')
np.savetxt('b1.csv', b1, delimiter = ',')
np.savetxt('w2.csv', w2, delimiter = ',')
np.savetxt('b2.csv', b2, delimiter = ',')









