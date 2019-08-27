import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# Xaiver初始化权重的值
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


# 定义自编码器类
class AdditiveGaussianAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=1.0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initalize_weights()
        self.weights = network_weights

        # 定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])  # 输入层
        self.noisex = self.x + scale * tf.random_normal((n_input,))  # 加入噪声的输入图像
        self.hidden = self.transfer(tf.add(tf.matmul(self.noisex, self.weights['w1']), self.weights['b1']))# 隐含层，给输入的数据加入了噪声
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']) # 输出层

        # 定义自编码器的损失函数，使用平方误差(SquaredError)作为cost
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        # 定义训练操作作为优化器self.optimizer对损失self.cost进行优化，默认是Adam
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 编写成员函数，初始化自编码器全部模型参数
    def _initalize_weights(self):
            all_weights = dict()
            all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
            all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
            all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))  # 输出神经元数和输入一样
            all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
            return all_weights

    # 定义计算损失cost以及执行一步训练的函数
    def partial_fit(self, X):
            cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
            return cost

    def calc_total_cost(self, X):
            return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 定义transform函数，返回隐含层的结果
    def transform(self,X):
            return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

    # 定义generate()函数
    def generate(self, hidden=None):
            if hidden:
                hidden = np.random.normal(size=self.weights["b1"])
            return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 返回输出层的结果
    def reconstrdata(self, X):
            return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    # 获取隐含层的权重w1
    def getWeights(self):
            return self.sess.run(self.weights["w1"])

    # 获取隐含层的偏置系数b1
    def getBiases(self):
            return self.sess.run(self.weights['b1'])

    #可视化对比原输入图像和加入噪声后的图像
    # def plot_noiseimg(self, img, show_comp=True):
    #     self.noise = self.sess.run(tf.random_normal((self.n_input,)))
    #     noiseimg = img + self.training_scale * self.noise
    #     plot_image(noiseimg)
    #     if show_comp:
    #         plt.subplot(121)
    #         plt.imshow(img.reshape((28, 28)), interpolation='nearest', cmap='binary')
    #         plt.subplot(122)
    #         plot_image(noiseimg)


# 定义一个对训练、测试数据进行标准化处理（0均值，且标准差为1的分布）的函数
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)  # 保证训练、测试数据都使用完全相同Scalar
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


# batch处理
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: start_index + batch_size]


# 利用TensorFlow提供的读取示例数据的函数载入MNIST数据集
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# n_samples = mnist.train.num_examples
# training_epoch = 20
# batch_size = 128
#
# autoencoder = AdditiveGaussianAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
#
# for epoch in range(training_epoch):
#     avg_cost = 0
#     step = 1
#     while step * batch_size < n_samples:
#         batch_x = get_random_block_from_data(X_train, batch_size)
#         cost = autoencoder.partial_fit(batch_x)
#         avg_cost += cost / batch_size
#         step += 1
#     print('Epoch: %d' % epoch, 'cost: %.3f' % avg_cost)
#
# print('Total cost: %0.3f' % autoencoder.calc_total_cost(X_test))
#
# batch_test = get_random_block_from_data(X_test, 5)
# encoder_test = autoencoder.reconstrdata(batch_test)
# print(type(encoder_test))
#
# fig, ax = plt.subplots(nrows=2, ncols=5)
# for i in range(5):
#     ax[0][i].imshow(batch_test[i].reshape((28, 28)), cmap='Greys', interpolation='nearest')
#     ax[1][i].imshow(encoder_test[i].reshape((28, 28)), cmap='Greys', interpolation='nearest')
# plt.tight_layout()
# plt.show()












