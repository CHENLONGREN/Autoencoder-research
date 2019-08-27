import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt(open("C:/Users/chenl/PycharmProjects/research/traning/nmf_auto.csv","rb"),delimiter=",", skiprows=0)
plt.plot(data['epoch'], data['cost'])
plt.xticks(rotation=2)
plt.xlabel('epoch')
plt.ylabel('avg_cost')
plt.title('traning cost')
plt.savefig('./nmf_cost.jpg')
plt.show