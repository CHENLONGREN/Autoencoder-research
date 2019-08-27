import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/chenl/PycharmProjects/research/traning/per_auto.csv")
plt.plot(data['epoch'], data['cost'])
plt.xticks(rotation=2)
plt.xlabel('epoch')
plt.ylabel('avg_cost')
plt.title('traning cost')
plt.savefig('./per_cost.png')
plt.show