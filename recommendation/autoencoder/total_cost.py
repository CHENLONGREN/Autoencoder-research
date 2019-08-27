import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np

data = pd.read_csv("D:/PyCharm 2017.3.2/guaduate research/traning/total_cost.csv", index_col=0)

plt.figure(figsize=(9, 6))

plt.plot(data['per'], label='per', marker="o")
plt.plot(data['user_per'], label='user_per', marker="o")
plt.plot(data['item_per'], label='item_per', marker="o")
# x = [1, 2, 3, 10, 20, 50, 100, 250, 500, 750, 1000, 1300, 1700, 2000, 2500, 3000, 3500, 4000]
plt.xticks(np.arange(18), ('1', '2', '3', '10', '20', '50', '100', '250', '500', '750', '1000', '1300', '1700', '2000', '2500', '3000', '3500', '4000'))
x_major_locator = MultipleLocator(10)
plt.legend()

plt.title('total_cost')
plt.xlabel('hidden_layer')
plt.ylabel('cost')
# plt.savefig('./total_cost.png')
plt.show()