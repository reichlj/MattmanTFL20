import numpy as np

prices= [p for p in range(1,101)]
amounts= [0.2*p for p in range(1,101)]
revenue = 0
for price, amount in zip(prices, amounts):
     revenue += price * amount
print('Revenue',revenue)
revenue = np.dot(prices, amounts)
print('Revenue Numpy',revenue)
