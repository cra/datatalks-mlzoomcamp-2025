import pandas as pd
import numpy as np

df = pd.read_csv('laptops.csv')
masq = df['Brand'] == 'Innjoo'
features = ['RAM', 'Storage', 'Screen']
prices = [1100, 1300, 800, 900, 1000, 1100]

X = df[masq][features].to_numpy()
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array(prices)
w = XTX_inv @ X.T @ y
print(sum(w))
