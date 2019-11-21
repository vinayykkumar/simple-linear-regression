import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('sats.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
sat = LinearRegression()
sat.fit(x_train, y_train)

gpa = sat.predict(x_test)

plt.scatter(x_train, y_train, color = 'green')
plt.plot(x_train, sat.predict(x_train))
plt.show()