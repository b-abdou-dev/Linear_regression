### linear regression multivariables

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
print(df)

### cleaning data

bedr_median = df.bedrooms.median()
print(bedr_median)
df['bedrooms'].fillna(bedr_median, inplace=True) ## inplace enable to change the same data frame
print(df)

### finding regreassion line

lreg = linear_model.LinearRegression()
lreg.fit(df[['area', 'bedrooms', 'age']], df['price'])
print(lreg)

### finding coefficinets
print(lreg.coef_)

### finding intercept
print(lreg.intercept_)

### predict prices of the original price
predicted_prices = lreg.predict(df[['area', 'bedrooms', 'age']])
print(predicted_prices)

### export predicted prices to csv file
df['predicted prices'] = predicted_prices
df.to_csv('homeprices.csv')

### plot both original and predicted prices versus area

plt.scatter(df['area'], df['price'], color='blue')
plt.plot(df['area'], df['predicted prices'], color='red')
plt.show()



