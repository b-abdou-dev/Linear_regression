import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

### import csv to data frame
capital_df = pd.read_csv('canada_per_capita_income.csv')
print(capital_df)


### apply linear regression algorithm
ln_reg = linear_model.LinearRegression()
ln_reg.fit(capital_df[['year']], capital_df[['per capita income (US$)']])
print(ln_reg)

#### find a and b
print(ln_reg.coef_)
print(ln_reg.intercept_)

#### predict given prices
capital_predicted = ln_reg.predict(capital_df[['year']])
print(capital_predicted)
capital_df['predicted capitals'] = capital_predicted
capital_df.to_csv('canada_per_capita_income.csv')

#### plot both predicted and original data

plt.scatter(capital_df['year'], capital_df['per capita income (US$)'],color='k')
plt.plot(capital_df['year'], capital_predicted,color='g')

plt.show()