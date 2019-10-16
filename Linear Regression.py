import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
print(df)

# plt.xlabel('area(square ft)')
# plt.ylabel('price(US$')
# plt.scatter(df.area, df.price, color='red', marker ='+')
# plt.show()

#### since distribution is suitble for linear regression we choose it
ln_reg = linear_model.LinearRegression()
ln_reg.fit(df[['area']], df.price)

## finding coefficent a and and intercept b
print("a is ", ln_reg.coef_)
print("b is ", ln_reg.intercept_)



# predict price of an area of 3300
ln_reg.predict([[3300]])

print(ln_reg.predict([[3300]]))


#### predict other areas read from area.csv
areas_df = pd.read_csv('areas.csv')
print(areas_df.head(5))

#### To keep just the first column (areas coulmn)

areas_df = pd.read_csv('areas.csv')
keep_col = ['area']
new_f = areas_df[keep_col]
new_f.to_csv("areas.csv", index=False)
print(areas_df.head(5))

predicted_prices = ln_reg.predict(areas_df)
print(predicted_prices)


#### create new column to areas data frame
areas_df['prices'] = predicted_prices
print(areas_df)

#### export changes of areas data frame to areas.csv file
areas_df.to_csv('areas.csv', index=False)

plt.xlabel('area(square ft)')
plt.ylabel('price(US$')
plt.scatter(df.area, df.price, color='red', marker ='+')
plt.plot(areas_df.area, predicted_prices, color='blue')
plt.show()


