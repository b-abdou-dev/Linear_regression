import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

### importing csv file
df = pd.read_csv('hiring.csv')
print(df)
### cleaning the dataframe
df['experience'].fillna(0, inplace=True)
median = df['test_score(out of 10)'].median()
df['test_score(out of 10)'].fillna(median, inplace=True)
print(df)

### converting number words to numbers
print()
num_list = []
for i in df['experience'].iloc[2:]:
    i = w2n.word_to_num(i)
    num_list.append(i)
df.loc[2:,('experience')] = num_list
print(df['experience'].iloc[2:])
print(df)

#### creating linear regression model
lreg = linear_model.LinearRegression()
lreg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])
print(lreg)

### printing the coefficients

print(lreg.coef_)
print(lreg.intercept_)

### predicting values

print(lreg.predict([[2, 9, 6]]))
print(lreg.predict([[12, 10, 10]]))