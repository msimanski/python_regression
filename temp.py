# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# I wanted to use my own data, but it was going to be such a pain to mess with getting 
# the spreadsheet into memory. I might try later if I have the time.
# I also found where Colton stole his code. https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

iris = datasets.load_iris()

# Initially I was splitting the dataset in half evenly, but it was not sorted so I got a negative variance score.
# I shufled the data by taking evens and odds.

predictorsX = iris.data
predictorsX_train = predictorsX[::2] # take every even item
predictorsX_test = predictorsX[1::2] # take every odd item

targetY = iris.target
targetY_train = targetY[::2]
targetY_test = targetY[1::2]

regression = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
regression.fit(predictorsX_train, targetY_train) # building the regression model here

irisYPrediction = regression.predict(predictorsX_test)

print("Iris data regression analysis:")
print("Coefficients: ")
print(regression.coef_)

print("Mean squared error: ") 
print(mean_squared_error(targetY_test, irisYPrediction)) # print the mean square error

print("Varience score: ")
print(r2_score(targetY_test, irisYPrediction)) # print variance score

# now make a graph, or not it wont work now wtf
# targetY_test = targetY_test[:, 1:2]
# plt.scatter(predictorsX_test, targetY_test, color = "red")
# plt.plot(predictorsX_test, irisYPrediction, color = "black", linewidth = 3)
# plt.xticks(())
# plt.yticks(())
# plt.show()