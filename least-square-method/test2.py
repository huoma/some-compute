import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def stdError_func(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
    return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
    y_mean = np.array(y)
    y_mean[:] = y.mean()
    return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)


filename = "least-square-method/data.csv"
df = pd.read_csv(filename)
x = np.array(df.iloc[:, 0:2].values)

y = np.array(df.iloc[:, 4].values)

print(x)
print("-"*140)
print(y)

poly_reg = PolynomialFeatures(degree=2)  # 三次多项式
X_ploy = poly_reg.fit_transform(x)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_ploy, y)
predict_y = lin_reg_2.predict(X_ploy)
strError = stdError_func(predict_y, y)
R2_1 = R2_1_func(predict_y, y)
R2_2 = R2_2_func(predict_y, y)
score = lin_reg_2.score(X_ploy, y)  # sklearn中自带的模型评估，与R2_1逻辑相同

print("coefficients", lin_reg_2.coef_)
print("intercept", lin_reg_2.intercept_)
print('degree={}: strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    3, strError, R2_1, R2_2, score))
