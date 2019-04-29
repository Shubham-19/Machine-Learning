import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
import numpy as np
from re import sub
from decimal import Decimal


data = pd.read_csv('listings/Listings_1.csv')
df = data.head(1000)

X = np.asarray(df[['beds', 'review_scores_rating']])
X = X.astype(np.float64, copy = False)
X[np.isnan(X)] = np.median(X[~np.isnan(X)])

Y = np.asarray([float(Decimal(sub(r'[^\d.]', '', x))) for x in df['price']])
x_train, x_test, y_train, y_test = tts(X, Y, test_size = 0.2, random_state = 88)


reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
print('coefficients: ', reg.coef_)
predicted_values = reg.predict(x_test)
print(r2_score(y_test, predicted_values))
# plt.scatter(x_test[:,0], y_test)
plt.ylim(0, 500)
plt.xlim(-1, 100)
# plt.plot(x_test[:,0], predicted_values, color = 'red')
# plt.show()
plt.plot(y_test)
plt.plot(predicted_values)
plt.show()
