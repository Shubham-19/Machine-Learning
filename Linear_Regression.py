import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('File_1.csv')
print(data)
x = data['x'].values
y = data['y'].values
print('x: {}'.format(x))
plt.scatter(x, y)
plt.show()

x_mean = sum(x)/len(x)
y_mean = sum(y)/len(y)

Nr, Dr = 0, 0
sumNr, sumDr = 0, 0
for i in range(len(x)):
	Nr = (x[i]-x_mean)*(y[i] - y_mean)
	sumNr += Nr
	Dr = (x[i] - x_mean)**2
	sumDr += Dr

slope = sumNr/sumDr
print('slope: {}'.format(slope))

c = y_mean - (slope*x_mean)
print('intercept: {}'.format(c))

YP = []
for i in x:
	Yp = (slope*i) + c
	YP.append(Yp)

print(YP)
plt.plot(x, y, color = 'red')
plt.scatter(x, YP, color = 'green')
plt.plot(x, YP, '--b')
plt.show()

YP_mean = sum(YP)/len(YP)
print(YP_mean)
sumErNr = 0
err = 0

sum_3, sum_4 = 0, 0
for i in range(len(YP)):
	sum_3 += (YP[i] - YP_mean)**2
	sum_4 += (y[i] - y_mean)**2

err = sum_3/sum_4
print(err)
