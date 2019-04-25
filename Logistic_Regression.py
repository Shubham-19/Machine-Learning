import pandas as pd
import math as m
import matplotlib.pyplot as plt
data = pd.read_csv('dbm.csv')
x = data['spend'].values
y = data['sex'].values
plt.scatter(x, y)
plt.show()
XP = []
for i in x:
	XP.append(m.exp(i)/(1 + m.exp(i)))
print(XP)

YP = []
for value in XP:
	if (value < 0.6):
		YP.append('F')
	else:
		YP.append('M')

print(len(XP), len(YP))
plt.scatter(XP, YP)
plt.show()
