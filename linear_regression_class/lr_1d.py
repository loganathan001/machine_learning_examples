import numpy as np
import matplotlib.pyplot as plt

#load data
X_list=[]
Y_list=[]
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X_list.append(float(x))
    Y_list.append(float(y))
    
X = np.array(X_list)
Y = np.array(Y_list)

plt.scatter(X,Y)
plt.show()

#apply equations

#denominator = X.dot(X) - X.mean() * X.sum()
#a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
#b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

denominator =  (X**2).mean() - X.mean()**2

a = ((X*Y).mean() - X.mean()*Y.mean())/denominator
b = (Y.mean()*(X**2).mean() - X.mean()*(X*Y).mean())/denominator

#Calculate predicted Y
Yhat = a*X + b
plt.scatter(X,Y)
plt.plot(X, Yhat)
plt.show()

# Calculate R**2
diff_res = Y - Yhat
diff_mean = Y - Y.mean()

sse_res = diff_res.dot(diff_res)
sse_tot = diff_mean.dot(diff_mean)

r_squard = 1 - (sse_res/sse_tot)

print("RQuared: %f" % r_squard)

