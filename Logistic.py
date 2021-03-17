import numpy as np
import math
import matplotlib.pyplot as plt

class logisticRegression:
    def fit(self, X, y, w, t, n_iteration):
        m = X.shape[0]
        self.w = w[1:]
        self.bias = w[0]
        self.cost_ = []
        self.w_ = []
        for _ in range(n_iteration):
            yp = self.predict(X)
            hx = yp - y
            gb = np.sum(hx)
            gw = np.dot(hx, X)
            self.bias = self.bias - t * (1 / m) * (gb)
            self.w = self.w - t * (1 / m) * (gw)
            cost = self.costFunction(X, y)
            self.cost_.append(cost)
            self.w_.append(np.concatenate(([self.bias], self.w)))
    def predict(self, X):
        tht_t_x = np.dot(self.w, X.T) + self.bias
        return np.array([self.sigmoid(x) for x in tht_t_x])
    def sigmoid(self, x):
        return 1/(1+math.exp(-x))
    def costFunction(self, X, y):
        yp = self.predict(X)
        m = X.shape[0]
        l = np.sum(y*np.log(yp))
        r = np.sum((1-y)*np.log(1-yp))
        return -1 / m * (l + r)

#CHOUDHARY Muhammad Haris