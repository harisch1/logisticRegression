import numpy as np
import matplotlib.pyplot as plt
from Logistic import logisticRegression

if __name__ == '__main__':
    y = np.array([1, 0, 0, 1])
    X = np.array([[-0.12, 0.3, -0.01],[0.2, -0.03, -0.35],[-0.37, 0.25, 0.07],[-0.1, 0.14, -0.52]])
    theta = np.array([-0.09, 0, -0.19, 0.21])
    model = logisticRegression()
    model.fit(X, y, theta, 0.2, 50000)
    c = model.cost_
    plt.plot([x for x in range(len(c))], c)
    plt.show()

#CHOUDHARY Muhammad Haris