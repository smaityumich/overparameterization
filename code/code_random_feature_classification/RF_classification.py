"""
To reproduce Figure 3 in the Summary section, run experiment(...) with
    - theta in [0, 0.25, 0.5, 0.75, 1]
    - psi in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    - seed in [0, 1, ..., 19]
Aggregate/average results given by 20 replicates
"""

import numpy as np
from sklearn import svm
from sklearn import metrics

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.max([0, x])

def experiment(theta, psi, seed_for_produce):
    n = 400
    d = 200
    N = int(d * psi)

    beta_0 = [10] + [0 for _ in range(n-1)]
    beta_1 = [10*np.cos(np.pi*theta), 10*np.sin(np.pi*theta)] + [0 for _ in range(n-2)]

    np.random.seed(seed=seed_for_produce)
    W = np.random.normal(0, 1, (d, N)).tolist()

    X_train = [np.random.normal(0, 1, d).tolist() for _ in range(n)]
    y_train = [(2 * np.random.binomial(1, sigmoid(beta_0[0] * X_train[i][0]), 1) - 1).item() for i in range(360)] + [(2 * np.random.binomial(1, sigmoid(beta_1[0] * X_train[i][0] + beta_1[1] * X_train[i][1]), 1) - 1).item() for i in range(360, 400)]
    X_train = np.matmul(X_train, W).tolist()
    X_train = [[relu(X_train[i][j]) for j in range(len(X_train[0]))] for i in range(len(X_train))]

    X_test_0 = [np.random.normal(0, 1, d).tolist() for _ in range(n*10)]
    y_test_0 = [(2 * np.random.binomial(1, sigmoid(beta_0[0] * X_test_0[i][0]), 1) - 1).item() for i in range(n*10)] 
    X_test_0 = np.matmul(X_test_0, W).tolist()
    X_test_0 = [[relu(X_test_0[i][j]) for j in range(len(X_test_0[0]))] for i in range(len(X_test_0))]

    X_test_1 = [np.random.normal(0, 1, d).tolist() for _ in range(n*10)]
    y_test_1 = [(2 * np.random.binomial(1, sigmoid(beta_1[0] * X_test_1[i][0] + beta_1[1] * X_test_1[i][1]), 1) - 1).item() for i in range(n*10)]
    X_test_1 = np.matmul(X_test_1, W).tolist()
    X_test_1 = [[relu(X_test_1[i][j]) for j in range(len(X_test_1[0]))] for i in range(len(X_test_1))]

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred_0 = clf.predict(X_test_0)
    y_pred_1 = clf.predict(X_test_1)
    
    print(theta)
    print(psi)
    print(1 - metrics.accuracy_score(y_test_0, y_pred_0))
    print(1 - metrics.accuracy_score(y_test_1, y_pred_1))

