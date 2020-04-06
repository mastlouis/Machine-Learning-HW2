import numpy as np

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    newdata = faces.flatten('F')
    newdata = np.reshape(newdata, (faces.shape[1] ** 2, faces.shape[0]))
    ones = np.ones((1,newdata.shape[1]))
    newdata = np.append(newdata, ones, axis=0)
    # newdata = np.vstack([newdata, ones])
    return newdata
    pass

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    pass

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    temp = np.dot(Xtilde.T, w)
    temp -= y
    temp = Xtilde.dot(temp)
    temp /= Xtilde.shape[0]
    # temp2 = w.dot(w)(2*X.shape[0])
    return temp
    # return Xtilde.dot(np.dot(Xtilde.T, w) - y)/Xtilde.shape[0]

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    pass

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    weights = gradientDescent(Xtilde, y)
    print("Weights: " + str(weights))
    return weights

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    pass

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    # EPSILON = 3e-3  # Step size aka learning rate
    EPSILON = 3e-4
    T = 5000  # Number of gradient descent iterations
    weights = np.random.randn(Xtilde.shape[0]) * 0.01;  # Standard deviation of 0.01
    for i in range(T):
        weights = weights - (EPSILON * gradfMSE(weights, Xtilde, y, alpha))
    return weights


if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    # w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    # ...
