import numpy as np

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s(faces):
    faces = faces.reshape(-1, 48, 48)
    faces = faces.T
    faces = np.reshape(faces, (faces.shape[0] ** 2, faces.shape[2]))
    ones = np.ones((faces.shape[1]))
    faces = np.vstack((faces, ones))
    return faces

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE(w, Xtilde, y):
    return 1 / 5000 * (y - Xtilde.T.dot(w)).T.dot(y - Xtilde.T.dot(w))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    pass

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1(Xtilde, y):
    Xtranspose = np.dot(Xtilde, Xtilde.T)
    Xid = np.eye(Xtranspose.shape[0])
    return np.dot(np.linalg.solve(Xtranspose, Xid), Xtilde.dot(y))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    pass

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    pass

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    # ...
    fMSE_ = fMSE(w1, Xtilde_tr, ytr)
    print("Method one: ", fMSE_)
