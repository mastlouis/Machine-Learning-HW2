import numpy as np
from matplotlib import pyplot as plt

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
    error = y - Xtilde.T.dot(w)
    return np.mean(error * error)
    # return 1 / 5000 * (y - Xtilde.T.dot(w)).T.dot(y - Xtilde.T.dot(w))
    # return  Xtilde.dot(Xtilde.T.dot(w) - y) / Xtilde.shape[1]

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    temp = np.dot(Xtilde.T, w)
    temp -= y
    temp = Xtilde.dot(temp)
    temp /= Xtilde.shape[1]
    wslope = w[:-1]
    temp2 = wslope * wslope.dot(wslope)
    temp2 /= Xtilde.shape[1]
    temp2 *= alpha
    return temp + np.append(0,temp2)
    # return Xtilde.dot(np.dot(Xtilde.T, w) - y)/Xtilde.shape[0]

    # return np.dot(np.dot(Xtilde.T, w) - y, Xtilde)/Xtilde.shape[1]

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1(Xtilde, y):
    Xtranspose = np.dot(Xtilde, Xtilde.T)
    Xid = np.eye(Xtranspose.shape[0])
    return np.dot(np.linalg.solve(Xtranspose, Xid), Xtilde.dot(y))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    weights = gradientDescent(Xtilde, y)
    return weights

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    weights = gradientDescent(Xtilde, y, ALPHA)
    return weights

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    # EPSILON = 1e-3
    T = 5000  # Number of gradient descent iterations
    weights = np.random.randn(Xtilde.shape[0]) * 0.01  # Standard deviation of 0.01
    for i in range(T):
        weights = weights - (EPSILON * gradfMSE(weights, Xtilde, y, alpha))
    return weights


def display_image(x):
    x = np.reshape(x, (48, 48))
    plt.imshow(x.T)
    plt.show()


if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    print("Method one training: ", fMSE(w1, Xtilde_tr, ytr))
    print("Method one testing: ", fMSE(w1, Xtilde_te, yte))
    print(w1)
    w2 = method2(Xtilde_tr, ytr)
    print("Method two training: ", fMSE(w2, Xtilde_tr, ytr))
    print("Method two testing: ", fMSE(w2, Xtilde_te, yte))
    print(w2)
    w3 = method3(Xtilde_tr, ytr)
    print("Method three training: ", fMSE(w3, Xtilde_tr, ytr))
    print("Method three testing: ", fMSE(w3, Xtilde_te, yte))
    print(w3)

    w1_test = fMSE(w1, Xtilde_te, yte)
    w2_test = gradfMSE(w2, Xtilde_te, yte)
    w3_test = gradfMSE(w3, Xtilde_te, yte)
    # Report fMSE cost using each of the three learned weight vectors
    reg_gradient_desc = gradfMSE(w3, Xtilde_tr, ytr)
    print("Regularized_gradient_descent_MSE_training: ", reg_gradient_desc)
    print("One_shot_test_MSE: " + str(w1_test))
    print("Simple_gradient_descent_test_MSE: " + str(w2_test))
    print("Regularized_gradient_descent_MSE_training: " + str(w3_test))

    display_image(w1[:-1])
    display_image(w2[:-1])
    display_image(w3[:-1])

    w3guesses = Xtilde_te.T.dot(w3)
    w3errors = w3guesses - yte
    w3errors = w3errors * w3errors
    for i in range(5):
        worst_error_index = w3errors.argmax()
        print('Photo number ' + str(worst_error_index) + ' was number ' + str(i+1) + ' for worst error.')
        print('The actual age was ' + str(yte[worst_error_index]) + ' while the model guessed ' + str(w3guesses[worst_error_index]))
        w3errors[worst_error_index] = 0  # Let the next worst item be the worst
        display_image(Xtilde_te[:-1,worst_error_index])