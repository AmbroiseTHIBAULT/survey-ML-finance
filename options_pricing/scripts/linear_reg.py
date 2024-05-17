import pandas as pd 
import numpy as np


def compute_r_ii(X, i):
    return compute_mat_R(X)[i, i]


def compute_mat_R(X):
    # R = I_n-X(X^TX)^{-1}X^T
    return np.eye(len(X)) - X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T)).reset_index(drop=True)


def compute_residuals(y_true, y_pred):
    return (y_true - y_pred).reset_index(drop=True)


def compute_var_hat(y_true, y_pred, p):
    residuals = compute_residuals(y_true, y_pred)
    return (1 / (len(y_true) - p)) * np.sum(residuals**2)


def compute_standardized_residuals_i(X, y_true, y_pred, i, p):
    residuals = compute_residuals(y_true, y_pred)
    r_ii = compute_r_ii(X, i)
    return residuals[i] / (np.sqrt(compute_var_hat(y_true, y_pred, p)) * np.sqrt(r_ii)) 


def compute_standardized_residuals(X, y_true, y_pred, p):
    R = compute_mat_R(X).to_numpy()
    residuals = compute_residuals(y_true, y_pred)
    var_hat = compute_var_hat(y_true, y_pred, p)
    return np.array([residuals[i] / (np.sqrt(var_hat) * np.sqrt(R[i, i])) for i in range(len(y_true))])


def compute_standardized_residuals_fast(X, y_true, y_pred, p):
    R = compute_mat_R(X)
    residuals = compute_residuals(y_true, y_pred)
    var_hat = compute_var_hat(y_true, y_pred, p)
    sqrt_var_hat = np.sqrt(var_hat)
    sqrt_R = np.sqrt(np.diag(R))
    return residuals / (sqrt_var_hat * sqrt_R)