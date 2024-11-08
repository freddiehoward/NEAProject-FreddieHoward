import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample data setup for testing
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
y = np.array([10, 20, 30, 40])

# Initialize scalers
mm = MinMaxScaler()
ss = StandardScaler()

def test_standard_scaler_initialization():
    '''Test that StandardScaler is initialized and can transform data'''
    X_trans = ss.fit_transform(X)
    assert X_trans.shape == X.shape, "Transformed X should match the original shape of X."

def test_min_max_scaler_initialization():
    '''Test that MinMaxScaler is initialized and can transform data'''
    y_reshaped = y.reshape(-1, 1)
    y_trans = mm.fit_transform(y_reshaped)
    assert y_trans.shape == y_reshaped.shape, "Transformed y should match the reshaped shape of y."

def test_X_trans_mean_approx_zero():
    '''Test that StandardScaler scales X to have approximately zero mean for each feature'''
    X_trans = ss.fit_transform(X)
    means = X_trans.mean(axis=0)
    for mean in means:
        assert abs(mean) < 1e-6, "Each feature's mean in X_trans should be close to zero."

def test_X_trans_unit_variance():
    '''Test that StandardScaler scales X to have approximately unit variance for each feature'''
    X_trans = ss.fit_transform(X)
    variances = X_trans.var(axis=0)
    for variance in variances:
        assert abs(variance - 1) < 1e-6, "Each feature's variance in X_trans should be close to one."

def test_y_trans_range():
    '''Test that MinMaxScaler scales y to the range [0, 1]'''
    y_reshaped = y.reshape(-1, 1)
    y_trans = mm.fit_transform(y_reshaped)
    for value in y_trans:
        assert 0 <= value <= 1, "Values in y_trans should be within the range [0, 1]."

def test_y_reshaping():
    '''Test that y reshapes correctly for scaling with MinMaxScaler'''
    y_reshaped = y.reshape(-1, 1)
    assert y_reshaped.shape[0] == len(y), "Reshaped y should have the same number of rows as original y."
    assert y_reshaped.shape[1] == 1, "Reshaped y should have a single column."