'''
sklearn = Scikit-learn
'''
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()

'''
X_trans is the X input data after it has been fitted, ie the paramaters
needed for scaling, the min and max value as well as the mean and standard
deviation, have been worked out, and also transformed/scaled so the equations
in my documentation have been performed. The same goes for y_trans
'''
X_trans = ss.fit_transform(X)
y_trans = mm.fit_transform(y.reshape(-1, 1))
'''
y needs to be reshaped so that it has the same size of the final dimension
ie if X has the shape (2516, 4), to be able to be used, y which would have a
shape of (2516), needs to be changed to a shape of (2516, 1) to be used with X
'''