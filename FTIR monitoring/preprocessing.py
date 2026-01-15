from scipy.signal import savgol_filter
from pyspectra.transformers.spectral_correction import msc, detrend ,sav_gol,snv
from sklearn.preprocessing import MinMaxScaler

def preprocess (df,n):
    row = []
    for i in range (0, n):
        y = df.iloc[:,i].values
        ySG = savgol_filter (y, 11, 3, deriv=3, mode='interp', cval=0.0)
        SNV= snv()
        ySN=SNV.fit_transform(ySG)
        row.append(ySN)
    return row

def norm(X_train, X_test, y_train):
    X_norm1 = MinMaxScaler.fit_transform(X_train, y_train)
    X_norm2 = MinMaxScaler.transform(X_test)
    return X_norm1, X_norm2