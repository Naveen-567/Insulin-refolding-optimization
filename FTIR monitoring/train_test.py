from sklearn.model_selection import train_test_split

def tts(x,y,ts):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = ts, random_state = 0)
    return X_train, X_test, y_train, y_test