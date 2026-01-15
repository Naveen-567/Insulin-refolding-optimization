from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from pyopls import OPLS

def model(X_train, X_test, y_train, y_test):
    #opls = OPLS(11)
    #X_train = opls.fit_transform(X_train,y_train)
    #X_test = opls.transform(X_test)
    regressor=xgb.XGBRegressor(eval_metric='rmsle')
    param_grid = {"max_depth":    [2, 4, 5, 6],
                  "n_estimators": [100, 200, 500, 600, 700],
                  "learning_rate": [0.01, 0.015, 0.5, 0.02]}
    search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
    bestp = search.best_params_
    regressor=xgb.XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                               n_estimators  = search.best_params_["n_estimators"],
                               max_depth     = search.best_params_["max_depth"],)
    #regressor = xgb.XGBRegressor(learning_rate = 0.01, n_estimators = 500, max_depth = 4)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    y_pred2 = regressor.predict(X_train)
    v1 = r2_score(y_test, y_pred)
    v2 = r2_score(y_train,y_pred2)
    return v1, v2, bestp