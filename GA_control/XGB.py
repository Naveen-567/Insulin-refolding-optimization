import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE

trained_model = None

def train_model():
    global trained_model
    
    df = pd.read_excel('your data')
    df = df.dropna(axis=1)
    x = df.drop(['Yield %'], axis = 1)
    y = df['Yield %']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    # Train the model
    regressor = xgb.XGBRegressor(learning_rate = 0.01, reg_lambda = 1.0, n_estimators = 400, max_depth = 12)
    regressor.fit(X_train, y_train)

    # Evaluate the model
    y_pred = regressor.predict(X_test)
    print("R2_test:", r2_score(y_test, y_pred))
    
    y_pred2 = regressor.predict(X_train)
    print("R2_train or R2_calib:", r2_score(y_train, y_pred2))
    
    rmse = np.sqrt(MSE(y_test, y_pred))
    mse = MSE(y_test, y_pred)
    print("MSE:", mse)
    print("RMSE:", rmse)
    
    trained_model = regressor
    return regressor

def predict_yield(pH, Temp, Conc, DF, Time, model=None):
    global trained_model
    
    if model is None:
        if trained_model is None:
            raise ValueError("No trained model available. Please train the model first using train_model()")
        model = trained_model
    
    input_data = np.array([[pH, Conc, Temp, DF, Time]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return prediction[0]

# Main execution
if __name__ == "__main__":
    model = train_model()
    print("\n" + "="*50)
    print("Example Predictions:")
    print("="*50)
    pH = 10.5
    Temp = 14.5
    Conc = 65
    DF = 14.5
    
    time_points = [1, 2, 3]
    for time in time_points:
        predicted_yield = predict_yield(pH, Temp, Conc, DF, time)
        print(f"Time: {time}hr | pH: {pH}, Temp: {Temp}, Conc: {Conc}, DF: {DF}")
        print(f"Predicted Yield: {predicted_yield:.2f}%")
    print(f"pH: {pH}, Temp: {Temp}, Conc: {Conc}, DF: {DF}")
    print(f"Predicted Yield: {predicted_yield:.2f}%")