import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    return pd.read(file_path)

def train_model(data):
    X = data [['Wind', 'Solar', 'Wind+Solar']]
    y = data['Consumption']
    
    # dela upp i träning och testning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    
    return model

def main():
    data = load_data('./data/cleaned_data.csv')
    model = train_model(data)
    
    if __name__ == '__main__':
        main()