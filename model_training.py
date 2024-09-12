import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

print("Current working directory:", os.getcwd())

def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def train_model(model_class, X_train, y_train):
    model = model_class()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    print(f'Model Mean Squared Error: {mse}')
    print(f'Model R² Score: {r2}')
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {cv_scores.mean()}')
    
    return mse, r2

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print('Best parameters found:', grid_search.best_params_)
    return grid_search.best_estimator_

def plot_performance(performance_dict):
    models = list(performance_dict.keys())
    mse_values = [performance_dict[model]['MSE'] for model in models]
    r2_values = [performance_dict[model]['R2'] for model in models]

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0].bar(models, mse_values, color='lightblue')
    ax[0].set_title('Model Mean Squared Error (MSE)')
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('MSE')
    ax[0].tick_params(axis='x', rotation=45)

    ax[1].bar(models, r2_values, color='lightgreen')
    ax[1].set_title('Model R² Score')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('R² Score')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show(block=True)  # Ensure the plot stays open

def analyze_data(data):
    print("Analyzing data...")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    try:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show(block=True)
    except Exception as e:
        print(f"Error in plotting correlation matrix: {e}")
    
    try:
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=data, x='Date', y='Consumption')
        plt.title('Energy Consumption Over Time')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.xticks(rotation=45)
        plt.show(block=True)
    except Exception as e:
        print(f"Error in plotting energy consumption: {e}")
    
    try:
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=data, x='Date', y='Wind+Solar')
        plt.title('Renewable Energy (Wind+Solar) Over Time')
        plt.xlabel('Date')
        plt.ylabel('Energy from Wind and Solar')
        plt.xticks(rotation=45)
        plt.show(block=True)
    except Exception as e:
        print(f"Error in plotting renewable energy: {e}")

def visualize_predictions(data, model):
    X = data[['Wind', 'Solar', 'Wind+Solar']]
    y_pred = model.predict(X)
    
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(data['Date'], data['Consumption'], label='Actual Consumption')
        plt.plot(data['Date'], y_pred, label='Predicted Consumption', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.title('Actual vs Predicted Energy Consumption')
        plt.legend()
        plt.xticks(rotation=45)
        plt.show(block=True)
    except Exception as e:
        print(f"Error in plotting predictions: {e}")

def main():
    data = load_data('./data/cleaned_data.csv')
    
    X = data[['Wind', 'Solar', 'Wind+Solar']]
    y = data['Consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    performance = {}

    # Train and evaluate Random Forest
    rf_model = train_model(RandomForestRegressor, X_train, y_train)
    mse_rf, r2_rf = evaluate_model(rf_model, X_test, y_test)
    performance['Random Forest'] = {'MSE': mse_rf, 'R2': r2_rf}

    # Train and evaluate Gradient Boosting
    gb_model = train_model(GradientBoostingRegressor, X_train, y_train)
    mse_gb, r2_gb = evaluate_model(gb_model, X_test, y_test)
    performance['Gradient Boosting'] = {'MSE': mse_gb, 'R2': r2_gb}

    # Hyperparameter tuning
    tuned_rf_model = hyperparameter_tuning(X_train, y_train)
    mse_tuned_rf, r2_tuned_rf = evaluate_model(tuned_rf_model, X_test, y_test)
    performance['Tuned Random Forest'] = {'MSE': mse_tuned_rf, 'R2': r2_tuned_rf}

    # Plot performance
    plot_performance(performance)
    
    # Analyze data
    analyze_data(data)

if __name__ == '__main__':
    main()
