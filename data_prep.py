import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    print(data.info())
    print(data.describe())

    #profiering, kontrollera saknade värden
    missing_values = data.isnull().sum()
    print('Missing values per column')
    print(missing_values)
    
    #Visualisera saknade värden som en värmekarta
    plt.figure(figsize=(12,6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()
    
    #väljer endast de kolumner som är av numeriska typer (float,int)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    # fyller saknade värden med medelvärdet endast för numeriska kolumner
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    # data.fillna(data.mean(), inplace=True)
    return data

def create_visualizations(data):
    #skapa diagram
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data, x='Wind', y='Solar')
    plt.title('Wind vs Solar Power')
    plt.xlabel('Wind Power')
    plt.ylabel('Solar Power')
    plt.show()

def feature_engineering(data):
    # Skapa nya funktioner som moving averages, lagged features etc.
    data['Weelday'] = pd.to_datetime(data['Date']).dt.weekday
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    
    return data

def main():
    file_path = './data/opsd_germany_daily.csv'
    data = load_data(file_path)
    processed_data = preprocess_data(data)
        
    create_visualizations(processed_data)
    
    processed_data.to_csv('./data/cleaned_data.csv', index=False)
    
if __name__ == '__main__':
    main()
