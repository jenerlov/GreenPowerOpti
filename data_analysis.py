import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def analyze_data(data):
    
    # numeriska kolumner för korrelation
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    #korrelation mellan variabler
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    
    # visualisera elkonsumtion över tid
    plt.figure(figsize=(14,6))
    sns.lineplot(data=data, x='Date', y='Consumption')
    plt.title('Energy Consumption Over Time')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption')
    plt.show()
    
    # visualisera förnybar energi (vind + sol) över tid
    plt.figure(figsize=(14,6))
    sns.lineplot(data=data, x='Date', y='Wind+Solar')
    plt.title('Renewable Energy (Wind+Solar) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Energy from Wind and Solar')
    plt.show()
    
def main():
    data = load_data('./data/cleaned_data.csv')
    analyze_data(data)
    
if __name__ == '__main__':
    main()