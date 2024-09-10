import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# etfersom denna fil körs separat läser vi in datan igen
# data = pd.read_csv('data/sustainable-energy.csv')
data = pd.read_csv('data/opsd_germany_daily.csv')

numeric_data = data.select_dtypes(include=['float64', 'int64'])

# korrelation mellan variabler
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# energikonsumtion över tid

# Diagrammet visualiserar hur energikonsumtionen förändras över tid, vilket kan ge insiter om trender och
# jämförelser mellan olika år och länder. 
# Ett effektivt sätt att förstå den tidsmässiga utvecklingen i energikonsumtion

# kontrollera att nödvändiga kolumner finns

if 'Year' in data.columns and 'Energy_Consumption' and 'Country' in data.columns:
    plt.figure(figsize=(14,6))
    sns.lineplot(data=data, x='Year', y='Energy_Consumption', hue='Country')
    plt.title('Energy Consumption OT')
    plt.xlabel('Year')
    plt.ylabel('Energy Consumption')
    plt.show()
else:
    print("Columns for 'Year', 'Energy_Consumption' or'Country' is missing")
    

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

plt.savefig('/data/')