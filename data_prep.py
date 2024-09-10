import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data = pd.read_csv('data/sustainable-energy.csv')
data = pd.read_csv('data/opsd_germany_daily.csv')

# print(data.head())

# kontrollera datatyper och saknade värden
print(data.info())
print(data.describe())

# profilering, kontrollera saknade värden
missing_values = data.isnull().sum()
print('Missing values per column:')
print(missing_values)

sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# hantera saknade värden 
data.fillna(data.mean(), inplace=True)

# skapa en visualisering för att analysera förnybar energianvändning vs. koldioxidutsläpp
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='Electricity from renewables (TWh)', y='Value_co2_emission_kt_by_country', hue='Entity')
plt.title('Renewable Energy vs CO2 Emissions')
plt.xlabel('Electricity from Renewables (TWh)')
plt.ylabel('CO2 Emissions (kt)')
plt.show()



