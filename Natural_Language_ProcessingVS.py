import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('vgsales.csv')

df['Platform'].value_counts()

print("First few rows of the dataset:")
print(df.head())

print("\nInformation about the dataset:")
print(df.info())

print("\nSummary statistics of the dataset:")
print(df.describe())


print("\nNumber of games per platform:")
print(df['Platform'].value_counts())

print("Missing values in the dataset:")
print(df.isnull().sum())

median_year = df['Year'].median()
df['Year'].fillna(median_year, inplace=True)

df.dropna(subset=['Publisher'], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

plt.hist(df['Global_Sales'], bins=20)
plt.xlabel('Global Sales')
plt.ylabel('Frequency')
plt.title('Distribution of Global Sales')
plt.show()

genre_counts = df['Genre'].value_counts()
plt.bar(genre_counts.index, genre_counts.values)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Distribution of Games by Genre')
plt.xticks(rotation=90)
plt.show()