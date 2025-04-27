import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('IMDB-Movie-Data.csv')

# Remove rows with missing revenue (target variable)
data=data.dropna(subset=['Revenue (Millions)'])

# Optional: Fill or drop other NaNs
data.fillna("", inplace=True)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
data['Sentiment_Score'] = data['Description'].apply(lambda x: sid.polarity_scores(x)['compound'])

if 'Genre' in data.columns:
    data['Genre'] = data['Genre'].apply(lambda x: x.split('|')[0])  # Or one-hot encode all genres
    data = pd.get_dummies(data, columns=['Genre'], drop_first=True)
else:
    print("Column 'Genre' not found. Available columns:", data.columns)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target
features = ['Rating', 'Metascore', 'Votes', 'Sentiment_Score'] + [col for col in data.columns if col.startswith('Genre_')]
X = data[features]
y = data['Revenue (Millions)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data=data.dropna(subset=['Revenue (Millions)'])
data = data.fillna(0)
# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")

# Remove rows with missing revenue (target variable)
data=data.dropna(subset=['Revenue (Millions)'])
for col in ['Rating', 'Metascore', 'Votes']:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
data.fillna("Unknown", inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
genre_columns = [col for col in data.columns if col.startswith('Genre_')]

# Calculate average sentiment for each genre
genre_sentiment = data.groupby(genre_columns)['Sentiment_Score'].mean()
# Reshape and sort for plotting
genre_sentiment = genre_sentiment.reset_index().melt(id_vars=['Sentiment_Score'], value_vars=genre_columns)
genre_sentiment = genre_sentiment.groupby('value')['Sentiment_Score'].mean().sort_values(ascending=False)


# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_sentiment.values, y=genre_sentiment.index)
plt.title('Average Sentiment Score by Genre')
plt.xlabel('Sentiment Score')
plt.ylabel('Genre')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import nltk
nltk.download('vader_lexicon')

data = pd.read_csv('IMDB-Movie-Data.csv')
print(data.head())
print(data.info())

# Drop missing target (revenue)
data = data.dropna(subset=['Revenue (Millions)'])

# Fill others as needed
data.fillna({'Metascore': data['Metascore'].median()}, inplace=True)
data.fillna("", inplace=True)

# Simplify genre (first listed genre only)
data['Main_Genre'] = data['Genre'].apply(lambda x: x.split(',')[0])

# Simulate basic review sentiment (real review scraping can be added later)
data['Reviews'] = data['Description']  # Using Description as proxy for reviews

# VADER Sentiment
sid = SentimentIntensityAnalyzer()
data['Sentiment'] = data['Reviews'].apply(lambda x: sid.polarity_scores(x)['compound'])

# One-hot encode Main Genre
data = pd.get_dummies(data, columns=['Main_Genre'], drop_first=True)

# Feature selection
features = ['Rating', 'Metascore', 'Votes', 'Sentiment'] + [col for col in data.columns if col.startswith('Main_Genre_')]
X = data[features]
y = data['Revenue (Millions)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Group and visualize
# Use the original 'Genre' column before one-hot encoding
genre_sentiment = data.groupby(data['Genre'].str.split(',').str[0])['Sentiment'].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_sentiment.values, y=genre_sentiment.index, palette='coolwarm')
plt.title("Average Sentiment by Genre")
plt.xlabel("Sentiment Score")
plt.ylabel("Genre")
plt.show()

data.to_excel("movie_analysis_output.xlsx", index=False)

