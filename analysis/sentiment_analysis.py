import nltk
nltk.download('vader_lexicon')

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load the TSV file into a DataFrame
df = pd.read_csv('p2 transcriptions/p2.tsv', delimiter="\t")

# Calculate the duration of each row
df['time'] = df['end'] - df['start']

# --- Sentiment Analysis ---
sia = SentimentIntensityAnalyzer()
results = df['text'].apply(lambda text: sia.polarity_scores(text))
labels = results.apply(lambda score: 'POSITIVE' if score['compound'] > 0 else 'NEGATIVE')
scores = results.apply(lambda score: score['compound'])
df['labels'] = labels
df['scores'] = scores

# --- Text Length Analysis ---
df['text_length'] = df['text'].apply(lambda x: len(x))

# --- Time Analysis ---
df['start_seconds'] = df['start'] / 1000
df['time_seconds'] = df['time'] / 1000

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.scatter(df['start_seconds'], df['time_seconds'], c=df['scores'], cmap='RdYlGn')
plt.title('Sentiment Over Time')
plt.xlabel('Start Time (seconds)')
plt.ylabel('Duration (seconds)')
plt.colorbar(label='Sentiment Score')
plt.show()

# --- Print some stats ---
print("Mean text length:", df['text_length'].mean())
print("Median text length:", df['text_length'].median())