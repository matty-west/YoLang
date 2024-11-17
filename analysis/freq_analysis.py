import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd
import os
print(os.getcwd())

for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Load the TSV file
df = pd.read_csv('p2 transcriptions/p2.tsv', sep='\t')

# Preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return tokens

df['tokens'] = df['text'].apply(preprocess_text)

# Perform word frequency analysis
all_words = [word for tokens in df['tokens'] for word in tokens]
word_counts = Counter(all_words)

# Print the most common words
print(word_counts.most_common(10))