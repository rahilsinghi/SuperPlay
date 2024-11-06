import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid Tkinter issues

# Load the CSV file
df_tracks = pd.read_csv('tracks_dataset.csv')

# Step 1: Handle Missing Data
# Drop rows where crucial information is missing
df_tracks = df_tracks.dropna(subset=['danceability', 'energy', 'tempo'])

# Fill missing lyrics with an empty string
df_tracks['lyrics'] = df_tracks['lyrics'].fillna('')

# Impute numerical features with the mean
imputer = SimpleImputer(strategy='mean')
numerical_features = ['danceability', 'energy', 'tempo', 'valence', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
df_tracks[numerical_features] = imputer.fit_transform(df_tracks[numerical_features])

# Step 2: Remove Duplicates
# Drop duplicate rows based on track ID
df_tracks = df_tracks.drop_duplicates(subset='track_id', keep='first')

# Step 3: Clean Text Data (Lyrics)
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_lyrics(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords
    tokens = word_tokenize(text, language='english')
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing
df_tracks['lyrics'] = df_tracks['lyrics'].apply(preprocess_lyrics)

# Step 4: Encode Categorical Data
# Convert 'mode' to categorical
df_tracks['mode'] = df_tracks['mode'].astype('category')

# One-hot encode categorical columns if they exist in the dataset
categorical_columns = ['genre', 'key', 'time_signature']
for column in categorical_columns:
    if column in df_tracks.columns:
        df_tracks = pd.get_dummies(df_tracks, columns=[column], drop_first=True)

# Step 5: Feature Engineering
# Sentiment Analysis on Lyrics
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a value between -1 and 1

df_tracks['lyrics_sentiment'] = df_tracks['lyrics'].apply(get_sentiment)

# Step 6: Normalize Numerical Features
scaler = StandardScaler()
df_tracks[numerical_features] = scaler.fit_transform(df_tracks[numerical_features])

# Step 7: Save Cleaned Dataset
df_tracks.to_csv('tracks_dataset_cleaned.csv', index=False)

# Step 8: Exploratory Data Analysis (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Save plot as an image
plt.figure(figsize=(10, 6))
sns.histplot(df_tracks['danceability'], bins=20, kde=True)
plt.title('Distribution of Danceability')
plt.savefig('danceability_distribution.png')

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_tracks.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Audio Features')
plt.savefig('correlation_heatmap.png')
