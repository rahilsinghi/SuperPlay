import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Spotify credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

# Initialize Spotify API client
scope = "user-library-read"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=scope
))

# Genius credentials
GENIUS_ACCESS_TOKEN = os.getenv('GENIUS_CLIENT_ACCESS_TOKEN')
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

# Prepare empty list to store track details
tracks_data = []

# Function to collect track data
def collect_track_data(track):
    track_info = {
        "track_name": track['name'],
        "artist_name": track['artists'][0]['name'],
        "track_id": track['id'],
        "album_name": track['album']['name']
    }

    # Get audio features
    audio_features = sp.audio_features(track['id'])[0]
    if audio_features:
        track_info.update({
            "danceability": audio_features['danceability'],
            "energy": audio_features['energy'],
            "key": audio_features['key'],
            "loudness": audio_features['loudness'],
            "mode": audio_features['mode'],
            "speechiness": audio_features['speechiness'],
            "acousticness": audio_features['acousticness'],
            "instrumentalness": audio_features['instrumentalness'],
            "liveness": audio_features['liveness'],
            "valence": audio_features['valence'],
            "tempo": audio_features['tempo'],
            "duration_ms": audio_features['duration_ms']
        })

    # Fetch lyrics using Genius API
    try:
        song = genius.search_song(track_info["track_name"], track_info["artist_name"])
        if song:
            track_info["lyrics"] = song.lyrics
        else:
            track_info["lyrics"] = None
    except Exception as e:
        track_info["lyrics"] = None

    return track_info

# Search for popular tracks to gather data
search_query = "top tracks 2023"
results = sp.search(q=search_query, type='track', limit=20)

for item in results['tracks']['items']:
    track_data = collect_track_data(item)
    tracks_data.append(track_data)
    time.sleep(1)  # Adding a delay to avoid rate limiting

# Save data into a Pandas DataFrame
df_tracks = pd.DataFrame(tracks_data)

# Save to CSV for future use
df_tracks.to_csv('tracks_dataset.csv', index=False)
