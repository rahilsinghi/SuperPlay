import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius

# Load environment variables
load_dotenv()

# Spotify credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

# Genius credentials
GENIUS_ACCESS_TOKEN = os.getenv('GENIUS_CLIENT_ACCESS_TOKEN')

scope = "user-library-read"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=scope
))

# Test Spotify Authentication: Get Current User's Saved Tracks
results = sp.current_user_saved_tracks(limit=10)
for idx, item in enumerate(results['items']):
    track = item['track']
    print(f"{idx+1}. {track['name']} by {track['artists'][0]['name']}")

# Initialize Genius API client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

# Fetch a track from Spotify
track = sp.search(q='track:Blinding Lights artist:The Weeknd', type='track', limit=1)
track_info = track['tracks']['items'][0]
track_name = track_info['name']
track_artist = track_info['artists'][0]['name']

print(f"Track Name: {track_name}, Artist: {track_artist}")

# Fetch the lyrics from Genius
song = genius.search_song(track_name, track_artist)
if song:
    print(f"Lyrics for {song.title} by {song.artist}:\n")
    print(song.lyrics[:1000])  # Print the first 500 characters of the lyrics

