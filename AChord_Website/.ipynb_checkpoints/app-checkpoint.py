from flask import Flask, render_template, request, jsonify
import pandas as pd
import ast
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# ----------------------------
# 🔐 Spotify API
# ----------------------------
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="fe0663759ed0404cb9a1573dbc448026",
    client_secret="5969b6b401b64f59ac90a122e53d57b1"
))

# ----------------------------
# 📂 Load Dataset
# ----------------------------
df = pd.read_csv("songs_with_genres.csv")

# ----------------------------
# 🧹 Clean Data
# ----------------------------
def ensure_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    return x

df['artists'] = df['artists'].apply(ensure_list)
df['name_clean'] = df['name'].str.strip().str.lower()

# ----------------------------
# 🎼 GENRE NORMALIZATION
# ----------------------------
df['clean_genre'] = df['clean_genre'].str.lower().str.strip()

genre_map = {
    "hip hop": "hip-hop",
    "rap": "hip-hop",
    "trap": "hip-hop",
    "k-pop": "pop",
    "kpop": "pop",
    "pop music": "pop"
}

df['clean_genre'] = df['clean_genre'].replace(genre_map)

# ----------------------------
# 🎼 Genre Encoding
# ----------------------------
genre_dummies = pd.get_dummies(df['clean_genre'], prefix='genre') * 0.3
df = pd.concat([df, genre_dummies], axis=1)

# ----------------------------
# 🎯 Features
# ----------------------------
base_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'popularity'
]

features = base_features + list(genre_dummies.columns)

# ----------------------------
# ⚙️ Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features]).astype('float32')

# ----------------------------
# ⚡ Spotify Cache
# ----------------------------
spotify_cache = {}

def get_spotify_data(track_id):
    if track_id in spotify_cache:
        return spotify_cache[track_id]

    try:
        track = sp.track(track_id)
        images = track['album'].get('images', [])

        data = {
            "image": images[0]['url'] if images else "https://via.placeholder.com/200",
            "url": track['external_urls']['spotify']
        }

        spotify_cache[track_id] = data
        return data

    except Exception as e:
        print("Spotify Error:", e)
        return {
            "image": "https://via.placeholder.com/200",
            "url": "#"
        }

# ----------------------------
# 🎵 Helpers
# ----------------------------
def get_main_artist(artist_list):
    return artist_list[0] if isinstance(artist_list, list) and len(artist_list) > 0 else None

def get_featured_artists(artist_list):
    return artist_list[1:] if isinstance(artist_list, list) and len(artist_list) > 1 else []

def has_common_artist(list1, list2):
    return len(set(list1) & set(list2)) > 0

# ----------------------------
# 🎛️ CONFIG
# ----------------------------
COLLAB_WEIGHT = 0.3
SAME_ARTIST_PENALTY = 0.4
POPULARITY_WEIGHT = 50

# ----------------------------
# 🎵 RECOMMENDER (STRICT + FIXED MATCH)
# ----------------------------
def recommend_songs(song_name, artist_name=None, num_recommendations=5):
    song_name_clean = song_name.strip().lower()

    matches = df[df['name_clean'] == song_name_clean]

    # 🔥 Filter by artist
    if artist_name:
        matches = matches[matches['artists'].apply(
            lambda x: artist_name.lower() in [a.lower() for a in x]
        )]

    if matches.empty:
        return []

    idx = matches.sort_values(by='popularity', ascending=False).index[0]

    query_vector = X_scaled[idx].reshape(1, -1)
    similarities = cosine_similarity(query_vector, X_scaled)[0]

    target_pop = df.loc[idx, 'popularity']
    target_genre = df.loc[idx, 'clean_genre']

    input_artists = df.loc[idx, 'artists']
    input_main_artist = get_main_artist(input_artists)
    input_features = get_featured_artists(input_artists)

    popularity_diff = abs(df['popularity'] - target_pop)
    popularity_penalty = popularity_diff / 100

    main_artist_series = df['artists'].apply(get_main_artist)
    featured_series = df['artists'].apply(get_featured_artists)

    same_artist = (main_artist_series == input_main_artist).astype(int)
    same_artist_penalty = same_artist * SAME_ARTIST_PENALTY

    collab_bonus = featured_series.apply(
        lambda x: has_common_artist(x, input_features)
    ).astype(int) * COLLAB_WEIGHT

    final_scores = (
        similarities
        + collab_bonus
        - (POPULARITY_WEIGHT * popularity_penalty)
        - same_artist_penalty
    )

    # 🔒 STRICT GENRE FILTER
    same_genre_idx = df[df['clean_genre'] == target_genre].index
    same_scores = final_scores[same_genre_idx]

    # remove query song
    same_scores = same_scores[same_genre_idx != idx]
    same_genre_idx = same_genre_idx[same_genre_idx != idx]

    top = same_scores.argsort()[::-1][:num_recommendations]
    top_indices = same_genre_idx[top]

    results = df.loc[top_indices][
        ['id', 'name', 'artists', 'clean_genre', 'popularity']
    ].to_dict(orient='records')

    for r in results:
        spotify = get_spotify_data(r['id'])
        r.update(spotify)
        time.sleep(0.05)

    return results

# ----------------------------
# 🔍 Search
# ----------------------------
@app.route("/search")
def search():
    query = request.args.get("q", "").lower()

    if len(query) < 2:
        return jsonify({"results": []})

    results = df[df['name_clean'].str.contains(query, na=False)].head(10)

    return jsonify({
        "results": results[['name', 'artists']].to_dict(orient="records")
    })

# ----------------------------
# 🎯 Recommend API
# ----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    song = data.get("song")
    artist = data.get("artist")

    recs = recommend_songs(song, artist)

    return jsonify(recs)

# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)