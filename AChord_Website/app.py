from flask import Flask, render_template, request, jsonify
import pandas as pd
import ast
import re
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# ----------------------------
# 🔐 Spotify API
# ----------------------------
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET"
))

# ----------------------------
# 📂 Load Dataset
# ----------------------------
df = pd.read_csv("songs_with_genres_final.csv")

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

def normalize_name(x):
    x = x.lower()
    x = re.sub(r'[^\w\s]', '', x)
    return x.strip()

df['artists'] = df['artists'].apply(ensure_list)
df['name_clean'] = df['name'].str.strip().str.lower()

# ----------------------------
# 🎼 MULTI-GENRE ENCODING
# ----------------------------
df['genre_list'] = df['genres_filtered'].fillna('').apply(
    lambda x: [g.strip() for g in x.split(',') if g]
)

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genre_list'])

genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_) * 0.5
df = pd.concat([df, genre_df], axis=1)

# ----------------------------
# 🎯 Features
# ----------------------------
base_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'popularity'
]

features = base_features + list(genre_df.columns)

df[features] = df[features].fillna(0)

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

    except:
        return {
            "image": "https://via.placeholder.com/200",
            "url": "#"
        }

# ----------------------------
# 🎵 Helpers
# ----------------------------
def get_main_artist(lst):
    return lst[0] if isinstance(lst, list) and lst else None

def has_common_artist(a, b):
    return len(set(a) & set(b)) > 0

# ----------------------------
# 🎛️ CONFIG (BALANCED)
# ----------------------------
COLLAB_WEIGHT = 0.15
SAME_ARTIST_PENALTY = 0.1

# ----------------------------
# 🎵 RECOMMENDER (NEW)
# ----------------------------
def recommend_songs(song_name, artist_name=None, num_recommendations=5):

    song_name_clean = song_name.strip().lower()

    matches = df[df['name_clean'] == song_name_clean]

    if artist_name:
        matches = matches[matches['artists'].apply(
            lambda x: artist_name.lower() in [a.lower() for a in x]
        )]

    if matches.empty:
        return []

    idx = matches.sort_values(by='popularity', ascending=False).index[0]

    # similarity
    query_vec = X_scaled[idx].reshape(1, -1)
    similarities = cosine_similarity(query_vec, X_scaled)[0]

    # artist logic
    input_artists = df.loc[idx, 'artists']
    main_artist = get_main_artist(input_artists)

    same_artist = df['artists'].apply(
        lambda x: main_artist in x if main_artist else False
    ).astype(int)

    same_artist_penalty = same_artist * SAME_ARTIST_PENALTY

    collab_bonus = df['artists'].apply(
        lambda x: has_common_artist(x, input_artists)
    ).astype(int) * COLLAB_WEIGHT

    final_scores = similarities + collab_bonus - same_artist_penalty

    # remove itself
    final_scores[idx] = -1

    # ----------------------------
    # 🔥 DEDUP + MAX 2 PER ARTIST
    # ----------------------------
    sorted_indices = final_scores.argsort()[::-1]

    seen_tracks = set()
    artist_count = defaultdict(int)
    top_indices = []

    for idx_candidate in sorted_indices:
        name = normalize_name(df.loc[idx_candidate, 'name'])
        artist = get_main_artist(df.loc[idx_candidate, 'artists'])

        if artist is None:
            continue

        key = (name, artist)

        if key in seen_tracks:
            continue

        if artist_count[artist] >= 2:
            continue

        seen_tracks.add(key)
        artist_count[artist] += 1
        top_indices.append(idx_candidate)

        if len(top_indices) >= num_recommendations:
            break

    results = df.loc[top_indices][
        ['id', 'name', 'artists', 'genres_filtered', 'popularity']
    ].to_dict(orient='records')

    for r in results:
        r.update(get_spotify_data(r['id']))

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