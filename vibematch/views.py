from django.shortcuts import redirect, render
from utils import get_spotify_token
import spotipy
from user_profile.models import Vibe, User
import numpy as np
import re
from dashboard.models import EmotionVector
from django.db.models import OuterRef, Subquery, F
from django.contrib import messages


def vibe_match(request):
    token_info = get_spotify_token(request)
    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])

        user_info = sp.current_user()
        user_id = user_info["id"]
        matches = k_nearest_neighbors(2, user_id)

        context = {"neighbors": matches}

        return render(request, "match.html", context)
    else:
        # No token, redirect to login again
        messages.error(request, "Vibe_match failed, please try again later.")
        return redirect("login:index")


def k_nearest_neighbors(k, target_user_id):
    # Fetch Emotion Vectors
    emotion_vectors = {
        str(emotion.emotion).lower(): vector_to_array(emotion.vector)
        for emotion in EmotionVector.objects.all()
    }

    # Get the most recent vibe for each user
    # Subquery to get the latest vibe_time for each user
    latest_vibe_times = (
        Vibe.objects.filter(user_id=OuterRef("user_id"))
        .order_by("-vibe_time")
        .values("vibe_time")[:1]
    )

    # Filter Vibe objects to only get those matching the latest vibe_time for each user
    latest_vibes = Vibe.objects.annotate(
        latest_vibe_time=Subquery(latest_vibe_times)
    ).filter(vibe_time=F("latest_vibe_time"))

    # Query to join latest vibes with the User model
    all_users = latest_vibes.filter(
        user_id__in=User.objects.all().values_list("user_id", flat=True)
    ).values_list(
        "user_id",
        "user_lyrics_vibe",
        "user_audio_vibe",
        "user_acousticness",
        "user_danceability",
        "user_energy",
        "user_valence",
        flat=False,
    )

    all_users_array = []
    target_user_features = None

    for user in all_users:
        user_id, lyrics_vibe, audio_vibe, *features = user
        print(user_id)
        lyrics_vector = emotion_vectors.get(
            lyrics_vibe, np.zeros_like(next(iter(emotion_vectors.values())))
        )
        audio_vector = emotion_vectors.get(
            audio_vibe, np.zeros_like(next(iter(emotion_vectors.values())))
        )
        features = [float(feature) for feature in features]

        if user_id != target_user_id:
            all_users_array.append(
                (user_id, [*lyrics_vector, *audio_vector, *features])
            )
        else:
            target_user_features = [*lyrics_vector, *audio_vector, *features]

    if target_user_features is None:
        return []

    # Calculate distances, excluding the target user
    distances = [
        (user_id, euclidean_distance(target_user_features[1:], features[1:]))
        for user_id, features in all_users_array
    ]

    # Sort by distance and select top k
    nearest_neighbors_ids = sorted(distances, key=lambda x: x[1])[:k]
    nearest_neighbors = [
        User.objects.get(user_id=uid).username for uid, _ in nearest_neighbors_ids
    ]

    return nearest_neighbors


def euclidean_distance(user_1, user_2):
    user_1 = np.array(user_1)
    user_2 = np.array(user_2)
    return np.sqrt(np.sum((user_1 - user_2) ** 2))


def vector_to_array(vector_str):
    clean = re.sub(r"[\[\]\n\t]", "", vector_str)
    clean = clean.split()
    clean = [float(e) for e in clean]
    return np.array(clean)
