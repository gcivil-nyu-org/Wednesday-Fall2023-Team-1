from django.shortcuts import render, redirect
import spotipy
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter
import json
import shutil

import threading
from user_profile.views import check_and_store_profile

# from dotenv import load_dotenv
from utils import get_spotify_token, vibe_calc_threads
from django.http import JsonResponse
from user_profile.models import Vibe, UserTop
from django.utils import timezone
import spacy

MAX_RETRIES = 2

# Load spaCy language model from the deployed location
nlp = spacy.load("dashboard/en_core_web_md/en_core_web_md-3.7.0")

""" AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def load_model_from_s3():
    with tempfile.NamedTemporaryFile() as tmp:
        s3.download_file("vibecheck-storage", "cc.en.12.bin", tmp.name)
        model = FastText.load_fasttext_format(tmp.name)
    return model


# Uncomment for loading from S3
model = load_model_from_s3()"""


# Uncomment for manual loading
# model = FastText.load_fasttext_format("dashboard/cc.en.32.bin")


def index(request):
    token_info = get_spotify_token(request)
    # token_info = request.session.get('token_info', None)

    if token_info:
        # Initialize Spotipy with stored access token
        sp = spotipy.Spotify(auth=token_info["access_token"])

        top_tracks = sp.current_user_top_tracks(limit=10, time_range="short_term")

        # Extract seed tracks, artists, and genres
        seed_tracks = [track["id"] for track in top_tracks["items"]]
        recommendations = sp.recommendations(seed_tracks=seed_tracks[:4])

        # EXTRA STUFF
        # top_artists = sp.current_user_top_artists(limit=2)
        # seed_artists = [artist['id'] for artist in top_artists['items']]
        # seed_genres = list(set(genre for artist in top_artists['items'] for genre in artist['genres']))

        tracks = []
        for track in top_tracks["items"]:
            tracks.append(
                {
                    "name": track["name"],
                    "artists": ", ".join(
                        [artist["name"] for artist in track["artists"]]
                    ),
                    "album": track["album"]["name"],
                    "uri": track["uri"],
                }
            )

        recommendedtracks = []
        for track in recommendations["tracks"]:
            recommendedtracks.append(
                {
                    "name": track["name"],
                    "artists": ", ".join(
                        [artist["name"] for artist in track["artists"]]
                    ),
                    "album": track["album"]["name"],
                    "uri": track["uri"],
                }
            )

        # Pass username to navbar
        user_info = sp.current_user()
        username = user_info["display_name"]

        def run_check_and_store_profile():
            check_and_store_profile(request)

        # Create a thread to run the function
        thread = threading.Thread(target=run_check_and_store_profile)

        # Start the thread
        thread.start()

        # Get top tracks
        top_tracks = get_top_tracks(sp)

        # Get top artists and top genres based on artist
        top_artists, top_genres = get_top_artist_and_genres(sp)

        # Get recommendation based on tracks
        recommendedtracks = get_recommendations(sp, top_tracks)

        user_id = user_info["id"]
        current_time = timezone.now().astimezone(timezone.utc)
        midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        recent_top = UserTop.objects.filter(user_id=user_id, time__gte=midnight).first()
        if not recent_top:
            # If no top info for this user today, save new row to UserTop database
            top_data = UserTop(
                user_id=user_id,
                time=current_time,
                top_track=[track["id"] for track in top_tracks],
                top_artist=[artist["id"] for artist in top_artists],
                top_genre=top_genres,
                recommended_tracks=[track["id"] for track in recommendedtracks],
            )
            top_data.save()

        current_year = current_time.year
        vibe_history = Vibe.objects.filter(
            user_id=user_id, vibe_time__year=current_year
        ).values("vibe_time", "user_audio_vibe", "user_lyrics_vibe")
        months = [
            {"number": 0, "short_name": "", "long_name": ""},
            {"number": 1, "short_name": "J", "long_name": "January"},
            {"number": 2, "short_name": "F", "long_name": "February"},
            {"number": 3, "short_name": "M", "long_name": "March"},
            {"number": 4, "short_name": "A", "long_name": "April"},
            {"number": 5, "short_name": "M", "long_name": "May"},
            {"number": 6, "short_name": "J", "long_name": "June"},
            {"number": 7, "short_name": "J", "long_name": "July"},
            {"number": 8, "short_name": "A", "long_name": "August"},
            {"number": 9, "short_name": "S", "long_name": "September"},
            {"number": 10, "short_name": "O", "long_name": "October"},
            {"number": 11, "short_name": "N", "long_name": "November"},
            {"number": 12, "short_name": "D", "long_name": "December"},
        ]

        vibe_or_not = calculate_vibe(sp, midnight)
        # Possible values: already_loaded if vibe already calculated within today,
        # asyn_started if vibe calculation is started and still loading,
        # no_songs if user has 0 recent songs to analyze

        context = {
            "username": username,
            "top_tracks": top_tracks,
            "top_artists": top_artists,
            "top_genres": top_genres,
            "recommendedtracks": recommendedtracks,
            "vibe_history": vibe_history,
            "iteratorMonth": months,
            "iteratorDay": range(0, 32),
            "currentYear": current_year,
            "midnight": midnight,
            "vibe_or_not": vibe_or_not,
        }

        extract_tracks(sp)

        return render(request, "dashboard/index.html", context)
    else:
        # No token, redirect to login again
        debug_info = f"Request: {request}"
        messages.error(
            request,
            f"Dashboard failed, please try again later. Debug info: {debug_info}",
        )
        return redirect("login:index")


def calculate_vibe(request):
    token_info = get_spotify_token(request)

    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])

        # Check if user vibe exists already for today
        user_info = sp.current_user()
        user_id = user_info["id"]
        # current_time = timezone.now()
        # time_difference = current_time - timezone.timedelta(hours=24)
        # recent_vibe = Vibe.objects.filter(user_id=user_id, vibe_time__gte=time_difference).first()
        # if recent_vibe:
        #     vibe_result = recent_vibe.user_vibe
        #     return JsonResponse({'result': vibe_result})
        # Skips having to perform vibe calculations below

    recent_tracks = sp.current_user_recently_played(limit=15)

    if not recent_tracks.get("items", []):
        return "no_songs"
    else:
        track_names = []
        track_artists = []
        track_ids = []

        for track in recent_tracks["items"]:
            track_names.append(track["track"]["name"])
            track_artists.append(track["track"]["artists"][0]["name"])
            track_ids.append(track["track"]["id"])

        # IF TESTING WITH TOP TRACKS INSTEAD OF RECENT
        """ top_tracks = sp.current_user_top_tracks(limit=10, time_range='short_term')
        for track in top_tracks['items']:
            track_names.append(track['name'])
            track_artists.append(track['artists'][0]['name'])
            track_ids.append(track['id']) """

        if track_ids:
            audio_features_list = sp.audio_features(track_ids)
            vibe_result = check_vibe(
                track_names, track_artists, track_ids, audio_features_list
            )
            # Add user vibe to vibe database
            time = timezone.now()
            vibe_data = Vibe(user_id=user_id, user_vibe=vibe_result, vibe_time=time)
            vibe_data.save()
        else:
            vibe_result = "Null"

        return JsonResponse({"result": vibe_result})
    else:
        # No token, redirect to login again
        # ERROR MESSAGE HERE?
        return redirect("login:index")


def logout(request):
    # Clear Django session data
    request.session.clear()
    return redirect("login:index")


def extract_tracks(sp):
    recently_played = sp.current_user_recently_played()
    timestamps = [track["played_at"] for track in recently_played["items"]]
    # Convert to datetime and extract hour and day
    hours_of_day = [datetime.fromisoformat(ts[:-1]).hour for ts in timestamps]
    days_of_week = [datetime.fromisoformat(ts[:-1]).weekday() for ts in timestamps]
    hours_count = Counter(hours_of_day)
    days_count = Counter(days_of_week)

    # Plot by Hour of Day
    hour_fig = go.Figure()
    hour_fig.add_trace(
        go.Bar(
            x=list(hours_count.keys()),
            y=list(hours_count.values()),
            marker_color="blue",
        )
    )
    hour_fig.update_layout(
        title="Listening Patterns by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Tracks Played",
        xaxis=dict(tickvals=list(range(24)), ticktext=list(range(24))),
        plot_bgcolor="black",  # Background color of the plotting area
        paper_bgcolor="black",  # Background color of the entire paper
        font=dict(color="white"),
    )

    # Plot by Day of Week
    days_labels = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    day_fig = go.Figure()
    day_fig.add_trace(
        go.Bar(x=days_labels, y=[days_count[i] for i in range(7)], marker_color="green")
    )

    # Update the layout
    day_fig.update_layout(
        title="Listening Patterns by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Number of Tracks Played",
        plot_bgcolor="black",  # Background color of the plotting area
        paper_bgcolor="black",  # Background color of the entire paper
        font=dict(color="white"),  # To make the font color white for better visibility
    )

    # Save as HTML

    hour_json = hour_fig.to_json()
    day_json = day_fig.to_json()

    # You can save this JSON data to a file or use some other method to transfer it to your webpage.
    with open("hour_data.json", "w") as f:
        json.dump(hour_json, f)

    with open("day_data.json", "w") as f:
        json.dump(day_json, f)

    shutil.move("hour_data.json", "login/static/login/hour_data.json")
    shutil.move("day_data.json", "login/static/login/day_data.json")


"""
# On huggingface spaces
def get_vector(word, model):
    # Get the word vector from the model.
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros(model.vector_size)
"""


def get_task_status(request, midnight):
    token_info = get_spotify_token(request)

    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])
        user_info = sp.current_user()
        user_id = user_info["id"]

        # Check if there is a result in the database
        recent_vibe = Vibe.objects.filter(
            user_id=user_id, vibe_time__gte=midnight
        ).first()

        if recent_vibe and recent_vibe.user_audio_vibe:
            vibe_result = recent_vibe.user_audio_vibe
            if recent_vibe.user_lyrics_vibe:
                vibe_result += " " + recent_vibe.user_lyrics_vibe
            description = recent_vibe.description
            response_data = {
                "status": "SUCCESS",
                "result": vibe_result,
                "description": description,
            }
            return JsonResponse(response_data)
        else:
            return JsonResponse({"status": "PENDING"})

    else:
        emotions.append("Relaxed")

    if loudness > -5:  # -5 dB is taken as a generic "loud" threshold
        emotions.append("Intense")

    return emotions


""" def normalize(vector):
    # Used for testing only for now
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude


def find_closest_emotion(final_vibe, model):
    emotion_words = [
        "Happy", "Sad", "Angry", "Joyful", "Depressed", "Anxious", "Content",
        "Excited", "Bored", "Nostalgic", "Frustrated", "Hopeful", "Afraid",
        "Confident", "Jealous", "Grateful", "Lonely", "Overwhelmed", "Relaxed",
        "Amused", "Curious", "Ashamed", "Sympathetic", "Disappointed", "Proud",
        "Guilty", "Enthusiastic", "Empathetic", "Shocked", "Calm", "Inspired",
        "Disgusted", "Indifferent", "Romantic", "Surprised", "Tense", "Euphoric",
        "Melancholic", "Restless", "Serene", "Sensual"
    ]
    max_similarity = -1
    closest_emotion = None
    for word in emotion_words:
        word_vec = get_vector(word, model)
        similarity = cosine_similarity(final_vibe, word_vec)
        if similarity > max_similarity:
            max_similarity = similarity
            closest_emotion = word
    return closest_emotion


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) """


def spacy_vectorize(vibe, constrain):
    vibe_string = " ".join(vibe)
    in_vocab_vibes = [token.text for token in nlp(vibe_string) if not token.is_oov]
    in_vocab_tokens = nlp(" ".join(in_vocab_vibes))

    max_similarity = -1
    closest_emotion = None

    for word in constrain:
        similarity = nlp(word).similarity(in_vocab_tokens)
        if similarity > max_similarity:
            max_similarity = similarity
            closest_emotion = word

    return closest_emotion
