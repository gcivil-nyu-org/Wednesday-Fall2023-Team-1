from django.shortcuts import render, redirect

# from django.contrib.auth.decorators import login_required
from dotenv import load_dotenv
import spotipy
from utils import get_spotify_token
from django.utils import timezone
from .models import User, Vibe

# import os

# Load variables from .env
load_dotenv()

# Create your views here.

# url: str = os.getenv("SUPABASE_URL")
# key: str = os.getenv("SUPABASE_KEY")
# supabase: Client = create_client(url, key)


def check_and_store_profile(request):
    token_info = get_spotify_token(request)

    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])

        time = timezone.now()

        user_info = sp.current_user()
        user_id = user_info["id"]
        user_exists = User.objects.filter(user_id=user_id).first()

        if not user_exists:
            user = User(
                user_id=user_id,
                username=user_info["display_name"],
                total_followers=user_info["followers"]["total"],
                profile_image_url=(
                    user_info["images"][0]["url"]
                    if ("images" in user_info and user_info["images"])
                    else None
                ),
                user_country=user_info["country"],
                user_last_login=time,
            )
            user.save()
        else:
            user = user_exists

            if user.username != user_info["display_name"]:
                user.username = user_info["display_name"]
            if user.total_followers != user_info["followers"]["total"]:
                user.total_followers = user_info["followers"]["total"]
            new_profile_image_url = (
                user_info["images"][0]["url"]
                if ("images" in user_info and user_info["images"])
                else None
            )
            if user.profile_image_url != new_profile_image_url:
                user.profile_image_url = new_profile_image_url
            if user.user_country != user_info["country"]:
                user.user_country = user_info["country"]

            user.user_last_login = time
            user.save()
        
        # Get user's most recent vibe, order by descending time
        recent_vibe = Vibe.objects.filter(user_id=user_id).order_by('-vibe_time').first()

        context = {
            "user": user,
            "vibe": recent_vibe,
            "default_image_path": "user_profile/blank_user_profile_image.jpeg",
        }
        return render(request, "user_profile/user_profile.html", context)
    else:
        # No token, redirect to login again
        # ERROR MESSAGE HERE?
        return redirect("login:index")


def update_user_profile(request, user_id):
    user = User.objects.filter(user_id=user_id).first()
    context = {"user": user}
    return render(request, "user_profile/update_profile.html", context)


# Updates the profile return to User_profile Page
def update(request, user_id):
    user = User.objects.filter(user_id=user_id).first()

    if request.method == "POST":
        print("Data is changed")
        bio = request.POST.get("user_bio")
        city = request.POST.get("user_city")
        if city:
            user.user_city = city
        if bio != user.user_bio:
            user.user_bio = bio
        user.save()

    return redirect("user_profile:profile_page")
