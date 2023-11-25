from django.shortcuts import render, redirect
from django.contrib import messages
from utils import sp_oauth

# Create your views here.


def index(request):
    return render(request, "login/index.html")


def authenticate_spotify(request):
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


def callback(request):
    auth_code = request.GET.get("code", None)
    if auth_code:
        token_info = sp_oauth.get_access_token(auth_code, check_cache=False)
        request.session["token_info"] = token_info
        return redirect("dashboard:index")
    else:
        debug_info = f"Request: {request}"
        messages.error(
            request, f"Login failed, please try again later. Debug info: {debug_info}"
        )
        return redirect("login:index")
