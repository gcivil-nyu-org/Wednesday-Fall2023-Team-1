from django.urls import path

from . import views

app_name = "vibematch"

urlpatterns = [
    path("", views.vibe_match, name="vibe_match"),
    path("store_location/", views.store_location, name="store_location"),
    path(
        "check_location_stored/",
        views.check_location_stored,
        name="check_location_stored",
    ),
]
