from django.urls import path

from . import views

app_name = "user_profile"
urlpatterns = [
    path("", views.check_and_store_profile, name="profile_page"),
    path("edit", views.edit, name="edit"),
    path("update", views.update, name="update"),
    path("search", views.search, name="search"),
    path("changeTrack", views.changeTrack, name="changeTrack"),
]
