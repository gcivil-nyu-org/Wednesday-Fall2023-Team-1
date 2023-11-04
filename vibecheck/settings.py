"""
Django settings for vibecheck project.

Generated by 'django-admin startproject' using Django 2.2.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""

import os

# from supabase import create_client, Client
# SUPABASE_URL = 'https://rndwvilajbirenkbpccu.supabase.co'
# SUPABASE_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJuZHd2aWxhamJpcmVua2JwY2N1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTUzNDg3NjksImV4cCI6MjAxMDkyNDc2OX0.pDRxht8VqaEd3B9Pwrl3QVjPT8HgRekkeV533qgXY3s'
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "0%mn4bp$ofc*%rt)vo)1s!0=@e#$@ni^sa$okg2e1aw59j*skz"
SECRET_KEY = "0%mn4bp$ofc*%rt)vo)1s!0=@e#$@ni^sa$okg2e1aw59j*skz"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = [
    "vcheck-app-env.eba-eai754zm.us-west-2.elasticbeanstalk.com",
    "vcheck-env-1014.eba-megnbk6g.us-west-2.elasticbeanstalk.com",
    "127.0.0.1",
]


# Application definition

INSTALLED_APPS = [
    "login",
    # "dashboard",
    'dashboard.apps.DashboardConfig',
    "user_profile",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_extensions",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "vibecheck.urls"
ROOT_URLCONF = "vibecheck.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "vibecheck.wsgi.application"
WSGI_APPLICATION = "vibecheck.wsgi.application"


# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "postgres",
        "USER": "postgres",
        "PASSWORD": "team1vibecheck",
        "HOST": "db.rndwvilajbirenkbpccu.supabase.co",
        "PORT": "5432",
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "postgres",
        "USER": "postgres",
        "PASSWORD": "team1vibecheck",
        "HOST": "db.rndwvilajbirenkbpccu.supabase.co",
        "PORT": "5432",
    }
}


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = "en-us"
LANGUAGE_CODE = "en-us"

TIME_ZONE = "America/New_York"
TIME_ZONE = "America/New_York"

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/

STATIC_URL = "/static/"

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "login/static"),
    os.path.join(BASE_DIR, "dashboard/static"),
    os.path.join(BASE_DIR, "user_profile/static"),
]

STATIC_ROOT = os.path.join(BASE_DIR, "static")

SESSION_ENGINE = (
    "django.contrib.sessions.backends.cache"  # Use database-backed sessions
)
SESSION_CACHE_ALIAS = (
    "default"  # Use the default cache (can be adjusted based on your caching settings)
)
