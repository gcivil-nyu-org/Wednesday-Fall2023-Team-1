# myproject/routing.py

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from chatroom import consumers

websocket_urlpatterns = [
    path("ws/chatroom/", consumers.GlobalChatConsumer.as_asgi()),
]

application = ProtocolTypeRouter(
    {
        # (http->django views is added by default)
        "websocket": AuthMiddlewareStack(URLRouter(websocket_urlpatterns)),
    }
)
