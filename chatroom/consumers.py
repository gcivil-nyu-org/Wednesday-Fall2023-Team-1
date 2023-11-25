import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import ChatMessage, RoomModel
from .views import get_user_exist


class GlobalChatConsumer(AsyncWebsocketConsumer):
    roomID = "global_room"
    sender = None
    message = ""

    async def connect(self):
        await self.channel_layer.group_add(self.roomID, self.channel_name)
        await self.accept()

        self.sender = await self.get_user()

        # Retrieve and send existing messages for the room
        # messages = await self.retrieve_messages()

        messages = await self.retrieve_room_messages()
        # messages_cycle = cycle(messages)

        # Use async for directly on the coroutine result
        async for message in messages:
            sender_username = await database_sync_to_async(
                lambda: message.sender.username
            )()
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "chat_message",
                        "message": message.content,
                        "sender": sender_username,
                    }
                )
            )

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.roomID, self.channel_name)

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)

        message_type = text_data_json.get("type")
        self.message = text_data_json.get("message")

        print("Message type", message_type)
        print("Message -> ", self.message)

        if message_type == "join_room":
            self.roomID = text_data_json.get("roomID")
            await self.channel_layer.group_add(self.roomID, self.channel_name)
            await self.send(
                text_data=json.dumps(
                    {"type": "info", "message": f"You have joined room {self.roomID}"}
                )
            )

        elif message_type == "chat_message":
            await self.save_chat_db()
            # Send message to room group
            await self.channel_layer.group_send(
                self.roomID,
                {
                    "type": "chat_message",
                    "message": self.message,
                    "sender": self.sender.username,
                },
            )

    async def chat_message(self, event):
        message = event["message"]
        sender = event["sender"]

        # Send the message to the WebSocket
        await self.send(
            text_data=json.dumps(
                {
                    "type": "chat_message",
                    "message": message,
                    "sender": sender,
                }
            )
        )

    @database_sync_to_async
    def get_user(self):
        return get_user_exist()

    @database_sync_to_async
    def retrieve_room_messages(self):
        room_messages = ChatMessage.objects.filter(room=self.roomID)
        return room_messages

    @database_sync_to_async
    def save_chat_db(self):
        print("Doing DB work")
        message = ChatMessage.objects.create(
            sender=self.sender,
            room=RoomModel.objects.get(roomID=self.roomID),
            content=self.message,
        )
        message.save()
