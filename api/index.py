from travel_app.asgi import application  # Adjust if your main folder name is different

async def handler(scope, receive, send):
    await application(scope, receive, send)
