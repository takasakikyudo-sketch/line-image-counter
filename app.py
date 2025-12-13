import os
import cv2
from flask import Flask, request, abort, send_from_directory

from linebot.v3 import Configuration
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import MessagingApi
from linebot.v3.webhooks import MessageEvent, ImageMessageContent

# =====================
# LINE設定
# =====================
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(
    access_token=CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(CHANNEL_SECRET)
messaging_api = MessagingApi(configuration)

# =====================
# Flask
# =====================
app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

@app.route("/images/<filename>")
def images(filename):
    return send_from_directory("static", filename)

# =====================
# 画像受信
# =====================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    message_id = event.message.id
    content = messaging_api.get_message_content(message_id)

    os.makedirs("static", exist_ok=True)
    with open("static/input.jpg", "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    # ここで画像処理（省略）

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
