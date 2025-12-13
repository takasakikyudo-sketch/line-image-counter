import os
import cv2
import numpy as np
from flask import Flask, request, abort, send_from_directory

from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import MessagingApi
from linebot.v3.webhooks import MessageEvent, ImageMessageContent


# =========================
# LINE Bot 設定
# =========================
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

handler = WebhookHandler(CHANNEL_SECRET)
messaging_api = MessagingApi(
    channel_access_token=CHANNEL_ACCESS_TOKEN
)

# =========================
# Flask
# =========================
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

# =========================
# 画像公開用
# =========================
@app.route("/images/<filename>")
def images(filename):
    return send_from_directory("static", filename)

# =========================
# LINE 画像受信処理
# =========================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    # 画像取得
    message_id = event.message.id
    content = messaging_api.get_message_content(message_id)

    os.makedirs("static", exist_ok=True)
    img_path = "static/input.jpg"

    with open(img_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    # ===== 画像処理 =====
    img = cv2.imread(img_path)

    green_pts = find_green_points(img)
    squares = split_rectangle(img, green_pts, rows=5, cols=5)

    # 0行0列目を保存
    cv2.imwrite("static/reply.png", squares[0][0])

    # ※返信しない（保存のみ）

# =========================
# 緑点検出・分割関数（前と同じ）
# =========================
# find_green_points(...)
# split_rectangle(...)
# sort_rectangle_points(...)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
