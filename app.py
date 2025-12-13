import os
from flask import Flask, request, abort

from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, ImageMessageContent
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob
)

# =========================
# 環境変数
# =========================
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise RuntimeError("LINEの環境変数が設定されていません")

# =========================
# Flask
# =========================
app = Flask(__name__, static_folder="static")

# =========================
# LINE SDK v3（正解構成）
# =========================
configuration = Configuration(
    access_token=CHANNEL_ACCESS_TOKEN
)
api_client = ApiClient(configuration)

handler = WebhookHandler(CHANNEL_SECRET)
messaging_api = MessagingApi(api_client)
messaging_api_blob = MessagingApiBlob(api_client)

# =========================
# Webhook
# =========================
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
# 画像メッセージ処理
# =========================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    os.makedirs("static/images", exist_ok=True)

    message_id = event.message.id

    # ✅ Blob API 正しい呼び方
    content = messaging_api_blob.get_message_content(message_id)
    image_bytes = content.read()

    save_path = "static/images/original.png"
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    print("画像保存完了:", save_path)

# =========================
# Render / ローカル
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
