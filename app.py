import os
from flask import Flask, request, abort

from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, ImageMessageContent
from linebot.v3.messaging import (
    Configuration,
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
# LINE SDK v3
# =========================
configuration = Configuration(
    access_token=CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(CHANNEL_SECRET)
messaging_api = MessagingApi(configuration)
messaging_api_blob = MessagingApiBlob(configuration)

# =========================
# ルート（画像一覧ページ）
# =========================
@app.route("/")
def index():
    return app.send_static_file("index.html")

# =========================
# LINE Webhook
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
    # static/images だけ作る（static自体は作らない）
    os.makedirs("static/images", exist_ok=True)

    message_id = event.message.id

    # v3 正解：Blob API で画像取得
    content = messaging_api_blob.get_message_content(message_id)
    image_bytes = content.read()

    # 保存（とりあえず1枚）
    save_path = "static/images/original.png"
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    print("画像を保存しました:", save_path)

# =========================
# ローカル起動用（Renderでは無視）
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
