from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from image_analyzer import analyze_symbol
import os

app = Flask(__name__)

# ★あなたのLINEチャネルの情報を入れてください
CHANNEL_ACCESS_TOKEN = os.environ.get("HQbj/0rUwacne8RnenycCX8RC+5bvS83k5Dj3ksI/iDiwuRAWFaVXAH6HFkZ0++nzTaAvvykPSJui75DSdOqRLCtKGa0cT6KjfIdAIo3PtNx5iYmEoZhUJLoKHC67jT1z5/q1ooQKn1y7rmRKpOKXwdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.environ.get("71abbc039ce27065bb7424aa50a0d695")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.route("/")
def index():
    return "Hello from Render!"


@app.route("/callback", methods=["POST"])
def callback():
    # LINE signature check
    signature = request.headers.get("X-Line-Signature", "")

    body = request.get_data(as_text=True)
    print("Request body:", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


# ==========================
# 画像メッセージの処理
# ==========================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # 画像取得
    message_content = line_bot_api.get_message_content(event.message.id)

    # 一時ファイルへ保存
    temp_path = f"/tmp/{event.message.id}.jpg"
    with open(temp_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # 記号解析
    result = analyze_symbol(temp_path)

    # 返信
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"判定結果：{result}")
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
