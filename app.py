from flask import Flask, request
from image_analyzer import analyze_symbol

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from Render!"

from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="画像を受信しました！")
    )



@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # 画像の内容を取得
    message_content = line_bot_api.get_message_content(event.message.id)

    # 一時ファイルとして保存
    temp_path = f"/tmp/{event.message.id}.jpg"
    with open(temp_path, "wb") as f:
        for chunk in message_content.iter_content():
            f.write(chunk)

    # 画像の記号を判定
    result = analyze_symbol(temp_path)

    # LINEへ返信
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"判定結果：{result}")
    )
