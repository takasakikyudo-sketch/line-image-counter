import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, request

from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

from classifier import classify_symbol
from score import calculate_score

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(LINE_CHANNEL_SECRET)


# ==========
# 記号解析
# ==========
def split_and_classify(img):
    h, w = img.shape[:2]

    ROWS = 10
    COLS = 10
    cell_h = h // ROWS
    cell_w = w // COLS

    symbols = []

    for r in range(ROWS):
        for c in range(COLS):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell = img[y1:y2, x1:x2]
            s = classify_symbol(cell)
            symbols.append(s)

    return symbols


# ==========
# 名前OCR
# ==========
def extract_name(img):
    h, w = img.shape[:2]
    bottom_area = img[int(h * 0.80):h, :]
    text = pytesseract.image_to_string(bottom_area, lang="jpn")
    return text.strip()


# ==========
# Webhook
# ==========
@app.route("/callback", methods=['POST'])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers["X-Line-Signature"]
    events = parser.parse(body, signature)

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, ImageMessage):
            handle_image(event)

    return "OK"


# ==========
# 画像処理 → 結果返却
# ==========
def handle_image(event):

    # 画像データ取得
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)
    img_data = content.content

    # OpenCV 読み込み
    img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 記号判定
    symbols = split_and_classify(img)

    # スコア計算
    score, total = calculate_score(symbols)

    # 名前抽出（今回は失敗もあるので仮名でもOK）
    # name = extract_name(img)
    name = "a"

    # 返す内容
    reply_text = (
        f"名前：{name}\n"
        f"スコア：{score}\n"
        f"合計：{total}\n"
    )

    # LINE 返信
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@app.route("/")
def home():
    return "LINE Bot is running"


if __name__ == "__main__":
    app.run(port=8080)
