import os
import cv2
import numpy as np
import pytesseract
import json
import base64

from flask import Flask, request

from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

import gspread
from google.oauth2.service_account import Credentials

from classifier import classify_symbol
from score import calculate_score

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
SPREADSHEET_NAME = os.environ["SPREADSHEET_NAME"]
SERVICE_ACCOUNT_BASE64 = os.environ["SERVICE_ACCOUNT_JSON"]  # ←追加

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(LINE_CHANNEL_SECRET)


# ★ service_account.json を環境変数から復元する関数
def load_service_account():
    if not SERVICE_ACCOUNT_BASE64:
        raise Exception("SERVICE_ACCOUNT_JSON env var not set")

    # Base64 → JSON 文字列 → dict
    decoded = base64.b64decode(SERVICE_ACCOUNT_BASE64).decode("utf-8")
    return json.loads(decoded)


# Google Sheets への書き込み
def write_to_sheet(name, score, total):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    service_info = load_service_account()
    creds = Credentials.from_service_account_info(service_info, scopes=scopes)

    gc = gspread.authorize(creds)
    sh = gc.open(SPREADSHEET_NAME)
    ws = sh.sheet1

    ws.append_row([name, score, total])


# 画像 → 記号配列へ
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


# 最下部の名前を OCR
def extract_name(img):
    h, w = img.shape[:2]
    bottom_area = img[int(h * 0.80):h, :]
    text = pytesseract.image_to_string(bottom_area, lang="jpn")
    return text.strip()


@app.route("/callback", methods=['POST'])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers["X-Line-Signature"]
    events = parser.parse(body, signature)

    for event in events:
        if isinstance(event, MessageEvent) and isinstance(event.message, ImageMessage):
            handle_image(event)

    return "OK"


def handle_image(event):

    # 画像取得
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)
    img_data = content.content

    # OpenCVで読み込み
    img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 記号解析
    symbols = split_and_classify(img)

    # スコア計算
    score, total = calculate_score(symbols)

    # 名前抽出（今は固定）
    name = "a"

    # Google Sheets へ保存
    write_to_sheet(name, score, total)

    # LINEに返信
    reply_text = f"{name}\nスコア：{score}\n合計：{total}\nGoogle Sheets に記録しました！"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@app.route("/")
def home():
    return "LINE Bot is running on Render"


if __name__ == "__main__":
    app.run(port=8080)
