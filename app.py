import os
import cv2
import numpy as np
from flask import Flask, request

from linebot import LineBotApi, WebhookParser
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage

from board import extract_board, split_board
from classifier import classify_symbol
from score import calculate_score

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(LINE_CHANNEL_SECRET)

# --- ユーザーごとの設定を保持 ---
user_grid_setting = {}   # user_id → (ROWS, COLS)


# =========================
#  LINE Webhook
# =========================
@app.route("/callback", methods=['POST'])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers["X-Line-Signature"]
    events = parser.parse(body, signature)

    for event in events:
        if isinstance(event, MessageEvent):

            # --- テキストメッセージ（行列設定）---
            if isinstance(event.message, TextMessage):
                handle_text(event)

            # --- 画像メッセージ ---
            if isinstance(event.message, ImageMessage):
                handle_image(event)

    return "OK"


# =========================
#  行列設定（例：10x12）
# =========================
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    # 書式：10x12, 10×12, 10 12 など
    import re
    m = re.match(r"(\d+)\s*[xX× ]\s*(\d+)", text)

    if m:
        rows = int(m.group(1))
        cols = int(m.group(2))
        user_grid_setting[user_id] = (rows, cols)

        reply = f"盤面サイズを設定しました： {rows} 行 × {cols} 列\n次に画像を送ってください！"
    else:
        reply = "行列の形式が正しくありません。\n例：10x12、8 8、5×5"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# =========================
# 画像処理
# =========================
def handle_image(event):
    user_id = event.source.user_id

    # 行列が設定されていない
    if user_id not in user_grid_setting:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="先に『10x10』のように行数と列数を送ってください！")
        )
        return

    ROWS, COLS = user_grid_setting[user_id]

    # --- 画像取得 ---
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)
    img_data = content.content

    img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    try:
        # --- 四隅マーカーで盤面抽出 ---
        board = extract_board(img)

        # --- 盤面分割 ---
        cells = split_board(board, ROWS, COLS)

        # --- 記号分類 ---
        symbols = [classify_symbol(c) for c in cells]

        # --- スコア計算 ---
        score, total = calculate_score(symbols)

        reply = (
            f"解析結果\n"
            f"行列：{ROWS}×{COLS}\n"
            f"スコア：{score}\n"
            f"トータル：{total}"
        )

    except Exception as e:
        reply = f"エラー: {str(e)}\n画像の四隅にマーカーがありますか？"

    # --- 返信 ---
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# =========================
# ルート
# =========================
@app.route("/")
def home():
    return "LINE Bot is running!"


if __name__ == "__main__":
    app.run(port=8080)
