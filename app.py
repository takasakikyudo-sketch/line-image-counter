# 完成版 app.py
# LINE Bot(v3) + 緑4点検出 + 5x5分割 + サーバー保存
# Python 3.13 / Render / Flask / OpenCV 対応

import os
import cv2
import numpy as np
from flask import Flask, request, abort, send_from_directory

from linebot.v3 import Configuration
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import MessagingApi
from linebot.v3.webhooks import MessageEvent, ImageMessageContent

# =========================
# LINE Bot 設定
# =========================
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

configuration = Configuration(
    access_token=CHANNEL_ACCESS_TOKEN
)

handler = WebhookHandler(CHANNEL_SECRET)
messaging_api = MessagingApi(configuration)

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

@app.route("/images/<filename>")
def images(filename):
    return send_from_directory("static", filename)

# =========================
# 画像処理関数
# =========================
def find_green_points(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([40, 80, 80])
    upper = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))

    if len(points) != 4:
        raise ValueError("緑点が4つ検出されていません")

    return points


def sort_rectangle_points(pts):
    pts = np.array(pts)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def split_rectangle(img, rect_pts, rows=5, cols=5, cell_size=64, margin=4):
    rect_pts = sort_rectangle_points(rect_pts)

    w = cols * cell_size
    h = rows * cell_size

    dst = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(img, M, (w, h))

    cells = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_size + margin
            y = r * cell_size + margin
            size = cell_size - margin * 2
            crop = warped[y:y+size, x:x+size]
            crop = cv2.resize(crop, (64, 64))
            cells.append(crop)

    return cells

# =========================
# LINE 画像受信処理
# =========================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    os.makedirs("static", exist_ok=True)

    message_id = event.message.id
    content = messaging_api.get_message_content(message_id)

    input_path = "static/input.jpg"
    with open(input_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    img = cv2.imread(input_path)

    green_pts = find_green_points(img)
    cells = split_rectangle(img, green_pts, rows=5, cols=5)

    # 0行0列目（index 0）を保存
    cv2.imwrite("static/reply.png", cells[0])

# =========================
# ローカル起動
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
