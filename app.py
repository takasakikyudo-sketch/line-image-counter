"""
import os
from flask import Flask, request, abort

from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, ImageMessageContent
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
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
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(configuration)

handler = WebhookHandler(CHANNEL_SECRET)
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

    # ✅ v3 正解：bytes が直接返る
    image_bytes = messaging_api_blob.get_message_content(message_id)

    save_path = "static/images/original.png"
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    print("画像保存成功:", save_path)

# =========================
# Render / ローカル
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
"""
import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, abort

from linebot.v3 import WebhookHandler
from linebot.v3.webhooks import MessageEvent, ImageMessageContent
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob
)

# ========================
# 設定
# ========================
CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

ROWS = 5
COLS = 5
CELL_SIZE = 64

STATIC_DIR = "static/results"
os.makedirs(STATIC_DIR, exist_ok=True)

# ========================
# Flask & LINE
# ========================
app = Flask(__name__)

handler = WebhookHandler(CHANNEL_SECRET)

config = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
api_client = ApiClient(config)
messaging_api = MessagingApi(api_client)
messaging_blob_api = MessagingApiBlob(api_client)

# ========================
# ユーティリティ
# ========================
def find_green_points(img):
    """緑4点を検出"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([40, 80, 80])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append([cx, cy])

    if len(points) != 4:
        raise ValueError("緑の点が4つ検出できません")

    pts = np.array(points, dtype=np.float32)

    # 並び替え（左上・右上・右下・左下）
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered

def denoise_image(img):
    # ガウシアンブラー（軽いノイズ除去）
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # カラー画像用ノイズ除去（かなり強力）
    img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10,        # 明るさノイズ
        hColor=10,   # 色ノイズ
        templateWindowSize=7,
        searchWindowSize=21
    )

    return img

def warp_rectangle(img, pts):
    """長方形に補正"""
    w = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))
    h = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))

    dst = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))


def split_and_save(img):
    """5×5分割して64×64で保存"""
    h, w = img.shape[:2]
    cell_h = h / ROWS
    cell_w = w / COLS

    saved_files = []

    for r in range(ROWS):
        for c in range(COLS):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = img[y1:y2, x1:x2]
            cell = cv2.resize(cell, (CELL_SIZE, CELL_SIZE))

            name = f"{uuid.uuid4()}.png"
            path = os.path.join(STATIC_DIR, name)
            cv2.imwrite(path, cell)
            saved_files.append(name)

    return saved_files


# ========================
# Webhook
# ========================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    message_id = event.message.id

    # 画像取得（bytes）
    content = messaging_blob_api.get_message_content(message_id)
    image_bytes = content

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 処理
    pts = find_green_points(img)
    img_denoised = denoise_image(img)
    warped = warp_rectangle(img, pts)
    files = split_and_save(warped)

    print("saved:", files)


# ========================
# 起動
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

