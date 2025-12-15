import os
import uuid
import threading
import cv2
import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
from flask import Flask, request, abort

from linebot.v3 import WebhookHandler
from linebot.v3.webhooks import (
    MessageEvent,
    ImageMessageContent,
    TextMessageContent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage
)
from tensorflow.keras.models import load_model

MODEL = load_model("model64.keras")

CLASS_NAMES = [
    "circle",
    "cross",
    "lcrosscircle",
    "rcrosscircle"
]

SCORE_MAP = {
    "circle": 2,
    "cross": 0,
    "lcrosscircle": 1,
    "rcrosscircle": 1
}

# ========================
# 環境変数
# ========================
CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]

# ========================
# 設定値
# ========================
ROWS = 4
COLS = 4
CELL_SIZE = 64

STATIC_DIR = "static/results"
os.makedirs(STATIC_DIR, exist_ok=True)

# 二重処理防止
PROCESSED_IDS = set()

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
# 画像処理系
# ========================
def find_green_points(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([40, 80, 80]),
        np.array([80, 255, 255])
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    points = []
    for c in contours:
        if cv2.contourArea(c) > 100:
            M = cv2.moments(c)
            if M["m00"] != 0:
                points.append([
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])
                ])

    if len(points) != 4:
        raise ValueError("緑点が4つ見つかりません")

    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def denoise_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.fastNlMeansDenoisingColored(
        img, None, 10, 10, 7, 21
    )


def warp_rectangle(img, pts):
    w = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))
    h = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))


def split_cells(img):
    h, w = img.shape[:2]
    cell_h = h / ROWS
    cell_w = w / COLS

    cells = [[None]*COLS for _ in range(ROWS)]

    for r in range(ROWS):
        for c in range(COLS):
            y1 = int(r * cell_h)
            y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w)
            x2 = int((c + 1) * cell_w)

            cell = img[y1:y2, x1:x2]
            cell = cv2.resize(cell, (64, 64))

            # BGR → HSV
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

            # 色マスク（青・緑・赤）
            mask_blue  = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            mask_green = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
            mask_red1  = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
            mask_red2  = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))

            mask = mask_blue | mask_green | mask_red1 | mask_red2
            cell[mask > 0] = [255, 255, 255]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # 正規化
            cell = cell.astype("float32") / 255.0

            # モデル入力形状へ
            cell = cell.reshape(1, 64, 64, 1)

            cells[r][c] = cell

    return cells

def calc_column_scores(cells_2d):
    """
    cells_2d: [row][col] -> (1,64,64,1)
    """
    all_cells = []
    index_map = []  # (row, col)

    for row in range(ROWS):
        for col in range(COLS):
            all_cells.append(cells_2d[row][col])
            index_map.append((row, col))

    # (N,64,64,1)
    batch = np.vstack(all_cells)

    print("predict batch start", flush=True)
    preds = MODEL.predict(batch, verbose=0)
    print("predict batch end", flush=True)

    # クラス → 点数
    scores = {}
    for (row, col), pred in zip(index_map, preds):
        cls = CLASS_NAMES[pred.argmax()]
        scores[(row, col)] = SCORE_MAP[cls]

    # 列ごとに集計（右→左）
    results = []
    for col in reversed(range(COLS)):
        col_score = sum(scores[(row, col)] for row in range(ROWS))
        results.append(col_score / (2 * ROWS))

    return results
# ========================
# 非同期処理本体
# ========================
def process_image_async(image_bytes, reply_token):
    try:

        
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        pts = find_green_points(img)
        img_denoised = denoise_image(img)
        print("warped start")
        warped = warp_rectangle(img_denoised, pts)
        print("split_cells start")
        cells = split_cells(warped)
        print("calc_column_scores start")
        
        scores = calc_column_scores(cells)
        print(scores)
        # 表示用テキスト
        text = "スコア\n"
        for i, s in enumerate(scores, 1):
            text += f"{i}列目: {s:.2f}\n"

        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text)]
            )
        )

    except Exception as e:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=f"エラー: {e}")]
            )
        )


# ========================
# Webhook
# ========================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    handler.handle(body, signature)

    # 👈 ここが超重要：即返す
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    global ROWS, COLS
    try:
        ROWS, COLS = map(int, event.message.text.split())
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text=f"{ROWS}行 × {COLS}列 に設定しました")
                ]
            )
        )
    except:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text="行数 列数 を半角スペースで送ってください")
                ]
            )
        )


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    message_id = event.message.id

    # 二重処理防止
    if message_id in PROCESSED_IDS:
        return
    PROCESSED_IDS.add(message_id)

    image_bytes = messaging_blob_api.get_message_content(message_id)

    # 非同期実行
    threading.Thread(
        target=process_image_async,
        args=(image_bytes, event.reply_token),
        daemon=True
    ).start()


# ========================
# 起動
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

