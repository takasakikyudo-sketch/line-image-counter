# app.py
import os
import base64
import io
import json
import requests
from flask import Flask, request, jsonify
import cv2
import numpy as np
from google.cloud import vision

app = Flask(__name__)

# 環境変数
GAS_RESULT_URL = os.environ.get('GAS_RESULT_URL')  # 例: https://script.google.com/macros/s/XXX/exec?action=result
VISION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')  # Render secret path
PROCESS_AUTH_KEY = os.environ.get('PROCESS_AUTH_KEY')  # 任意の簡易認証トークン

# Vision client
vision_client = vision.ImageAnnotatorClient()

# テンプレート読み込み（templates フォルダに置く）
TEMPLATES = {}
def load_templates():
    names = ['double_circle','cross','slash_left','slash_right','circle_slash_left','circle_slash_right']
    for n in names:
        p = os.path.join('templates', f'{n}.png')
        if os.path.exists(p):
            TEMPLATES[n] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
load_templates()

def decode_base64_image(b64):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def detect_cells_simple(img):
    # 簡易: グレースケール→二値化→外接矩形抽出→ソートしてセルリスト返却
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 1000: continue
        rects.append((x,y,w,h))
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    # グリッドの行列化は簡易化。実運用は透視変換や行列復元を推奨
    cells = [img[y:y+h, x:x+w] for x,y,w,h in rects]
    return cells, rects

def match_template(cell_gray):
    best_name = 'none'
    best_score = 0.0
    for name, templ in TEMPLATES.items():
        if templ is None: continue
        # テンプレートはセルより小さいことを想定。リサイズや正規化が必要な場合あり
        try:
            res = cv2.matchTemplate(cell_gray, templ, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_name = name
        except Exception:
            continue
    # 閾値は調整
    return best_name if best_score > 0.55 else 'none'

def ocr_with_vision(img):
    _, buf = cv2.imencode('.jpg', img)
    content = buf.tobytes()
    image = vision.Image(content=content)
    resp = vision_client.text_detection(image=image)
    if resp.text_annotations:
        return resp.text_annotations[0].description.strip()
    return ''

@app.route('/process', methods=['POST'])
def process():
    # 簡易認証ヘッダチェック（任意）
    auth = request.headers.get('X-Process-Key')
    if PROCESS_AUTH_KEY and auth != PROCESS_AUTH_KEY:
        return jsonify({'error':'unauthorized'}), 401

    data = request.get_json(force=True)
    file_b64 = data.get('fileBase64')
    file_name = data.get('fileName', 'upload.jpg')
    if not file_b64:
        return jsonify({'error':'no file'}), 400

    img = decode_base64_image(file_b64)
    if img is None:
        return jsonify({'error':'invalid image'}), 400

    # セル検出
    cells, rects = detect_cells_simple(img)
    # 仮に最下段の名前セルは最後の数個と仮定
    # 実運用では行列復元して最下段を特定する
    counts = {}
    # 初期化はOCRで得た名前で行うため一旦集計用リストを作る
    symbol_results = []
    for i, cell in enumerate(cells):
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        sym = match_template(gray)
        symbol_results.append(sym)

    # 最下段OCR（ここでは最後の N セルを名前セルと仮定）
    N = 5  # 実際の列数に合わせて調整
    names = []
    if len(cells) >= N:
        name_cells = cells[-N:]
        for nc in name_cells:
            txt = ocr_with_vision(nc)
            names.append(txt)
    else:
        names = ['unknown']

    # 仮の集計ロジック: names と symbol_results を対応付ける処理は実運用で行う
    # ここでは names を1つずつ取り、ランダムにカウントを作るダミー
    result_counts = []
    for name in names:
        result_counts.append({
            'name': name,
            'double_circle': int(symbol_results.count('double_circle')),
            'cross': int(symbol_results.count('cross')),
            'slashL': int(symbol_results.count('slash_left')),
            'slashR': int(symbol_results.count('slash_right'))
        })

    # GAS に結果を送信
    if GAS_RESULT_URL:
        try:
            headers = {'Content-Type':'application/json'}
            # 認証が必要ならヘッダに追加
            payload = {'counts': result_counts, 'sourceFileName': file_name}
            requests.post(GAS_RESULT_URL, json=payload, headers=headers, timeout=10)
        except Exception as e:
            # ログに残すが解析は成功として返す
            print('Failed to post to GAS', e)

    return jsonify({'status':'ok','counts': result_counts})
