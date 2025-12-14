import tensorflow as tf
import numpy as np
from PIL import Image

# ===== 設定 =====
MODEL_PATH = "model64.keras"
CLASS_NAME_FILE = "class_names.txt"
IMAGE_PATH = "input.png"   # 判別したい画像

# ===== モデル読み込み =====
model = tf.keras.models.load_model(MODEL_PATH)

# ===== クラス名読み込み =====
with open(CLASS_NAME_FILE, encoding="utf-8") as f:
    class_names = [line.strip() for line in f]

# ===== 画像前処理（学習時と完全一致）=====
img = Image.open(IMAGE_PATH).convert("L").resize((64, 64))
img = np.array(img) / 255.0
img = img.reshape(1, 64, 64, 1)

# ===== 推論 =====
pred = model.predict(img)
idx = pred.argmax()
label = class_names[idx]
confidence = pred[0][idx]

print("判定結果:", label)
print("信頼度:", confidence)

