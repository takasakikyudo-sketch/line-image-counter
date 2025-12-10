import cv2
import numpy as np

def analyze_symbol(image_path):
    """
    画像中の記号（◯、×、／、＼、◎、⊗）を判定して返す
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    _, th = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

    # 輪郭抽出
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "判定不可"

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    # 楕円フィット
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        (x, y), (MA, ma), angle = ellipse
        roundness = MA / ma
    else:
        roundness = 0

    # 直線検出
    lines = cv2.HoughLinesP(th, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
    line_count = 0
    if lines is not None:
        line_count = len(lines)

    # ◯ / 二重丸 / × / 斜線 の判定ロジック
    if roundness > 0.75 and line_count == 0:
        return "◯"

    if roundness > 0.75 and line_count == 2:
        return "◎（二重丸）"

    if line_count == 1:
        # 傾きを調べる
        x1, y1, x2, y2 = lines[0][0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if slope > 0:
            return "／（右上がり）"
        else:
            return "＼（左上がり）"

    if line_count >= 2:
        return "×"

    return "不明"
