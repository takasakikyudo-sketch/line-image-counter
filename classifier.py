import cv2
import numpy as np

def classify_symbol(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 円検出
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 30,
        param1=50, param2=30,
        minRadius=10, maxRadius=50
    )

    # 線検出
    lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
    line_cnt = len(lines) if lines is not None else 0
    has_circle = circles is not None

    # 二重丸
    if circles is not None and len(circles[0]) >= 2:
        return "double_circle"

    # 丸＋斜線
    if has_circle and line_cnt == 1:
        return "circle_slash"

    # バツ（角度が違う2直線）
    if line_cnt >= 2:
        return "cross"

    # 斜線
    if line_cnt == 1:
        return "slash"

    # 丸
    if has_circle:
        return "circle"

    return "none"
