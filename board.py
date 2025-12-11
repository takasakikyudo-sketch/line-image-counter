import cv2
import numpy as np

def extract_board(image, marker_min_area=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > marker_min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            markers.append((x, y, w, h))

    if len(markers) != 4:
        raise ValueError("四隅のマーカーが4つ見つかりません")

    pts = []
    for (x, y, w, h) in markers:
        pts.append([x + w // 2, y + h // 2])
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    src = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")

    board_size = 1000
    dst = np.array([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    board = cv2.warpPerspective(image, M, (board_size, board_size))

    return board


def split_board(board, ROWS, COLS):
    h, w = board.shape[:2]
    cell_h = h // ROWS
    cell_w = w // COLS

    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell = board[y1:y2, x1:x2]
            cells.append(cell)
    return cells
