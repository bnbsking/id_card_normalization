"""
證件（台胞證等）自動偵測 + 透視校正
流程：
1. 讀圖 + resize
2. 邊緣偵測 (Canny) + 形態學 closing 補齊斷裂邊界
3. 找輪廓，用「面積比例 + 長寬比」過濾出證件候選
4. 用 minAreaRect 取得四個角點，clip 到圖片範圍內
5. 透視變換 (warpPerspective) 把證件攤平輸出
"""

import cv2
import numpy as np


# ------------------------- 可調參數 -------------------------
IMG_PATH = "/dataset/0914/台胞證/20569495559749.jpg"
TARGET_WIDTH = 1000          # resize 後的寬度
CANNY_LOW = 50
CANNY_HIGH = 150
CLOSE_KERNEL_SIZE = 15       # closing 用的 kernel 大小，越大越能補齊大缺口
CLOSE_ITERATIONS = 3
AREA_MIN_RATIO = 0.10        # 候選輪廓面積下限（佔整張圖比例）
AREA_MAX_RATIO = 0.85        # 候選輪廓面積上限（避免抓到整張圖）
ASPECT_RATIO_MIN = 1.3       # 證件長寬比容許範圍
ASPECT_RATIO_MAX = 1.9
OUTPUT_WIDTH = 856           # 校正後輸出的寬（可自訂，跟長寬比對應即可）
OUTPUT_HEIGHT = 540
DEBUG_DIR = "/app/outputs/preprocess"
# -------------------------------------------------------------


def order_points(pts: np.ndarray) -> np.ndarray:
    """把四個角點排序成：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上：x+y 最小
    rect[2] = pts[np.argmax(s)]   # 右下：x+y 最大

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上：x-y 最小
    rect[3] = pts[np.argmax(diff)]  # 左下：x-y 最大
    return rect


def find_card_contour(img: np.ndarray):
    """回傳 (card_box, debug_images_dict) ; card_box 為 None 表示沒找到"""
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS
    )

    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_box = None
    best_rect_info = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < img_area * AREA_MIN_RATIO or area > img_area * AREA_MAX_RATIO:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect

        if rw == 0 or rh == 0:
            continue

        ratio = max(rw, rh) / min(rw, rh)
        if not (ASPECT_RATIO_MIN < ratio < ASPECT_RATIO_MAX):
            continue

        box = cv2.boxPoints(rect)
        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)

        card_box = box.astype(np.int32)
        best_rect_info = {"area": area, "ratio": ratio, "rect": rect}
        break

    debug_images = {
        "debug_gray.jpg": gray,
        "debug_blur.jpg": blur,
        "debug_edges.jpg": edges,
        "debug_closed.jpg": closed,
    }

    # 畫出前 5 大輪廓，方便排查
    vis_contours = img.copy()
    cv2.drawContours(vis_contours, contours[:5], -1, (0, 255, 0), 2)
    debug_images["debug_top5_contours.jpg"] = vis_contours

    if card_box is not None:
        vis_selected = img.copy()
        cv2.drawContours(vis_selected, [card_box], -1, (0, 0, 255), 3)
        debug_images["debug_selected.jpg"] = vis_selected
        print("找到候選證件輪廓:", best_rect_info)
        print("四個角點:\n", card_box)
    else:
        print("沒有找到符合條件的證件輪廓，請檢查 debug_top5_contours.jpg 確認雜訊狀況")

    return card_box, debug_images


def warp_card(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """用四個角點做透視變換，攤平輸出"""
    pts = order_points(box.astype("float32"))

    dst = np.array(
        [
            [0, 0],
            [OUTPUT_WIDTH - 1, 0],
            [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
            [0, OUTPUT_HEIGHT - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    return warped


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"讀不到圖片，請確認路徑：{IMG_PATH}")

    scale = TARGET_WIDTH / img.shape[1]
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    card_box, debug_images = find_card_contour(img)

    # 輸出所有 debug 圖，方便肉眼確認每一步
    for name, im in debug_images.items():
        cv2.imwrite(f"{DEBUG_DIR}/{name}", im)

    if card_box is not None:
        warped = warp_card(img, card_box)
        cv2.imwrite(f"{DEBUG_DIR}/card_warped.jpg", warped)
        print(f"已輸出校正後證件圖：{DEBUG_DIR}/card_warped.jpg")
    else:
        print("請調整 AREA_MIN_RATIO / AREA_MAX_RATIO / ASPECT_RATIO 範圍後再試一次")


if __name__ == "__main__":
    main()