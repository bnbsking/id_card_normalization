"""
證件照片校正工具
輸入: 已裁切但可能歪斜/透視變形的證件圖片
流程: 1. 偵測證件四個角落座標
      2. 用 Homography (透視變換) 將其校正為矩形
輸出: 校正後的矩形圖片

用法:
    python id_card_rectify.py input.jpg output.jpg
"""
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    將四個點依照 [左上, 右上, 右下, 左下] 的順序排列。
    利用座標和(x+y)與差(x-y)來判斷位置:
      左上 -> 和最小
      右下 -> 和最大
      右上 -> 差最小 (x-y)
      左下 -> 差最大 (x-y)
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def _largest_quad_from_contour(contour, min_area_ratio, total_area):
    """
    嘗試把輪廓近似成四邊形；若 approxPolyDP 抓不到剛好 4 點，
    改用 minAreaRect 取代（適用於圓角證件，approx 容易抓成 5~8 點）。
    """
    if cv2.contourArea(contour) < min_area_ratio * total_area:
        return None

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype("float32")

    # approx 點數不是 4（常見於圓角、雜訊），退而求其次用最小外接矩形
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype("float32")


def _find_corners_by_threshold(resized, total_area):
    """
    方法一：亮度二值化 (Otsu)。
    適合「證件 vs 背景對比明顯」的情況（例如深色桌面 + 淺色證件），
    比 Canny 更不容易被背景紋理、桌面花紋干擾。
    """
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 若證件比背景暗（極少數情況），Otsu 結果可能反相，兩種都試
    for candidate in (thresh, cv2.bitwise_not(thresh)):
        # 閉運算補洞、去除小雜訊
        closed = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        quad = _largest_quad_from_contour(c, 0.3, total_area)
        if quad is not None:
            return quad
    return None


def _find_corners_by_canny(resized, total_area):
    """
    方法二：Canny 邊緣偵測（備援）。
    適合背景與證件顏色接近、亮度對比不明顯，但邊界輪廓清楚的情況。
    """
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=2)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        quad = _largest_quad_from_contour(c, 0.2, total_area)
        if quad is not None:
            return quad
    return None


def find_document_corners(image: np.ndarray) -> np.ndarray:
    """
    在圖片中找出證件四個角座標。
    先嘗試「亮度二值化法」，抓不到再退回「Canny 邊緣法」。
    回傳: shape (4,2) 的 float32 array，若兩種方法都找不到則回傳 None
    """
    orig_h, orig_w = image.shape[:2]

    # 縮小圖片加速處理，之後再把座標放大回原尺寸
    ratio = 800.0 / orig_w if orig_w > 800 else 1.0
    resized = cv2.resize(image, (int(orig_w * ratio), int(orig_h * ratio)))
    total_area = resized.shape[0] * resized.shape[1]

    pts = _find_corners_by_threshold(resized, total_area)
    if pts is None:
        pts = _find_corners_by_canny(resized, total_area)

    if pts is None:
        return None

    pts /= ratio  # 換算回原始圖片座標
    return order_points(pts)


def warp_by_homography(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    利用四個角座標與 Homography，將證件區域轉正為矩形。
    輸出寬高依照偵測到的邊長自動計算，維持證件比例。
    """
    (tl, tr, br, bl) = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # 目標矩形四個角座標 (輸出圖片座標系)
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 計算 Homography 矩陣並做透視變換
    H, _ = cv2.findHomography(corners, dst, method=0)
    warped = cv2.warpPerspective(image, H, (max_width, max_height))
    return warped


def rectify_id_card(input_path: str, output_path: str):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片: {input_path}")

    corners = find_document_corners(image)

    if corners is None:
        print("警告: 自動偵測四角失敗，改用整張圖片邊界作為四角。")
        h, w = image.shape[:2]
        corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
        )
    else:
        print("偵測到四個角座標 (左上, 右上, 右下, 左下):")
        for name, pt in zip(["左上", "右上", "右下", "左下"], corners):
            print(f"  {name}: ({pt[0]:.1f}, {pt[1]:.1f})")

    warped = warp_by_homography(image, corners)
    cv2.imwrite(output_path, warped)
    print(f"已輸出校正後圖片: {output_path}")

    # 額外輸出一張標示四角的除錯圖，方便確認偵測是否正確
    debug_img = image.copy()
    for pt in corners:
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
    cv2.polylines(
        debug_img, [corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2
    )
    debug_path = output_path.rsplit(".", 1)[0] + "_debug.jpg"
    cv2.imwrite(debug_path, debug_img)
    print(f"已輸出四角標示除錯圖: {debug_path}")


if __name__ == "__main__":
    rectify_id_card(
        "/app/tests/data/raw.jpg",
        "/app/tests/data/homo.jpg"
    )
