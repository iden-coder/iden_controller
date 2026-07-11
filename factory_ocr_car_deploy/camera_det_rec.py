# -*- coding: utf-8 -*-
import os
import cv2
import time
import math
import threading
import numpy as np
from collections import deque, Counter, defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from rknnlite.api import RKNNLite

try:
    import pyclipper
    from shapely.geometry import Polygon
except Exception as e:
    print("缺少依赖 pyclipper / shapely，请先执行：")
    print("pip install pyclipper shapely")
    raise e


# ============================================================
# 模型路径
# ============================================================
DET_MODEL = "./ch_ppocrv3_det.rknn"
REC_MODEL = "./factory_rec_student.rknn"

# 你现在推荐使用 480x640 det.rknn
# 如果你的 det.rknn 还是 640x640，把这里改成 640, 640
DET_INPUT_H = 480
DET_INPUT_W = 640

CAM_ID = 0
MIRROR_FIX = True

WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

REC_SHAPE = (3, 48, 320)

# ============================================================
# 官方 TextDetector / DBPostProcess 风格参数
# 和你训练裁剪脚本一致
# ============================================================
DET_DB_THRESH = 0.25
DET_DB_BOX_THRESH = 0.40
DET_DB_UNCLIP_RATIO = 1.8
DET_DB_SCORE_MODE = "fast"
DET_MAX_CANDIDATES = 1000
DET_MIN_SIZE = 3

# crop_by_box 参数，和你训练裁剪脚本一致
MIN_W = 35
MIN_H = 10
MIN_AREA = 500
PAD_X_RATIO = 0.16
PAD_Y_RATIO = 0.45

# OCR 投票
VOTE_WINDOW = 12
VOTE_NEED = 9

# OCR 字符表
CHARSET = ['blank', '食', '品', '加', '工', '车', '间', '日', '用', '电', '子', '产', '生']

LABELS = [
    "食品加工车间",
    "日用品加工车间",
    "电子产品生产车间",
]

SHORT = {
    "食品加工车间": "food",
    "日用品加工车间": "daily",
    "电子产品生产车间": "electric",
    "unknown": "unknown",
}


# ============================================================
# Web 可视化
# ============================================================
latest_vis_jpeg = None
latest_crop_jpeg = None
latest_status_text = "waiting..."
web_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global latest_vis_jpeg, latest_crop_jpeg, latest_status_text

        if self.path == "/" or self.path.startswith("/index"):
            html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Factory OCR Debug</title>
<style>
body { font-family: Arial, sans-serif; background:#111; color:#eee; }
img { max-width: 95vw; border:2px solid #555; margin:8px 0; }
pre { background:#222; padding:10px; font-size:16px; white-space:pre-wrap; }
</style>
</head>
<body>
<h1>Factory OCR Debug</h1>
<h2>det_rec_vis</h2>
<img src="/stream.mjpg">
<h2>crop_sheet</h2>
<img src="/crop.mjpg">
<h2>Status</h2>
<pre id="status">loading...</pre>
<script>
async function updateStatus(){
  try {
    let r = await fetch('/status');
    let t = await r.text();
    document.getElementById('status').textContent = t;
  } catch(e) {}
}
setInterval(updateStatus, 500);
updateStatus();
</script>
</body>
</html>
"""
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/status"):
            with web_lock:
                data = latest_status_text.encode("utf-8", errors="ignore")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/stream.mjpg"):
            self._serve_mjpeg("vis")
            return

        if self.path.startswith("/crop.mjpg"):
            self._serve_mjpeg("crop")
            return

        self.send_response(404)
        self.end_headers()

    def _serve_mjpeg(self, kind):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            with web_lock:
                jpg = latest_vis_jpeg if kind == "vis" else latest_crop_jpeg

            if jpg is None:
                time.sleep(0.05)
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.05)
            except Exception:
                break


def start_web_server():
    server = ThreadedHTTPServer((WEB_HOST, WEB_PORT), WebHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    print(f"网页可视化已开启: http://<小车IP>:{WEB_PORT}")
    return server


def update_web(vis, crop_sheet, status):
    global latest_vis_jpeg, latest_crop_jpeg, latest_status_text

    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    vis_jpg = buf.tobytes() if ok else None

    if crop_sheet is None or crop_sheet.size == 0:
        crop_sheet = np.zeros((120, 600, 3), dtype=np.uint8)
        cv2.putText(
            crop_sheet,
            "no good crop",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    ok2, buf2 = cv2.imencode(".jpg", crop_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    crop_jpg = buf2.tobytes() if ok2 else None

    with web_lock:
        if vis_jpg is not None:
            latest_vis_jpeg = vis_jpg
        if crop_jpg is not None:
            latest_crop_jpeg = crop_jpg
        latest_status_text = status


# ============================================================
# RKNN
# ============================================================
def load_rknn(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"--> Load {name}: {path}")
    rknn = RKNNLite()

    ret = rknn.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"load {name} failed: {ret}")

    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init {name} failed: {ret}")

    print(f"{name} ready")
    return rknn


def safe_infer_det(rknn, inp, name):
    """
    det.rknn 实测必须用 NHWC + data_format=['nhwc']。
    """
    try:
        return rknn.inference(inputs=[inp], data_format=["nhwc"])
    except Exception as e:
        print(f"[WARN] {name} inference failed:", e)
        return None


def safe_infer_rec(rknn, inp, name):
    """
    rec.rknn 不传 data_format。
    之前 rec 能正常识别，说明默认输入方式就是对的。
    传 data_format=['nchw'] 会在 RKNNLite 1.4.0 报 KeyError: 'nchw'。
    """
    try:
        return rknn.inference(inputs=[inp])
    except Exception as e:
        print(f"[WARN] {name} inference failed:", e)
        return None


# ============================================================
# DET 预处理
# ============================================================
def det_preprocess(frame):
    """
    关键修正：
    RKNN det 实测正确输入是 NHWC + data_format=['nhwc']。
    所以这里输出 shape = [1, 480, 640, 3]。
    颜色使用 BGR，因为 AutoDL ONNX 的 bgr_norm 是正确的。
    """
    src_h, src_w = frame.shape[:2]

    resized = cv2.resize(frame, (DET_INPUT_W, DET_INPUT_H))

    x = resized.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    x = (x - mean) / std

    # 不要 transpose！
    # NHWC: [1, 480, 640, 3]
    x = np.expand_dims(x, 0).astype(np.float32)

    ratio_h = DET_INPUT_H / float(src_h)
    ratio_w = DET_INPUT_W / float(src_w)

    shape_info = np.array([src_h, src_w, ratio_h, ratio_w], dtype=np.float32)

    return x, shape_info


def extract_score_map(det_out):
    """
    RKNN det 输出转成 [H, W] score map。
    """
    out = np.array(det_out[0])
    out = np.squeeze(out)

    if out.ndim == 3:
        if out.shape[0] == 1:
            out = out[0]
        elif out.shape[-1] == 1:
            out = out[:, :, 0]
        else:
            out = out[0]

    if out.ndim != 2:
        raise RuntimeError(f"bad det output shape: {out.shape}")

    out = out.astype(np.float32)

    # 如果是 logits，转 sigmoid；如果已经是概率，保持
    if out.max() > 1.5 or out.min() < -0.5:
        out = 1.0 / (1.0 + np.exp(-out))

    return out


# ============================================================
# 官方 DBPostProcess 逻辑移植版
# ============================================================
class DBPostProcess:
    def __init__(
        self,
        thresh=0.25,
        box_thresh=0.40,
        max_candidates=1000,
        unclip_ratio=1.8,
        min_size=3,
        score_mode="fast",
        use_dilation=False,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.score_mode = score_mode
        self.dilation_kernel = None

        if use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def __call__(self, pred, shape_list):
        """
        pred: numpy [1, 1, H, W] 或 [1, H, W]
        shape_list: numpy [1, 4], 每行为 [src_h, src_w, ratio_h, ratio_w]
        """
        if pred.ndim == 4:
            pred = pred[:, 0, :, :]
        elif pred.ndim == 3:
            pass
        elif pred.ndim == 2:
            pred = pred[np.newaxis, :, :]
        else:
            raise RuntimeError(f"bad pred shape: {pred.shape}")

        segmentation = pred > self.thresh

        boxes_batch = []

        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            src_h = int(src_h)
            src_w = int(src_w)

            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel,
                )
            else:
                mask = segmentation[batch_index]

            boxes, scores = self.boxes_from_bitmap(
                pred[batch_index],
                mask,
                src_w,
                src_h,
            )

            boxes_batch.append({
                "points": boxes,
                "scores": scores,
            })

        return boxes_batch

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        pred: [H, W] 概率图
        bitmap: [H, W] 二值图
        dest_width, dest_height: 原图尺寸
        """
        height, width = bitmap.shape

        bitmap = (bitmap * 255).astype(np.uint8)

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]

            points, sside = self.get_mini_boxes(contour)

            if sside < self.min_size:
                continue

            points_np = np.array(points)

            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points_np.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)

            if self.box_thresh > score:
                continue

            box = self.unclip(points_np, self.unclip_ratio)

            if box is None or len(box) == 0:
                continue

            box = np.array(box).reshape(-1, 1, 2)

            box, sside = self.get_mini_boxes(box)

            if sside < self.min_size + 2:
                continue

            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width),
                0,
                dest_width,
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height),
                0,
                dest_height,
            )

            boxes.append(box.astype("int32"))
            scores.append(float(score))

        return np.array(boxes, dtype=np.int32), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)

        if poly.length <= 0:
            return None

        distance = poly.area * unclip_ratio / poly.length

        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        expanded = np.array(offset.Execute(distance))

        if expanded.size == 0:
            return None

        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3

        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4],
        ]

        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape[:2]

        box = box.copy()

        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin

        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        h, w = bitmap.shape[:2]

        contour = contour.copy()

        xmin = np.clip(np.min(contour[:, 0, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 0, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 0, 1]), 0, h - 1)

        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0, 0] = contour[:, 0, 0] - xmin
        contour[:, 0, 1] = contour[:, 0, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


db_post = DBPostProcess(
    thresh=DET_DB_THRESH,
    box_thresh=DET_DB_BOX_THRESH,
    max_candidates=DET_MAX_CANDIDATES,
    unclip_ratio=DET_DB_UNCLIP_RATIO,
    min_size=DET_MIN_SIZE,
    score_mode=DET_DB_SCORE_MODE,
    use_dilation=False,
)


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def filter_tag_det_res(dt_boxes, image_shape):
    """
    PaddleOCR TextDetector 里的 filter_tag_det_res 风格。
    过滤过小框，修正点顺序和边界。
    """
    img_height, img_width = image_shape[:2]

    dt_boxes_new = []

    for box in dt_boxes:
        box = np.array(box).astype(np.float32)
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)

        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))

        if rect_width <= 3 or rect_height <= 3:
            continue

        dt_boxes_new.append(box)

    return np.array(dt_boxes_new)


def order_points_clockwise(pts):
    """
    PaddleOCR 常用四点排序：
    左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def sorted_boxes(dt_boxes):
    """
    PaddleOCR sorted_boxes 风格：
    先按 y，再按 x 排。
    """
    num_boxes = dt_boxes.shape[0]

    sorted_boxes_list = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))

    _boxes = list(sorted_boxes_list)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break

    return np.array(_boxes)


def run_det(det_rknn, frame):
    inp, shape_info = det_preprocess(frame)

    out = safe_infer_det(det_rknn, inp, "det-rknn")

    if out is None:
        return [], "det_failed", 0.0, 0.0

    score = extract_score_map(out)

    pred = score[np.newaxis, np.newaxis, :, :].astype(np.float32)
    shape_list = shape_info[np.newaxis, :].astype(np.float32)

    post_result = db_post(pred, shape_list)

    dt_boxes = post_result[0]["points"]
    scores = post_result[0]["scores"]

    if dt_boxes is None or len(dt_boxes) == 0:
        return [], "rknn+official_dbpost", float(score.max()), float(score.mean())

    dt_boxes = filter_tag_det_res(dt_boxes, frame.shape)

    if len(dt_boxes) > 0:
        dt_boxes = sorted_boxes(dt_boxes)

    items = []

    for i, box in enumerate(dt_boxes):
        x1 = int(box[:, 0].min())
        y1 = int(box[:, 1].min())
        x2 = int(box[:, 0].max())
        y2 = int(box[:, 1].max())

        bw = x2 - x1
        bh = y2 - y1

        if bw < 8 or bh < 5:
            continue

        score_i = scores[i] if i < len(scores) else 1.0

        items.append({
            "box": box.astype(np.float32),
            "bbox": (x1, y1, x2, y2),
            "score": float(score_i),
        })

    return items, "rknn+official_dbpost", float(score.max()), float(score.mean())


# ============================================================
# crop_by_box：复刻训练数据裁剪逻辑
# ============================================================
def crop_quality_filter(crop):
    """
    严格过滤送入 rec 的 crop。

    目标保留：
      白纸/浅灰底 + 黑色中文字

    强制过滤：
      蓝地
      深色塑料板
      绿色/彩色英文板
      纯白纸边
      只有边缘线的 crop
    """
    if crop is None or crop.size == 0:
        return False, "empty"

    h, w = crop.shape[:2]

    if w < MIN_W or h < MIN_H:
        return False, "small"

    if w * h < MIN_AREA:
        return False, "area_small"

    aspect = w / float(max(h, 1))

    if aspect < 0.5 or aspect > 25:
        return False, f"bad_aspect={aspect:.2f}"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 1. 蓝地过滤
    blue_ratio = float(((H > 90) & (H < 135) & (S > 55)).mean())

    if blue_ratio > 0.35:
        return False, f"blue={blue_ratio:.2f}"

    mean_s = float(S.mean())
    mean_v = float(V.mean())

    # 2. 白纸/浅灰底比例
    # 中文车间牌的 crop 应该大部分是低饱和、高亮区域
    paper_ratio = float(((V > 125) & (S < 95)).mean())
    white_ratio = float(((V > 155) & (S < 80)).mean())

    # 3. 彩色/塑料板比例
    color_ratio = float((S > 120).mean())
    strong_color_ratio = float((S > 150).mean())

    # 4. 黑字比例
    dark_mask = (gray < 145)
    dark_ratio = float(dark_mask.mean())

    # 深色英文板、背景板
    if mean_v < 90:
        return False, f"dark_board={mean_v:.1f}"

    # 饱和度太高，大概率是绿色/彩色塑料板，不是白纸
    if mean_s > 105 and paper_ratio < 0.45:
        return False, f"color_board_s={mean_s:.1f},paper={paper_ratio:.2f}"

    if color_ratio > 0.38 and paper_ratio < 0.55:
        return False, f"color_ratio={color_ratio:.2f},paper={paper_ratio:.2f}"

    if strong_color_ratio > 0.22 and paper_ratio < 0.60:
        return False, f"strong_color={strong_color_ratio:.2f}"

    # 必须像白纸
    if paper_ratio < 0.42 and white_ratio < 0.25:
        return False, f"not_paper={paper_ratio:.2f},white={white_ratio:.2f}"

    # 纯白纸边 / 空白区域
    if dark_ratio < 0.006:
        return False, f"blank={dark_ratio:.4f}"

    # 太黑也不对
    if dark_ratio > 0.42:
        return False, f"too_dark={dark_ratio:.2f}"

    ys, xs = np.where(dark_mask)

    if len(xs) < 8:
        return False, "no_dark_pixels"

    dx1, dx2 = int(xs.min()), int(xs.max())
    dy1, dy2 = int(ys.min()), int(ys.max())

    dark_span_x = (dx2 - dx1 + 1) / float(w)
    dark_span_y = (dy2 - dy1 + 1) / float(h)

    # 只有一条横线，不是文字
    if dark_span_x > 0.50 and dark_span_y < 0.08:
        return False, "horizontal_line"

    # 只有竖边，不是文字
    if dark_span_x < 0.12 and dark_span_y > 0.35:
        return False, "vertical_edge"

    # 黑色像素太贴边，常见于塑料板边缘/纸边
    if (dx1 < w * 0.03 or dx2 > w * 0.97) and dark_span_x < 0.45:
        return False, "border_dark"

    # 中文一整行通常横向跨度不会太小
    if dark_span_x < 0.18:
        return False, f"dark_too_narrow={dark_span_x:.2f}"

    return True, f"ok paper={paper_ratio:.2f} white={white_ratio:.2f} dark={dark_ratio:.3f}"


def crop_by_box_like_training(img, box):
    """
    这部分基本照搬你训练时的 crop_by_box：
    外接矩形 -> pad_x=0.30 -> pad_y=0.80 -> crop -> 过滤蓝底
    """
    h, w = img.shape[:2]

    pts = np.array(box).astype(np.float32)

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))

    bw = x2 - x1
    bh = y2 - y1

    if bw < MIN_W or bh < MIN_H:
        return None, "small"

    if bw * bh < MIN_AREA:
        return None, "area_small"

    pad_x = int(bw * PAD_X_RATIO)
    pad_y = int(bh * PAD_Y_RATIO)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = img[y1:y2, x1:x2].copy()

    ok, reason = crop_quality_filter(crop)

    if not ok:
        return None, reason

    crop_box = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)

    return {
        "crop": crop,
        "box": crop_box,
        "source": "official_dbpost_crop_by_box",
        "reason": reason,
    }, reason


def build_good_crops(frame, det_items):
    crops = []
    rejected = []

    for it in det_items:
        item, reason = crop_by_box_like_training(frame, it["box"])

        if item is None:
            rejected.append((it, reason))
            continue

        crops.append(item)

    # 去重
    uniq = []
    seen = set()

    for c in crops:
        box = c["box"]
        x1 = int(box[:, 0].min())
        y1 = int(box[:, 1].min())
        x2 = int(box[:, 0].max())
        y2 = int(box[:, 1].max())

        key = (x1 // 8, y1 // 8, x2 // 8, y2 // 8)

        if key in seen:
            continue

        seen.add(key)
        uniq.append(c)

    return uniq, rejected


# ============================================================
# REC
# ============================================================
def rec_preprocess(crop):
    """
    保留原来的 NCHW 默认 rec 输入。
    """
    return rec_preprocess_mode(crop, "nchw_default")


def rec_preprocess_mode(crop, mode):
    """
    rec 多模式输入：
      nchw_default: 原来的方式
      bgr_nhwc:     BGR + NHWC + data_format=['nhwc']
      rgb_nhwc:     RGB + NHWC + data_format=['nhwc']

    det 已经证明 RKNN 对 data_format 很敏感，
    rec 也做多模式一致性，减少偶发错误。
    """
    imgC, imgH, imgW = REC_SHAPE

    if crop is None or crop.size == 0:
        return None

    img = crop.copy()

    if mode.startswith("rgb"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        return None

    ratio = w / float(h)
    resized_w = int(math.ceil(imgH * ratio))
    resized_w = max(1, min(imgW, resized_w))

    resized = cv2.resize(img, (resized_w, imgH))
    resized = resized.astype(np.float32) / 255.0
    resized = (resized - 0.5) / 0.5

    if mode.endswith("nhwc"):
        pad = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        pad[:, :resized_w, :] = resized
        return np.expand_dims(pad, 0).astype(np.float32)

    # nchw_default
    resized = resized.transpose(2, 0, 1)
    pad = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    pad[:, :, :resized_w] = resized

    return np.expand_dims(pad, 0).astype(np.float32)


def rec_infer_mode(rec_rknn, inp, mode):
    if inp is None:
        return None

    try:
        if mode.endswith("nhwc"):
            return rec_rknn.inference(inputs=[inp], data_format=["nhwc"])
        else:
            return rec_rknn.inference(inputs=[inp])
    except Exception as e:
        print(f"[WARN] rec mode {mode} failed:", e)
        return None


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def ctc_decode(rec_out):
    arr = np.array(rec_out[0])
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        return "", 0.0

    # 兼容 [C,T] / [T,C]
    if arr.shape[0] == len(CHARSET) and arr.shape[1] != len(CHARSET):
        arr = arr.T

    if arr.shape[-1] != len(CHARSET):
        return "", 0.0

    if arr.max() > 1.5 or arr.min() < -0.5:
        prob = softmax(arr, axis=-1)
    else:
        prob = arr

    idxs = np.argmax(prob, axis=-1)
    maxp = np.max(prob, axis=-1)

    text = []
    conf = []
    last = -1

    for idx, p in zip(idxs, maxp):
        idx = int(idx)

        if idx != 0 and idx != last and idx < len(CHARSET):
            text.append(CHARSET[idx])
            conf.append(float(p))

        last = idx

    if not conf:
        return "", 0.0

    return "".join(text), float(np.mean(conf))


def levenshtein(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def closed_set_filter(text, rec_score):
    """
    OCR 后闭集过滤：
    食品：必须有“食”
    日用品：必须有“日”和“用”
    电子：必须有“电”和“子”
    """
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")

    if not text:
        return "unknown", -999.0

    if rec_score < 0.50:
        return "unknown", -999.0

    candidates = []

    if "食" in text and "日" not in text and "电" not in text:
        candidates.append("食品加工车间")

    if "日" in text and "用" in text and "食" not in text and "电" not in text:
        candidates.append("日用品加工车间")

    if "电" in text and "子" in text and "食" not in text and "日" not in text:
        candidates.append("电子产品生产车间")

    if not candidates:
        return "unknown", -999.0

    best_label = "unknown"
    best_score = -999.0

    for label in candidates:
        dist = levenshtein(text, label)
        edit_score = 1.0 - dist / float(max(len(text), len(label)))

        kw = 0.0

        for ch in label:
            if ch in text:
                kw += 0.45

        total = edit_score * 4.0 + kw + rec_score * 1.8

        if total > best_score:
            best_score = total
            best_label = label

    if best_score < 3.5:
        return "unknown", best_score

    return best_label, best_score


def recognize_crops(rec_rknn, crops):
    """
    对每个 crop 做多模式 rec。
    只有多个模式结果一致，才把该 crop 判为有效。
    这样可以明显减少“食品加工车间偶尔识别成电子”的情况。
    """
    results = []

    rec_modes = [
        "nchw_default",
        "bgr_nhwc",
        "rgb_nhwc",
    ]

    for idx, c in enumerate(crops):
        crop = c["crop"]

        mode_results = []

        for mode in rec_modes:
            inp = rec_preprocess_mode(crop, mode)
            out = rec_infer_mode(rec_rknn, inp, mode)

            if out is None:
                continue

            raw, rec_score = ctc_decode(out)
            label, match_score = closed_set_filter(raw, rec_score)

            mode_results.append({
                "mode": mode,
                "raw": raw,
                "rec_score": rec_score,
                "label": label,
                "match_score": match_score,
                "final_score": match_score + rec_score * 0.8,
            })

        if not mode_results:
            continue

        valid = [
            r for r in mode_results
            if r["label"] != "unknown" and r["rec_score"] >= 0.55
        ]

        # 默认 unknown
        chosen_label = "unknown"
        chosen_raw = ""
        chosen_rec_score = 0.0
        chosen_match_score = -999.0
        chosen_final_score = -999.0

        if valid:
            label_counter = Counter([r["label"] for r in valid])
            top_label, top_count = label_counter.most_common(1)[0]

            same = [r for r in valid if r["label"] == top_label]
            same.sort(key=lambda x: x["final_score"], reverse=True)

            # 要么至少两个模式一致
            # 要么单个模式非常强，但这种情况要求更严格
            if top_count >= 2:
                best = same[0]
                chosen_label = top_label
                chosen_raw = best["raw"]
                chosen_rec_score = best["rec_score"]
                chosen_match_score = best["match_score"]
                chosen_final_score = best["final_score"] + top_count * 0.7

            elif top_count == 1:
                best = same[0]

                # 单模式强结果也可以接受，但门槛更高
                if best["rec_score"] >= 0.90 and best["match_score"] >= 6.2:
                    chosen_label = top_label
                    chosen_raw = best["raw"]
                    chosen_rec_score = best["rec_score"]
                    chosen_match_score = best["match_score"]
                    chosen_final_score = best["final_score"]

        # 调试信息
        mode_debug = " || ".join([
            f"{r['mode']}:{r['raw']}({r['rec_score']:.2f})->{SHORT.get(r['label'], 'unknown')}"
            for r in mode_results
        ])

        results.append({
            "idx": idx,
            "crop": crop,
            "box": c["box"],
            "raw": chosen_raw,
            "rec_score": chosen_rec_score,
            "label": chosen_label,
            "match_score": chosen_match_score,
            "final_score": chosen_final_score,
            "source": c.get("source", ""),
            "reason": c.get("reason", ""),
            "mode_debug": mode_debug,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def decide_frame(results):
    valid = [r for r in results if r["label"] != "unknown"]

    if not valid:
        return "unknown", "", 0.0, None

    score_by_label = defaultdict(float)

    for r in valid:
        score_by_label[r["label"]] += r["final_score"]

    ranked = sorted(score_by_label.items(), key=lambda x: x[1], reverse=True)

    best_label, best_sum = ranked[0]
    second_sum = ranked[1][1] if len(ranked) > 1 else 0.0

    if second_sum > 0 and best_sum - second_sum < 0.8:
        return "unknown", "", 0.0, None

    best_items = [r for r in valid if r["label"] == best_label]
    best_items.sort(key=lambda x: x["final_score"], reverse=True)

    best = best_items[0]

    return best_label, best["raw"], best["final_score"], best


# ============================================================
# 可视化
# ============================================================
def draw_poly(img, box, color, thick=2):
    pts = np.array(box).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thick)


def put_text(img, s, y, color=(0, 255, 255)):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def make_crop_sheet(crops, results, max_show=10):
    res_by_idx = {}

    for r in results:
        res_by_idx[r["idx"]] = r

    thumbs = []

    for i, c in enumerate(crops[:max_show]):
        crop = c["crop"]

        if crop is None or crop.size == 0:
            continue

        img = crop.copy()
        h, w = img.shape[:2]

        target_h = 90
        new_w = int(w * target_h / max(1, h))
        new_w = max(120, min(600, new_w))

        img = cv2.resize(img, (new_w, target_h))

        canvas = np.zeros((130, 640, 3), dtype=np.uint8)
        canvas[:target_h, :new_w] = img

        r = res_by_idx.get(i)

        if r is not None:
            label = SHORT.get(r["label"], "unknown")
            txt = f"{i}:{label} {r['rec_score']:.2f} {r['raw'][:14]}"
        else:
            txt = f"{i}:no_rec"

        cv2.putText(canvas, txt, (5, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        thumbs.append(canvas)

    if not thumbs:
        blank = np.zeros((120, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "no good crop", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return blank

    rows = []

    for i in range(0, len(thumbs), 2):
        if i + 1 < len(thumbs):
            rows.append(np.hstack([thumbs[i], thumbs[i + 1]]))
        else:
            rows.append(np.hstack([thumbs[i], np.zeros_like(thumbs[i])]))

    return np.vstack(rows)


# ============================================================
# 主程序
# ============================================================
def main():
    print("字符表:", CHARSET)

    det_rknn = load_rknn(DET_MODEL, "det model")
    rec_rknn = load_rknn(REC_MODEL, "rec model")

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("摄像头打开失败")
        return

    ret, test = cap.read()

    if not ret or test is None:
        print("摄像头读帧失败")
        cap.release()
        return

    print("摄像头打开成功:", test.shape)

    start_web_server()

    vote = deque(maxlen=VOTE_WINDOW)

    print("开始 det + rec 识别")
    print("当前版本：RKNN det + PaddleOCR 官方 DBPostProcess + 训练式 crop_by_box + filter + RKNN rec")
    print("按 Ctrl+C 退出")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (640, 480))

            if MIRROR_FIX:
                frame = cv2.flip(frame, 1)

            vis = frame.copy()

            # 1. RKNN det + 官方 DBPostProcess
            det_items, det_layout, det_smax, det_smean = run_det(det_rknn, frame)

            # 绿色：官方 DBPostProcess 得到的 dt_boxes
            for i, it in enumerate(det_items):
                draw_poly(vis, it["box"], (0, 255, 0), 2)
                x1, y1, x2, y2 = it["bbox"]
                cv2.putText(
                    vis,
                    f"{i}:{it['score']:.2f}",
                    (x1, max(18, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )

            # 2. 训练式 crop_by_box + 过滤
            crops, rejected = build_good_crops(frame, det_items)

            # 蓝色：真正送 rec 的 good crop
            for c in crops:
                draw_poly(vis, c["box"], (255, 0, 0), 2)

            # 3. RKNN rec
            results = recognize_crops(rec_rknn, crops)

            # 4. 单帧决策
            frame_label, frame_text, frame_score, best = decide_frame(results)

            # 红色：best crop
            if best is not None:
                draw_poly(vis, best["box"], (0, 0, 255), 3)

            # 5. 多帧投票
            vote.append(frame_label)

            cnt = Counter([x for x in vote if x != "unknown"])

            lead_label = "unknown"
            lead_count = 0

            if cnt:
                lead_label, lead_count = cnt.most_common(1)[0]

            crop_sheet = make_crop_sheet(crops, results)

            put_text(vis, f"det:{len(det_items)} crops:{len(crops)} ocr:{len(results)}", 25)
            put_text(vis, f"frame:{SHORT.get(frame_label, 'unknown')} vote:{lead_count}/{VOTE_WINDOW}", 50)
            put_text(vis, f"layout:{det_layout} max:{det_smax:.3f}", 75)

            ocr_show = " | ".join([
                f"{r['raw']}({r['rec_score']:.2f})->{SHORT.get(r['label'], 'unknown')}"
                for r in results[:8]
            ])

            mode_show = "\n".join([
                r.get("mode_debug", "")
                for r in results[:4]
            ])

            reject_reasons = Counter([reason for _, reason in rejected])
            reject_show = " | ".join([
                f"{k}:{v}" for k, v in reject_reasons.most_common(6)
            ])

            status = (
                f"det框: {len(det_items)}\n"
                f"good crops: {len(crops)}\n"
                f"rejected crops: {len(rejected)}\n"
                f"OCR结果数: {len(results)}\n"
                f"当前帧: {SHORT.get(frame_label, 'unknown')}\n"
                f"领先投票: {lead_count}/{VOTE_WINDOW}\n"
                f"领先类别: {SHORT.get(lead_label, 'unknown')}\n"
                f"det_layout: {det_layout}\n"
                f"det_score_max: {det_smax:.4f}\n"
                f"det_score_mean: {det_smean:.6f}\n"
                f"best_raw_text: {frame_text}\n"
                f"best_score: {frame_score:.3f}\n"
                f"OCR: {ocr_show}\n"
                f"REC多模式:\n{mode_show}\n"
                f"reject: {reject_show}"
            )

            update_web(vis, crop_sheet, status)

            if lead_label != "unknown" and lead_count >= VOTE_NEED:
                print("=" * 40)
                print(f"识别到了：{lead_label}")
                print(f"原始OCR：{frame_text}  投票：{lead_count}/{VOTE_WINDOW}")
                print("=" * 40)
                vote.clear()
            else:
                print(
                    f"等待识别中... det:{len(det_items)} crops:{len(crops)} "
                    f"ocr:{len(results)} 当前帧:{SHORT.get(frame_label, 'unknown')} "
                    f"投票:{lead_count}/{VOTE_WINDOW} OCR:{ocr_show}"
                )

    except KeyboardInterrupt:
        print("退出识别")

    finally:
        cap.release()
        det_rknn.release()
        rec_rknn.release()


if __name__ == "__main__":
    main()
