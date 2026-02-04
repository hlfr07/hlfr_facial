import cv2
import numpy as np
import onnxruntime as ort

# =========================
# CONFIG
# =========================
MODEL_PATH = "scrfd_10g_320_batch64.onnx"
VIDEO_PATH = "video.mp4"   # o 0 para webcam

INPUT_SIZE = (320, 320)
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4

MEAN = 127.5
STD = 128.0

FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2   # SCRFD usa 2 anchors

# =========================
# UTILS
# =========================
def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def generate_anchors(height, width, stride):
    shifts_x = np.arange(0, width) * stride
    shifts_y = np.arange(0, height) * stride
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    centers = np.stack((shift_x, shift_y), axis=-1).reshape(-1, 2)

    if NUM_ANCHORS > 1:
        centers = np.repeat(centers, NUM_ANCHORS, axis=0)

    return centers.astype(np.float32)

# =========================
# LOAD MODEL
# =========================
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("VIDEO", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI_FACE", cv2.WINDOW_NORMAL)

# =========================
# LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    # -------- PREPROCESS --------
    img = cv2.resize(frame, INPUT_SIZE)
    blob = (img.astype(np.float32) - MEAN) / STD
    blob = blob.transpose(2, 0, 1)[None]

    outputs = sess.run(output_names, {input_name: blob})

    detections = []

    fmc = 3  # feature map count

    for idx, stride in enumerate(FEAT_STRIDES):
        scores = outputs[idx].reshape(-1)
        bbox_preds = outputs[idx + fmc].reshape(-1, 4) * stride

        feat_h = INPUT_SIZE[1] // stride
        feat_w = INPUT_SIZE[0] // stride

        centers = generate_anchors(feat_h, feat_w, stride)

        mask = scores >= CONF_THRESHOLD
        if not np.any(mask):
            continue

        scores = scores[mask]
        bbox_preds = bbox_preds[mask]
        centers = centers[mask]

        boxes = distance2bbox(centers, bbox_preds)

        boxes[:, [0, 2]] *= w0 / INPUT_SIZE[0]
        boxes[:, [1, 3]] *= h0 / INPUT_SIZE[1]

        for box, score in zip(boxes, scores):
            detections.append([*box, score])

    if detections:
        dets = np.array(detections, dtype=np.float32)
        dets = dets[nms(dets, NMS_THRESHOLD)]

        for x1, y1, x2, y2, score in dets:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w0, x2); y2 = min(h0, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow("ROI_FACE", face)

    cv2.imshow("VIDEO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
