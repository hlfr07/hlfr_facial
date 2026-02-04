import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "scrfd_10g_gnkps_fp32.onnx"
VIDEO_PATH = "video.mp4"

INPUT_SIZE = (640, 640)  # Aumentado para mejor detección
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

MEAN = 127.5
STD = 128.0

FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

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

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    img = cv2.resize(frame, INPUT_SIZE)
    blob = (img.astype(np.float32) - MEAN) / STD
    blob = blob.transpose(2, 0, 1)[None]

    outputs = sess.run(output_names, {input_name: blob})
    detections = []

    for i, stride in enumerate(FEAT_STRIDES):
        bbox_idx = i * 3
        score_idx = i * 3 + 1
        
        bbox_blob = outputs[bbox_idx]
        score_blob = outputs[score_idx]
        
        # Usar scores RAW (ya vienen normalizados)
        scores = score_blob[0, :, 0]
        bbox_preds = bbox_blob[0]
        
        bbox_preds = bbox_preds * stride
        
        fh = INPUT_SIZE[1] // stride
        fw = INPUT_SIZE[0] // stride
        
        ys, xs = np.meshgrid(
            np.arange(fh),
            np.arange(fw),
            indexing="ij"
        )
        centers = np.stack(
            [(xs + 0.5) * stride, (ys + 0.5) * stride],
            axis=-1
        ).reshape(-1, 2)
        
        centers = np.repeat(centers, NUM_ANCHORS, axis=0)
        
        mask = scores >= CONF_THRESHOLD
        if not np.any(mask):
            continue
        
        boxes = distance2bbox(
            centers[mask],
            bbox_preds[mask]
        )
        
        # Escalar a dimensiones originales
        boxes[:, [0, 2]] *= w0 / INPUT_SIZE[0]
        boxes[:, [1, 3]] *= h0 / INPUT_SIZE[1]
        
        # Clip a los límites de la imagen
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w0)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h0)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w0)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h0)
        
        detections.append(
            np.hstack((boxes, scores[mask, None]))
        )

    if detections:
        dets = np.vstack(detections)
        keep_indices = nms(dets, NMS_THRESHOLD)
        dets = dets[keep_indices]

        for x1, y1, x2, y2, score in dets:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Verificar que el box sea válido
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

    cv2.imshow("VIDEO", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ESTE NO SIRVE AUN TIENE ERORRES