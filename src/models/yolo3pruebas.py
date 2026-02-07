# import onnxruntime as ort
# import numpy as np
# import cv2
# import time

# MODEL_PATH = "yolo11x-seg.onnx"  # tu modelo ONNX
# IMAGE_PATH = "images.jpg"              # imagen a probar
# CONF_THRESHOLD = 0.25                  # confianza mínima para mostrar detecciones

# # ----------------- CLASE -----------------
# class BoundingBox:
#     def __init__(self, x, y, w, h, class_id, confidence_score, mask_coefficients):
#         self.xywh = [x, y, w, h]
#         self.class_id = class_id
#         self.confidence_score = confidence_score
#         self.mask_coefficients = mask_coefficients
#         self.mask = None

# # ----------------- UTILIDADES -----------------
# def resize_letter_box(img: np.ndarray, new_size: tuple) -> np.ndarray:
#     img_width, img_height = img.shape[1], img.shape[0]
#     new_width, new_height = new_size
#     scale_ratio = min(new_width / img_width, new_height / img_height)

#     new_unpad_width = int(round(img_width * scale_ratio))
#     new_unpad_height = int(round(img_height * scale_ratio))

#     dw = (new_width - new_unpad_width) / 2
#     dh = (new_height - new_unpad_height) / 2

#     if new_width != new_unpad_width or new_height != new_unpad_height:
#         img = cv2.resize(img, (new_unpad_width, new_unpad_height), interpolation=cv2.INTER_LANCZOS4)

#     pad_top = round(dh - 0.1)
#     pad_bottom = round(dh + 0.1)
#     pad_left = round(dw - 0.1)
#     pad_right = round(dw + 0.1)
#     return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114,114,114))

# def preprocess_img(img: np.ndarray) -> np.ndarray:
#     img = resize_letter_box(img, (640, 640))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.ascontiguousarray(img.transpose(2,0,1))
#     img = img[None,...].astype(np.float32) / 255.0
#     return img

# def get_filtered_boxes_fast(boxes: np.ndarray) -> list[BoundingBox]:
#     coords = boxes[:, :4, 0]
#     class_scores = boxes[:, 4:84, 0]
#     mask_coefficients = boxes[:, 84:, 0]

#     class_ids = class_scores.argmax(axis=1)
#     confidence_scores = class_scores[np.arange(len(class_ids)), class_ids]
#     keep = confidence_scores > CONF_THRESHOLD
#     if not np.any(keep):
#         return []

#     coords = coords[keep]
#     class_ids = class_ids[keep]
#     confidence_scores = confidence_scores[keep]
#     mask_coefficients = mask_coefficients[keep]

#     widths = coords[:,2].astype(np.int32)
#     heights = coords[:,3].astype(np.int32)
#     xs = np.maximum(0, coords[:,0]-0.5*widths).astype(np.int32)
#     ys = np.maximum(0, coords[:,1]-0.5*heights).astype(np.int32)

#     return [
#         BoundingBox(int(x), int(y), int(w), int(h), class_id, confidence_score, mask_coeff)
#         for x,y,w,h,class_id,confidence_score,mask_coeff in zip(xs,ys,widths,heights,class_ids,confidence_scores,mask_coefficients)
#     ]

# def scale_boxes(boxes: list[BoundingBox], orig_image_size: tuple, letterbox_shape=(640,640)):
#     orig_w, orig_h = orig_image_size
#     lb_w, lb_h = letterbox_shape
#     scale = min(lb_h/orig_h, lb_w/orig_w)
#     pad_x = (lb_w - orig_w*scale)/2
#     pad_y = (lb_h - orig_h*scale)/2
#     for box in boxes:
#         x, y, w, h = box.xywh
#         x = (x - pad_x)/scale
#         y = (y - pad_y)/scale
#         w /= scale
#         h /= scale
#         box.xywh = [x,y,w,h]
#     return boxes

# def nms_boxes(filtered_boxes: list[BoundingBox]):
#     if not filtered_boxes:
#         return []
#     boxes = [b.xywh for b in filtered_boxes]
#     scores = [b.confidence_score for b in filtered_boxes]
#     indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, 0.5)
#     if len(indices)==0:
#         return []
#     indices = indices.flatten() if isinstance(indices, np.ndarray) else [i[0] for i in indices]
#     return [filtered_boxes[i] for i in indices]

# def resize_mask_remove_letterbox(mask, original_img_size, letterbox_size=(640,640)):
#     lb_w, lb_h = letterbox_size
#     orig_w, orig_h = original_img_size
#     scale = min(lb_h/orig_h, lb_w/orig_w)
#     unpad_w = int(round(orig_w*scale))
#     unpad_h = int(round(orig_h*scale))
#     pad_w = lb_w - unpad_w
#     pad_h = lb_h - unpad_h
#     pad_left = pad_w//2
#     pad_top = pad_h//2

#     resized = cv2.resize(mask, letterbox_size, interpolation=cv2.INTER_LINEAR)
#     cropped = resized[pad_top:pad_top+unpad_h, pad_left:pad_left+unpad_w]
#     return cv2.resize(cropped, original_img_size, interpolation=cv2.INTER_LINEAR)

# def process_masks(proto_masks: np.ndarray, bboxes: list[BoundingBox], img_size: tuple):

#     # proto shape → (1, C, H, W)
#     _, proto_channels, mask_h, mask_w = proto_masks.shape

#     # Flatten dinámico
#     proto_masks = proto_masks.reshape(proto_channels, -1)

#     width, height = img_size

#     for box in bboxes:
#         coeffs = box.mask_coefficients

#         # Ajustar tamaño si no coincide
#         if coeffs.shape[0] != proto_channels:
#             coeffs = coeffs[:proto_channels]

#         combined_mask = coeffs @ proto_masks
#         combined_mask = combined_mask.reshape((mask_h, mask_w))

#         resized_cropped_mask = resize_mask_remove_letterbox(
#             combined_mask,
#             (width, height),
#             (640, 640)
#         )

#         x, y, w, h = map(int, box.xywh)

#         cropped_mask = resized_cropped_mask[y:y+h, x:x+w]

#         cropped_mask = cv2.resize(
#             cropped_mask,
#             (w, h),
#             interpolation=cv2.INTER_NEAREST
#         )

#         _, cropped_mask = cv2.threshold(
#             cropped_mask, 0.5, 255, cv2.THRESH_BINARY
#         )

#         box.mask = cropped_mask

#     return bboxes



# def draw_boxes(img: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
#     """
#     Dibuja los bounding boxes y sus máscaras sobre la imagen.
#     """
#     # Dibuja cajas y superpone máscaras en la imagen
#     for box in boxes:
#         x, y, w, h = map(int, box.xywh)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Dibuja máscara
#         if box.mask is not None and w > 0 and h > 0:
#            roi = img[y:y+h, x:x+w]
#            mask = box.mask

#             # Asegurar mismo tamaño
#            mh, mw = mask.shape[:2]
#            rh, rw = roi.shape[:2]

#            h_min = min(mh, rh)
#            w_min = min(mw, rw)

#            roi = roi[:h_min, :w_min]
#            mask = mask[:h_min, :w_min]

#            mask_bool = (mask == 255)

#            color_img = np.zeros_like(roi, dtype=np.uint8)
#            color_img[:] = (0, 0, 255)

#            alpha = 0.5
#            blended_roi = cv2.addWeighted(roi, 1 - alpha, color_img, alpha, 0)

#            roi[mask_bool] = blended_roi[mask_bool]
#            img[y:y+h_min, x:x+w_min] = roi


#     return img

# # ----------------- MAIN -----------------
# img = cv2.imread(IMAGE_PATH)
# orig_size = img.shape[1], img.shape[0]
# input_tensor = preprocess_img(img)

# session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
# outputs = session.run(None, {"images": input_tensor})

# boxes = get_filtered_boxes_fast(outputs[0].transpose())
# boxes = scale_boxes(boxes, orig_size)
# boxes = nms_boxes(boxes)
# boxes = process_masks(outputs[1], boxes, (orig_size[0], orig_size[1]))
# result = draw_boxes(img.copy(), boxes)

# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






from ultralytics import YOLO
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = "yolo26x-seg.onnx"
IMAGE_PATH = "images.jpg"
CONF = 0.4
# ---------------------------------------

# Cargar modelo ONNX
model = YOLO(MODEL_PATH)

# Inferencia
results = model.predict(
    source=IMAGE_PATH,
    conf=CONF,
    save=False,   # no guardar
    show=False    # no mostrar automático
)

# Obtener imagen resultante con máscaras/detecciones dibujadas
annotated_frame = results[0].plot()

# Mostrar en ventana
cv2.imshow("Resultado YOLO11 Segmentation", annotated_frame)

print("Presiona cualquier tecla para cerrar...")
cv2.waitKey(0)
cv2.destroyAllWindows()
