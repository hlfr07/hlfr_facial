import onnxruntime as ort
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = "yolov8m.onnx"  # tu modelo ONNX
IMAGE_PATH = "images.jpg"      # la imagen a probar
CONF_THRESHOLD = 0.8         # confianza mínima para mostrar detecciones
INPUT_SIZE = 640

# Clases COCO para YOLOv8
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Colores random para los boxes
colors = np.random.randint(0, 255, size=(len(yolo_classes), 3), dtype=int)

# ---------------- FUNCIONES ----------------
def prepare_input(image_path):
    img = Image.open(image_path)
    img_width, img_height = img.size
    img_resized = img.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    input_tensor = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1)
    input_tensor = input_tensor[np.newaxis, :, :, :]
    return input_tensor, img_width, img_height

def run_model(input_tensor):
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    outputs = session.run(["output0"], {"images": input_tensor})
    return outputs[0]

def process_output(output, img_width, img_height):
    output = output[0].astype(float).T  # (num_boxes, 85)
    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < CONF_THRESHOLD:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / INPUT_SIZE * img_width
        y1 = (yc - h/2) / INPUT_SIZE * img_height
        x2 = (xc + w/2) / INPUT_SIZE * img_width
        y2 = (yc + h/2) / INPUT_SIZE * img_height
        boxes.append([int(x1), int(y1), int(x2), int(y2), label, float(prob)])
    return boxes

def draw_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    for x1, y1, x2, y2, label, prob in boxes:
        color = colors[yolo_classes.index(label)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), 2)
        cv2.putText(img, f"{label} {prob:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
input_tensor, w, h = prepare_input(IMAGE_PATH)
output = run_model(input_tensor)
boxes = process_output(output, w, h)
print(f"Detecciones: {len(boxes)}")
for b in boxes:
    print(b)

draw_boxes(IMAGE_PATH, boxes)
