import onnxruntime as ort
from PIL import Image
import numpy as np
import cv2
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
import os
import base64
from io import BytesIO

MODEL_PATH = "yolov8m.onnx"  # tu modelo ONNX
IMAGE_PATH = "images.jpg"      # la imagen a probar
CONF_THRESHOLD = 0.7         # confianza mínima para mostrar detecciones
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
def prepare_input(image_input):
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = Image.open(image_input)
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

def draw_boxes(image_input, boxes):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img_pil = Image.open(image_input)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    for x1, y1, x2, y2, label, prob in boxes:
        color = colors[yolo_classes.index(label)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), 2)
        cv2.putText(img, f"{label} {prob:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
    return img

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear carpeta uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Object Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-form {
            text-align: center;
            margin: 20px 0;
        }
        .file-input {
            padding: 10px;
            border: 2px dashed #ddd;
            border-radius: 5px;
            margin: 10px;
        }
        .btn {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            text-align: center;
        }
        .detection-list {
            text-align: left;
            margin-top: 20px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 YOLO Object Detection</h1>
        
        <div class="upload-form">
            <form method="POST" enctype="multipart/form-data">
                <div class="file-input">
                    <input type="file" name="image" accept="image/*" required>
                </div>
                <button type="submit" class="btn">Detectar Objetos</button>
            </form>
        </div>
        
        {% if result_image %}
        <div class="result">
            <h3>Resultado:</h3>
            <img src="data:image/jpeg;base64,{{ result_image }}" alt="Resultado">
            
            {% if detections %}
            <div class="detection-list">
                <h4>Detecciones encontradas ({{ detections|length }}):</h4>
                <ul>
                {% for detection in detections %}
                    <li><strong>{{ detection[4] }}</strong> - Confianza: {{ "%.2f"|format(detection[5]) }}%</li>
                {% endfor %}
                </ul>
            </div>
            {% else %}
            <div class="detection-list">
                <p>No se detectaron objetos con confianza superior a {{ conf_threshold }}%</p>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

def image_to_base64(img_cv2):
    """Convierte imagen CV2 a base64 para mostrar en HTML"""
    _, buffer = cv2.imencode('.jpg', img_cv2)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # Procesar imagen
                input_tensor, w, h = prepare_input(file)
                output = run_model(input_tensor)
                boxes = process_output(output, w, h)
                
                # Dibujar boxes en la imagen
                file.seek(0)  # Reset file pointer
                result_img = draw_boxes(file, boxes)
                
                # Convertir a base64 para mostrar
                result_base64 = image_to_base64(result_img)
                
                return render_template_string(HTML_TEMPLATE, 
                                            result_image=result_base64,
                                            detections=boxes,
                                            conf_threshold=int(CONF_THRESHOLD*100))
            
            except Exception as e:
                return f"Error procesando imagen: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    print("🚀 Iniciando servidor YOLO Detection...")
    print(f"📊 Modelo: {MODEL_PATH}")
    print(f"🎯 Confianza mínima: {CONF_THRESHOLD*100}%")
    print("🌐 Abre tu navegador en: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
