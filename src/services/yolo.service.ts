import { Injectable, Logger } from '@nestjs/common';
import * as ort from 'onnxruntime-web';
import * as path from 'path';
import sharp from 'sharp';

export interface DetectionBox {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    label: string;
    confidence: number;
    classId: number;
}

@Injectable()
export class YoloService {
    private readonly logger = new Logger(YoloService.name);
    private yoloSession: ort.InferenceSession | null = null;
    private readonly INPUT_SIZE = 640;
    private readonly CONF_THRESHOLD = 0.7;

    private readonly yoloClasses = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];

    constructor() {
        this.initializeModel();
    }

    private async initializeModel() {
        try {
            const yoloPath = path.join(process.cwd(), 'src', 'models', 'yolov8m.onnx');
            this.logger.log(`🔍 Checking YOLO model: ${yoloPath}`);

            const fs = require('fs');
            if (!fs.existsSync(yoloPath)) {
                this.logger.error('❌ YOLO model not found!');
                return;
            }

            this.logger.log('⏳ Loading YOLO model...');
            this.yoloSession = await ort.InferenceSession.create(yoloPath,
                {
                    executionProviders: ['wasm'] // CPU
                });
            this.logger.log('✅ YOLO model loaded successfully');
        } catch (error) {
            this.logger.error('❌ Error loading YOLO model:', error);
        }
    }

    async detectObjects(imagePath: string): Promise<{
        boxes: DetectionBox[];
        originalWidth: number;
        originalHeight: number;
    }> {
        try {
            if (!this.yoloSession) {
                throw new Error('YOLO model not initialized');
            }

            // Leer metadata original (igual que Python: img.size)
            const metadata = await sharp(imagePath).metadata();
            const imgWidth = metadata.width!;
            const imgHeight = metadata.height!;

            this.logger.log(`📐 Original image size: ${imgWidth}x${imgHeight}`);

            // Preparar entrada: redimensionar DIRECTO a 640x640 (como Python img.resize())
            // SIN fit, lo que hace resize/stretch directo de la imagen
            const resizedBuffer = await sharp(imagePath)
                .resize(this.INPUT_SIZE, this.INPUT_SIZE, { fit: 'fill' })
                .removeAlpha()
                .raw()
                .toBuffer({ resolveWithObject: true });

            const { data, info } = resizedBuffer;

            this.logger.log(`📊 Resized to: ${info.width}x${info.height}, channels: ${info.channels}`);

            // Normalizar a [0, 1] y convertir a CHW format (igual que Python)
            const inputTensor = new Float32Array(3 * this.INPUT_SIZE * this.INPUT_SIZE);

            for (let i = 0; i < this.INPUT_SIZE * this.INPUT_SIZE; i++) {
                // Leer RGB directamente (sharp devuelve RGB en order)
                const r = data[i * 3] / 255.0;
                const g = data[i * 3 + 1] / 255.0;
                const b = data[i * 3 + 2] / 255.0;

                // Formato CHW (C=channel, H=height, W=width)
                inputTensor[i] = r;
                inputTensor[this.INPUT_SIZE * this.INPUT_SIZE + i] = g;
                inputTensor[this.INPUT_SIZE * this.INPUT_SIZE * 2 + i] = b;
            }

            this.logger.log(`✅ Input tensor ready: shape [1,3,${this.INPUT_SIZE},${this.INPUT_SIZE}]`);

            // Ejecutar modelo (igual que Python)
            const input = new ort.Tensor('float32', inputTensor, [1, 3, this.INPUT_SIZE, this.INPUT_SIZE]);
            const results = await this.yoloSession.run({ images: input });

            // Obtener output (igual que Python: session.run(["output0"], ...))
            const output0 = results['output0'];
            const outputData = output0.data as Float32Array;

            this.logger.log(`📊 Output shape: ${JSON.stringify(output0.dims)}`);
            this.logger.log(`📊 Output data length: ${outputData.length}`);

            // Procesar detecciones
            const boxes = this.processOutput(outputData, output0.dims as number[], imgWidth, imgHeight);

            return {
                boxes,
                originalWidth: imgWidth,
                originalHeight: imgHeight,
            };
        } catch (error) {
            this.logger.error('Error in object detection:', error);
            throw error;
        }
    }

    private processOutput(
        output: Float32Array,
        dims: number[],
        imgWidth: number,
        imgHeight: number
    ): DetectionBox[] {
        const boxes: DetectionBox[] = [];

        // Output shape es (1, 84, 8400) o similar
        // Python hace: output[0].T para transponer de (84, 8400) a (8400, 84)
        // Donde 84 = 4 bbox + 80 clases (sin objectness separado)
        // Y 8400 = número de detecciones

        const numChannels = dims[1];      // 84 (4 bbox + 80 classes)
        const numDetections = dims[2];    // 8400 (detection anchors)

        this.logger.log(`🔍 Processing ${numDetections} detections with ${numChannels} channels...`);
        let detectionCount = 0;

        // Iterar sobre detecciones (transpuesto)
        for (let d = 0; d < numDetections; d++) {
            // El tensor está en formato [batch, channel, detection]
            // Para acceder al elemento [0, c, d] en el array plano:
            // índice = 0 * (numChannels * numDetections) + c * numDetections + d
            // que simplifica a: c * numDetections + d

            // Extraer bbox: xc, yc, w, h (primeros 4 canales)
            const xc = output[0 * numDetections + d];
            const yc = output[1 * numDetections + d];
            const w = output[2 * numDetections + d];
            const h = output[3 * numDetections + d];

            // Extraer clase con máxima probabilidad (canales 4-83)
            let maxProb = 0;
            let classId = 0;

            for (let c = 4; c < numChannels; c++) {
                const prob = output[c * numDetections + d];
                if (prob > maxProb) {
                    maxProb = prob;
                    classId = c - 4;  // Restar 4 porque las clases empiezan en canal 4
                }
            }

            // Python: prob = row[4:].max()
            const confidence = maxProb;

            // Filtro threshold
            if (confidence < this.CONF_THRESHOLD) {
                continue;
            }

            detectionCount++;

            // Convertir de center (xc, yc, w, h) a corners (x1, y1, x2, y2)
            // Escalar de 640x640 a imagen original
            const x1 = ((xc - w / 2) / this.INPUT_SIZE) * imgWidth;
            const y1 = ((yc - h / 2) / this.INPUT_SIZE) * imgHeight;
            const x2 = ((xc + w / 2) / this.INPUT_SIZE) * imgWidth;
            const y2 = ((yc + h / 2) / this.INPUT_SIZE) * imgHeight;

            boxes.push({
                x1: Math.max(0, Math.floor(x1)),
                y1: Math.max(0, Math.floor(y1)),
                x2: Math.min(imgWidth, Math.floor(x2)),
                y2: Math.min(imgHeight, Math.floor(y2)),
                label: this.yoloClasses[classId] || `Class ${classId}`,
                confidence,
                classId,
            });
        }

        this.logger.log(`✅ Found ${detectionCount} detections above ${this.CONF_THRESHOLD} threshold`);
        return boxes;
    }

    async drawBoxesOnImage(imagePath: string, boxes: DetectionBox[]): Promise<Buffer> {
        try {
            let img = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true });
            const { data, info } = img;

            const width = info.width;
            const height = info.height;
            const channels = info.channels;

            // Dibujar boxes (línea roja de 3px)
            for (const box of boxes) {
                const color = this.getColorForClass(box.classId);

                // Dibujar rectángulo
                this.drawRectangle(data, width, height, channels, box, color);

                // Dibujar texto
                this.drawText(data, width, height, channels, box, color);
            }

            // Convertir back a imagen
            const output = await sharp(Buffer.from(data), {
                raw: { width, height, channels: 3 },
            })
                .png()
                .toBuffer();

            return output;
        } catch (error) {
            this.logger.error('Error drawing boxes:', error);
            throw error;
        }
    }

    private drawRectangle(
        data: Buffer,
        width: number,
        height: number,
        channels: number,
        box: DetectionBox,
        color: [number, number, number],
    ) {
        const thickness = 2;
        const [r, g, b] = color;

        // Top line
        for (let x = box.x1; x <= box.x2; x++) {
            for (let t = 0; t < thickness; t++) {
                const y = Math.min(box.y1 + t, height - 1);
                const idx = (y * width + x) * channels;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        // Bottom line
        for (let x = box.x1; x <= box.x2; x++) {
            for (let t = 0; t < thickness; t++) {
                const y = Math.max(box.y2 - t, 0);
                const idx = (y * width + x) * channels;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        // Left line
        for (let y = box.y1; y <= box.y2; y++) {
            for (let t = 0; t < thickness; t++) {
                const x = Math.min(box.x1 + t, width - 1);
                const idx = (y * width + x) * channels;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        // Right line
        for (let y = box.y1; y <= box.y2; y++) {
            for (let t = 0; t < thickness; t++) {
                const x = Math.max(box.x2 - t, 0);
                const idx = (y * width + x) * channels;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
    }

    private drawText(
        data: Buffer,
        width: number,
        height: number,
        channels: number,
        box: DetectionBox,
        color: [number, number, number],
    ) {
        // Dibujar fondo para el texto
        const text = `${box.label} ${(box.confidence * 100).toFixed(1)}%`;
        const textX = box.x1 + 5;
        const textY = box.y1 - 5;

        // Fondo blanco para el texto
        const bgWidth = text.length * 6;
        const bgHeight = 15;

        for (let y = Math.max(textY - bgHeight, 0); y < Math.min(textY, height); y++) {
            for (let x = Math.max(textX, 0); x < Math.min(textX + bgWidth, width); x++) {
                const idx = (y * width + x) * channels;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }
    }

    private getColorForClass(classId: number): [number, number, number] {
        const colors = [
            [255, 0, 0],    // Rojo
            [0, 255, 0],    // Verde
            [0, 0, 255],    // Azul
            [255, 255, 0],  // Amarillo
            [255, 0, 255],  // Magenta
            [0, 255, 255],  // Cyan
            [255, 128, 0],  // Naranja
            [128, 0, 255],  // Púrpura
        ];
        return colors[classId % colors.length] as [number, number, number];
    }
}
