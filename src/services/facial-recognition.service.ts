import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs';
import * as ort from 'onnxruntime-web';
import * as path from 'path';
import sharp from 'sharp';

export interface FaceEmbedding {
    name: string;
    embedding: number[];
    timestamp: string;
    imageBase64?: string;
}

export interface FaceRecord {
    name: string;
    embedding: number[];
    timestamp: string;
    imageBase64: string;
    imageSize: number;
    metadata: {
        confidence?: number;
        detectionCount?: number;
        detectionBox?: { x: number; y: number; w: number; h: number };
        croppedImageBase64?: string;
    };
}

export interface ProcessingResult {
    success: boolean;
    embedding?: number[];
    detectionBox?: { x: number; y: number; w: number; h: number };
    croppedImageBase64?: string;
    originalSize?: { width: number; height: number };
    logs: string[];
}

@Injectable()
export class FacialRecognitionService {
    // private readonly logger = new Logger(FacialRecognitionService.name);
    // private scrfdSession: ort.InferenceSession | null = null;
    // private arcfaceSession: ort.InferenceSession | null = null;
    // private databaseDir = path.join(__dirname, '..', 'database');
    // private embeddingsPath = path.join(this.databaseDir, '_embeddings.json');
    // private roiDir = path.join(__dirname, '..', 'roi'); // 📁 Carpeta para rostros recortados

    // constructor() {
    //     this.initializeDatabase();
    //     this.initializeModels();
    // }

    // private initializeDatabase() {
    //     if (!fs.existsSync(this.databaseDir)) {
    //         fs.mkdirSync(this.databaseDir, { recursive: true });
    //         this.logger.log('Database directory created');
    //     }
    //     if (!fs.existsSync(this.roiDir)) {
    //         fs.mkdirSync(this.roiDir, { recursive: true });
    //         this.logger.log('📁 ROI directory created for debugging');
    //     }
    // }

    // private async initializeModels() {
    //     try {
    //         console.log(path.join(process.cwd()));
    //         const scrfdPath = path.join(process.cwd(), 'src', 'models', 'scrfd_2.5g_bnkps.onnx');
    //         const arcfacePath = path.join(process.cwd(), 'src', 'models', 'arcface.onnx');

    //         this.logger.log('🔍 Checking model files...');
    //         this.logger.log(`  SCRFD path: ${scrfdPath}`);
    //         this.logger.log(`  ArcFace path: ${arcfacePath}`);

    //         const scrfdExists = fs.existsSync(scrfdPath);
    //         const arcfaceExists = fs.existsSync(arcfacePath);

    //         this.logger.log(`  SCRFD exists: ${scrfdExists}`);
    //         this.logger.log(`  ArcFace exists: ${arcfaceExists}`);

    //         if (!scrfdExists || !arcfaceExists) {
    //             this.logger.error('❌ Model files not found! System will not work.');
    //             return;
    //         }

    //         this.logger.log('⏳ Loading SCRFD model...');
    //         this.scrfdSession = await ort.InferenceSession.create(scrfdPath, {
    //             executionProviders: ['wasm'] // wasm
    //         });
    //         this.logger.log('✅ SCRFD model loaded');

    //         this.logger.log('⏳ Loading ArcFace model...');
    //         this.arcfaceSession = await ort.InferenceSession.create(arcfacePath, {
    //             executionProviders: ['wasm'] // CPU
    //         });
    //         this.logger.log('✅ ArcFace model loaded');

    //         // 🔍 LOG: ArcFace input/output names
    //         this.logger.log('📊 ArcFace Input Names:', this.arcfaceSession.inputNames);
    //         this.logger.log('📊 ArcFace Output Names:', this.arcfaceSession.outputNames);

    //         this.logger.log('🎉 ALL ONNX models loaded successfully');
    //     } catch (error) {
    //         this.logger.error('❌ Error loading ONNX models:', error);
    //     }
    // }

    // private readFaceRecord(name: string): FaceRecord | null {
    //     const filePath = path.join(this.databaseDir, `${name}.json`);
    //     if (fs.existsSync(filePath)) {
    //         try {
    //             const data = fs.readFileSync(filePath, 'utf-8');
    //             return JSON.parse(data);
    //         } catch (error) {
    //             this.logger.error(`Error reading face record for ${name}:`, error);
    //             return null;
    //         }
    //     }
    //     return null;
    // }

    // private saveFaceRecord(record: FaceRecord): void {
    //     const filePath = path.join(this.databaseDir, `${record.name}.json`);
    //     try {
    //         fs.writeFileSync(filePath, JSON.stringify(record, null, 2));
    //         this.logger.log(`Face record saved: ${record.name}.json`);
    //     } catch (error) {
    //         this.logger.error(`Error saving face record for ${record.name}:`, error);
    //     }
    // }

    // private readEmbeddingsIndex(): FaceEmbedding[] {
    //     if (fs.existsSync(this.embeddingsPath)) {
    //         try {
    //             const data = fs.readFileSync(this.embeddingsPath, 'utf-8');
    //             return JSON.parse(data);
    //         } catch (error) {
    //             this.logger.error('Error reading embeddings index:', error);
    //             return [];
    //         }
    //     }
    //     return [];
    // }

    // private updateEmbeddingsIndex(): void {
    //     const files = fs.readdirSync(this.databaseDir);
    //     const embeddings: FaceEmbedding[] = [];

    //     files.forEach((file) => {
    //         if (file.endsWith('.json') && file !== '_embeddings.json') {
    //             const record = this.readFaceRecord(file.replace('.json', ''));
    //             if (record) {
    //                 embeddings.push({
    //                     name: record.name,
    //                     embedding: record.embedding,
    //                     timestamp: record.timestamp,
    //                 });
    //             }
    //         }
    //     });

    //     try {
    //         fs.writeFileSync(this.embeddingsPath, JSON.stringify(embeddings, null, 2));
    //         this.logger.log('Embeddings index updated');
    //     } catch (error) {
    //         this.logger.error('Error updating embeddings index:', error);
    //     }
    // }

    // private async detectFace(imagePath: string): Promise<{
    //     x: number;
    //     y: number;
    //     w: number;
    //     h: number;
    // } | null> {
    //     try {
    //         if (!this.scrfdSession) {
    //             this.logger.error('❌ SCRFD model not loaded!');
    //             return null;
    //         }

    //         // Lee metadata original
    //         const metadata = await sharp(imagePath).metadata();
    //         const origWidth = metadata.width!;
    //         const origHeight = metadata.height!;

    //         // Redimensiona directamente a 320x320 (SIN padding)
    //         const inputSize = 320;
    //         const resizedBuffer = await sharp(imagePath)
    //             .resize(inputSize, inputSize, { fit: 'fill' })
    //             .raw()
    //             .toBuffer({ resolveWithObject: true });

    //         const { data } = resizedBuffer;

    //         // Normalización SCRFD: (img - 127.5) / 128.0
    //         const float32Data = new Float32Array(data.length);
    //         for (let i = 0; i < data.length; i++) {
    //             float32Data[i] = (data[i] - 127.5) / 128.0;
    //         }

    //         // CHW format
    //         const rgbData = new Float32Array(3 * inputSize * inputSize);
    //         for (let i = 0; i < inputSize * inputSize; i++) {
    //             rgbData[i] = float32Data[i * 3];
    //             rgbData[inputSize * inputSize + i] = float32Data[i * 3 + 1];
    //             rgbData[inputSize * inputSize * 2 + i] = float32Data[i * 3 + 2];
    //         }

    //         // Inferencia
    //         const inputTensor = new ort.Tensor('float32', rgbData, [1, 3, inputSize, inputSize]);
    //         const results = await this.scrfdSession.run({ 'input.1': inputTensor });

    //         // 🔍 DEBUG: Ver nombres y formas de salidas
    //         this.logger.log('\n📊 === OUTPUT NAMES AND SHAPES ===');
    //         const outputNames = Object.keys(results);
    //         const outputList: Array<{ name: string, dims: number[] }> = [];
    //         outputNames.forEach(key => {
    //             const tensor = results[key];
    //             outputList.push({ name: key, dims: tensor.dims as number[] });
    //             this.logger.log(`  ${key}: shape=${JSON.stringify(tensor.dims)}`);
    //         });

    //         // Parámetros (idéntico a código Python)
    //         const confThreshold = 0.6;
    //         const nmsThreshold = 0.4;
    //         const numAnchors = 2;
    //         const featStrides = [8, 16, 32];
    //         const fmc = 3;

    //         // Los outputs están en orden: score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32
    //         // Extraer índices de scores y bboxes (cada 3 outputs es un grupo)
    //         const scoreIndices: number[] = [];
    //         const bboxIndices: number[] = [];

    //         for (let i = 0; i < outputList.length; i += 3) {
    //             scoreIndices.push(i);
    //             bboxIndices.push(i + 1);
    //         }

    //         this.logger.log(`\n  Score indices: ${scoreIndices.join(', ')}`);
    //         this.logger.log(`  BBox indices: ${bboxIndices.join(', ')}`);

    //         if (scoreIndices.length === 0 || bboxIndices.length === 0) {
    //             this.logger.error('❌ No score or bbox outputs found!');
    //             this.logger.log('  Available outputs:', outputNames.join(', '));
    //             return null;
    //         }

    //         // Función: generar anchors
    //         const generateAnchors = (height: number, width: number, stride: number): number[][] => {
    //             const centers: number[][] = [];
    //             for (let y = 0; y < height; y++) {
    //                 for (let x = 0; x < width; x++) {
    //                     const cx = x * stride;
    //                     const cy = y * stride;
    //                     // 2 anchors por posición
    //                     for (let a = 0; a < numAnchors; a++) {
    //                         centers.push([cx, cy]);
    //                     }
    //                 }
    //             }
    //             return centers;
    //         };

    //         // Función: distance2bbox
    //         const distance2bbox = (points: number[][], distance: number[][]): number[][] => {
    //             const boxes: number[][] = [];
    //             for (let i = 0; i < points.length; i++) {
    //                 const [px, py] = points[i];
    //                 const [dx, dy, dw, dh] = distance[i];
    //                 boxes.push([px - dx, py - dy, px + dw, py + dh]);
    //             }
    //             return boxes;
    //         };

    //         // Función: NMS
    //         const nms = (detections: number[][]): number[] => {
    //             if (detections.length === 0) return [];

    //             const x1 = detections.map(d => d[0]);
    //             const y1 = detections.map(d => d[1]);
    //             const x2 = detections.map(d => d[2]);
    //             const y2 = detections.map(d => d[3]);
    //             const scores = detections.map(d => d[4]);

    //             const areas = detections.map((_, i) => (x2[i] - x1[i]) * (y2[i] - y1[i]));

    //             let order = scores
    //                 .map((_, i) => i)
    //                 .sort((i, j) => scores[j] - scores[i]);

    //             const keep: number[] = [];

    //             while (order.length > 0) {
    //                 const i = order[0];
    //                 keep.push(i);

    //                 if (order.length === 1) break;

    //                 const others = order.slice(1);
    //                 const xx1 = others.map(j => Math.max(x1[i], x1[j]));
    //                 const yy1 = others.map(j => Math.max(y1[i], y1[j]));
    //                 const xx2 = others.map(j => Math.min(x2[i], x2[j]));
    //                 const yy2 = others.map(j => Math.min(y2[i], y2[j]));

    //                 const w = xx2.map((x, idx) => Math.max(0, x - xx1[idx]));
    //                 const h = yy2.map((y, idx) => Math.max(0, y - yy1[idx]));
    //                 const inter = w.map((width, idx) => width * h[idx]);

    //                 const iou = inter.map((inter_area, idx) => {
    //                     const union = areas[i] + areas[others[idx]] - inter_area;
    //                     return union > 0 ? inter_area / union : 0;
    //                 });

    //                 order = others.filter((_, idx) => iou[idx] <= nmsThreshold);
    //             }

    //             return keep;
    //         };

    //         // Procesar cada escala
    //         const detections: number[][] = [];

    //         for (let idx = 0; idx < scoreIndices.length; idx++) {
    //             const scoreIdx = scoreIndices[idx];
    //             const bboxIdx = bboxIndices[idx];

    //             const scoreKey = outputNames[scoreIdx];
    //             const bboxKey = outputNames[bboxIdx];

    //             const scores = results[scoreKey].data as Float32Array;
    //             const bboxRaw = results[bboxKey].data as Float32Array;

    //             // Calcular stride basado en el número de elementos
    //             // 40*40*2 = 3200 (stride 8)
    //             // 20*20*2 = 800 (stride 16)
    //             // 10*10*2 = 200 (stride 32)
    //             const numElements = scores.length;
    //             let stride = 8;
    //             if (numElements === 800) stride = 16;
    //             else if (numElements === 200) stride = 32;

    //             this.logger.log(`\n  📍 Scale ${idx} (output[${scoreIdx}], stride=${stride}):`);
    //             this.logger.log(`    Scores: ${scores.length} elements, min=${Math.min(...scores).toFixed(4)}, max=${Math.max(...scores).toFixed(4)}`);
    //             this.logger.log(`    BBox: ${bboxRaw.length} elements`);

    //             const featH = inputSize / stride;
    //             const featW = inputSize / stride;

    //             // Generar anchors
    //             const centers = generateAnchors(featH, featW, stride);
    //             this.logger.log(`    Centers generated: ${centers.length}`);

    //             // Procesar bbox_preds: multiplicar por stride PRIMERO
    //             const bboxPreds: number[][] = [];
    //             for (let i = 0; i < bboxRaw.length; i += 4) {
    //                 bboxPreds.push([
    //                     bboxRaw[i] * stride,
    //                     bboxRaw[i + 1] * stride,
    //                     bboxRaw[i + 2] * stride,
    //                     bboxRaw[i + 3] * stride,
    //                 ]);
    //             }

    //             // Filtrar por confianza
    //             const validIndices: number[] = [];
    //             for (let i = 0; i < scores.length; i++) {
    //                 if (scores[i] >= confThreshold) {
    //                     validIndices.push(i);
    //                 }
    //             }

    //             this.logger.log(`    Scores >= ${confThreshold}: ${validIndices.length}`);

    //             if (validIndices.length === 0) continue;

    //             // Filtrar
    //             const validCenters = validIndices.map(i => centers[i]);
    //             const validBboxes = validIndices.map(i => bboxPreds[i]);
    //             const validScores = validIndices.map(i => scores[i]);

    //             // Decodificar bboxes
    //             const boxes = distance2bbox(validCenters, validBboxes);

    //             // ESCALAR A IMAGEN ORIGINAL (como en Python)
    //             for (let i = 0; i < boxes.length; i++) {
    //                 const [x1, y1, x2, y2] = boxes[i];
    //                 const scaledX1 = x1 * origWidth / inputSize;
    //                 const scaledY1 = y1 * origHeight / inputSize;
    //                 const scaledX2 = x2 * origWidth / inputSize;
    //                 const scaledY2 = y2 * origHeight / inputSize;

    //                 detections.push([scaledX1, scaledY1, scaledX2, scaledY2, validScores[i]]);
    //             }

    //             this.logger.log(`    Added ${boxes.length} detections`);
    //         }

    //         if (detections.length === 0) {
    //             this.logger.warn('No face detected');
    //             return null;
    //         }

    //         // Aplicar NMS
    //         const keepIndices = nms(detections);
    //         const kept = keepIndices.map(i => detections[i]);

    //         if (kept.length === 0) {
    //             this.logger.warn('No face after NMS');
    //             return null;
    //         }

    //         // Ordenar por score y tomar el mejor
    //         kept.sort((a, b) => b[4] - a[4]);
    //         const [x1, y1, x2, y2, score] = kept[0];

    //         // Convertir a [x, y, w, h] y clampear
    //         let x = Math.floor(x1);
    //         let y = Math.floor(y1);
    //         let w = Math.floor(x2 - x1);
    //         let h = Math.floor(y2 - y1);

    //         x = Math.max(0, x);
    //         y = Math.max(0, y);
    //         w = Math.min(w, origWidth - x);
    //         h = Math.min(h, origHeight - y);

    //         this.logger.log(`Face detected: score=${score.toFixed(4)}, box=[${x}, ${y}, ${w}, ${h}]`);

    //         return { x, y, w, h };
    //     } catch (error) {
    //         this.logger.error('Error in face detection:', error);
    //         return null;
    //     }
    // }

    // private async generateEmbedding(imagePath: string, box?: any): Promise<ProcessingResult> {
    //     const logs: string[] = [];
    //     try {
    //         logs.push('\n🚀 === STARTING FACE PROCESSING ===');

    //         logs.push('📍 Step 1: Face Detection');
    //         const detectionResult = box || await this.detectFace(imagePath);
    //         if (!detectionResult) {
    //             logs.push('❌ No face detected');
    //             logs.forEach(log => this.logger.log(log));
    //             return { success: false, logs };
    //         }

    //         // LEER METADATA ORIGINAL
    //         logs.push('\n📍 Step 2: Image Processing');
    //         const metadata = await sharp(imagePath).metadata();
    //         const imgWidth = metadata.width!;
    //         const imgHeight = metadata.height!;
    //         logs.push(`📐 Original size: ${imgWidth}x${imgHeight}`);

    //         // Ya está escalado directamente desde detectFace
    //         const scaledBox = {
    //             x: Math.max(0, detectionResult.x),
    //             y: Math.max(0, detectionResult.y),
    //             w: Math.min(detectionResult.w, imgWidth - detectionResult.x),
    //             h: Math.min(detectionResult.h, imgHeight - detectionResult.y),
    //         };

    //         // Validar
    //         if (scaledBox.w < 10 || scaledBox.h < 10) {
    //             logs.push(`❌ Invalid box dimensions: w=${scaledBox.w}, h=${scaledBox.h}`);
    //             logs.forEach(log => this.logger.log(log));
    //             return { success: false, logs };
    //         }

    //         logs.push(`✅ Box validated: [${scaledBox.x}, ${scaledBox.y}, ${scaledBox.w}, ${scaledBox.h}]`);

    //         // Step 3: Crop face
    //         logs.push('\n📍 Step 3: Cropping Face');
    //         const croppedBuffer = await sharp(imagePath)
    //             .extract({
    //                 left: scaledBox.x,
    //                 top: scaledBox.y,
    //                 width: scaledBox.w,
    //                 height: scaledBox.h,
    //             })
    //             .resize(112, 112, { fit: 'cover' })
    //             .toBuffer();

    //         // 💾 GUARDAR ROI para debugging
    //         const timestamp = Date.now();
    //         const roiPath = path.join(this.roiDir, `roi_${timestamp}.jpg`);
    //         await sharp(croppedBuffer).jpeg({ quality: 95 }).toFile(roiPath);
    //         logs.push(`💾 ROI saved: ${roiPath}`);

    //         const croppedBase64 = croppedBuffer.toString('base64');
    //         logs.push(`✅ Cropped to 112x112 (${croppedBuffer.length} bytes)`);

    //         // Step 4: Generate embedding
    //         logs.push('\n📍 Step 4: Generating Embedding');
    //         if (!this.arcfaceSession) {
    //             logs.push('❌ ArcFace model not loaded!');
    //             logs.forEach(log => this.logger.log(log));
    //             return { success: false, logs };
    //         }

    //         const { data, info } = await sharp(croppedBuffer)
    //             .raw()
    //             .toBuffer({ resolveWithObject: true });

    //         // Normalize for ArcFace: (img - 127.5) / 128.0
    //         // arcface.onnx espera formato NHWC: [1, 112, 112, 3]
    //         const float32Data = new Float32Array(1 * 112 * 112 * 3);
    //         for (let i = 0; i < data.length; i++) {
    //             float32Data[i] = (data[i] - 127.5) / 128.0;
    //         }

    //         logs.push('🧠 Running ArcFace model...');
    //         const inputTensor = new ort.Tensor('float32', float32Data, [1, 112, 112, 3]);

    //         const arcfaceInputName = this.arcfaceSession.inputNames[0];
    //         logs.push(`📊 Using input name: '${arcfaceInputName}'`);

    //         const results = await this.arcfaceSession.run({ [arcfaceInputName]: inputTensor });

    //         const embeddingTensor = results[Object.keys(results)[0]];
    //         const embedding = Array.from(embeddingTensor.data as Float32Array);

    //         logs.push(`✅ Embedding generated: ${embedding.length} dimensions`);
    //         logs.push(`📊 Embedding range: [${Math.min(...embedding).toFixed(4)}, ${Math.max(...embedding).toFixed(4)}]`);
    //         logs.push('\n🎉 === PROCESSING COMPLETE ===\n');

    //         logs.forEach(log => this.logger.log(log));

    //         return {
    //             success: true,
    //             embedding,
    //             detectionBox: scaledBox,
    //             croppedImageBase64: croppedBase64,
    //             originalSize: { width: imgWidth, height: imgHeight },
    //             logs,
    //         };
    //     } catch (error) {
    //         logs.push(`❌ Error: ${error.message}`);
    //         this.logger.error('Error generating embedding:', error);
    //         logs.forEach(log => this.logger.log(log));
    //         return { success: false, logs };
    //     }
    // }

    // private cosineSimilarity(a: number[], b: number[]): number {
    //     if (a.length !== b.length || a.length === 0) return 0;

    //     const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    //     const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    //     const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

    //     if (magnitudeA === 0 || magnitudeB === 0) return 0;
    //     return dotProduct / (magnitudeA * magnitudeB);
    // }

    // async registerFace(imagePath: string, name: string): Promise<ProcessingResult & { success: boolean }> {
    //     try {
    //         this.logger.log(`\n\n${'='.repeat(60)}`);
    //         this.logger.log(`🔐 REGISTER FACE: ${name}`);
    //         this.logger.log('='.repeat(60));

    //         if (!fs.existsSync(imagePath)) {
    //             throw new Error('Image file not found');
    //         }

    //         if (this.readFaceRecord(name)) {
    //             throw new Error(`Face with name "${name}" already exists`);
    //         }

    //         const result = await this.generateEmbedding(imagePath);
    //         if (!result.success || !result.embedding) {
    //             return { ...result, success: false };
    //         }

    //         const imageBuffer = fs.readFileSync(imagePath);
    //         const imageBase64 = imageBuffer.toString('base64');

    //         const record: FaceRecord = {
    //             name,
    //             embedding: result.embedding,
    //             timestamp: new Date().toISOString(),
    //             imageBase64,
    //             imageSize: imageBuffer.length,
    //             metadata: {
    //                 confidence: 0,
    //                 detectionCount: 1,
    //                 detectionBox: result.detectionBox,
    //                 croppedImageBase64: result.croppedImageBase64,
    //             },
    //         };

    //         this.saveFaceRecord(record);
    //         this.updateEmbeddingsIndex();

    //         this.logger.log(`\n✅ Face registered: ${name}.json`);
    //         this.logger.log('='.repeat(60) + '\n');
    //         return { ...result, success: true };
    //     } catch (error) {
    //         this.logger.error(`❌ Error registering face for ${name}:`, error);
    //         return { success: false, logs: [error.message] };
    //     }
    // }

    // async verifyFace(imagePath: string): Promise<ProcessingResult & { verified: boolean; name?: string; confidence?: number }> {
    //     try {
    //         this.logger.log(`\n\n${'='.repeat(60)}`);
    //         this.logger.log('🔍 VERIFY FACE');
    //         this.logger.log('='.repeat(60));

    //         if (!fs.existsSync(imagePath)) {
    //             return { verified: false, success: false, logs: ['Image not found'] };
    //         }

    //         const result = await this.generateEmbedding(imagePath);
    //         if (!result.success || !result.embedding) {
    //             return { ...result, verified: false };
    //         }

    //         this.logger.log('\n📊 Comparing with registered faces...');
    //         const embeddings = this.readEmbeddingsIndex();
    //         this.logger.log(`Found ${embeddings.length} registered faces`);

    //         let bestMatch = { name: '', similarity: 0 };
    //         const threshold = 0.6;

    //         for (const stored of embeddings) {
    //             const similarity = this.cosineSimilarity(result.embedding, stored.embedding);
    //             this.logger.log(`  - ${stored.name}: ${(similarity * 100).toFixed(2)}%`);
    //             if (similarity > bestMatch.similarity) {
    //                 bestMatch = { name: stored.name, similarity };
    //             }
    //         }

    //         if (bestMatch.similarity > threshold) {
    //             this.logger.log(`\n✅ MATCH FOUND: ${bestMatch.name} (${(bestMatch.similarity * 100).toFixed(2)}%)`);
    //             this.logger.log('='.repeat(60) + '\n');
    //             return {
    //                 ...result,
    //                 verified: true,
    //                 name: bestMatch.name,
    //                 confidence: Math.round(bestMatch.similarity * 100),
    //             };
    //         }

    //         this.logger.log('\n❌ No match found (best: ' + (bestMatch.similarity * 100).toFixed(2) + '%)');
    //         this.logger.log('='.repeat(60) + '\n');
    //         return { ...result, verified: false };
    //     } catch (error) {
    //         this.logger.error('❌ Error verifying face:', error);
    //         return { verified: false, success: false, logs: [error.message] };
    //     }
    // }

    // getAllRegisteredFaces(): FaceEmbedding[] {
    //     const files = fs.readdirSync(this.databaseDir);
    //     const faces: FaceEmbedding[] = [];

    //     files.forEach((file) => {
    //         if (file.endsWith('.json') && file !== '_embeddings.json') {
    //             const record = this.readFaceRecord(file.replace('.json', ''));
    //             if (record) {
    //                 faces.push({
    //                     name: record.name,
    //                     embedding: [],
    //                     timestamp: record.timestamp,
    //                 });
    //             }
    //         }
    //     });

    //     return faces;
    // }

    // getFaceRecord(name: string): FaceRecord | null {
    //     return this.readFaceRecord(name);
    // }

    // deleteRegisteredFace(name: string): boolean {
    //     const filePath = path.join(this.databaseDir, `${name}.json`);
    //     if (fs.existsSync(filePath)) {
    //         try {
    //             fs.unlinkSync(filePath);
    //             this.logger.log(`Face deleted: ${name}.json`);
    //             this.updateEmbeddingsIndex();
    //             return true;
    //         } catch (error) {
    //             this.logger.error(`Error deleting face ${name}:`, error);
    //             return false;
    //         }
    //     }

    //     return false;
    // }

    // getDatabaseStats() {
    //     const files = fs.readdirSync(this.databaseDir);
    //     const jsonFiles = files.filter((f) => f.endsWith('.json') && f !== '_embeddings.json');

    //     let totalSize = 0;
    //     jsonFiles.forEach((file) => {
    //         const filePath = path.join(this.databaseDir, file);
    //         const stats = fs.statSync(filePath);
    //         totalSize += stats.size;
    //     });

    //     return {
    //         totalFaces: jsonFiles.length,
    //         databasePath: this.databaseDir,
    //         totalSize: `${(totalSize / 1024).toFixed(2)} KB`,
    //         files: jsonFiles.map((f) => f.replace('.json', '')),
    //     };
    // }
}


