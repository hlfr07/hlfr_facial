import { Injectable, Logger } from '@nestjs/common';
import * as ort from 'onnxruntime-web';
import * as path from 'path';
import sharp from 'sharp';

export interface SegmentationBox {
    x: number;
    y: number;
    w: number;
    h: number;
    classId: number;
    label: string;
    confidence: number;
    mask: Buffer | null;
    maskCoefficients?: Float32Array;
}

class BoundingBoxSeg {
    xywh: [number, number, number, number];
    classId: number;
    confidence_score: number;
    mask_coefficients: Float32Array;
    mask: Buffer | null = null;

    constructor(
        x: number,
        y: number,
        w: number,
        h: number,
        classId: number,
        confidenceScore: number,
        maskCoefficients: Float32Array,
    ) {
        this.xywh = [x, y, w, h];
        this.classId = classId;
        this.confidence_score = confidenceScore;
        this.mask_coefficients = maskCoefficients;
    }
}

@Injectable()
export class YoloSegService {
    private readonly logger = new Logger(YoloSegService.name);
    private yoloSegSession: ort.InferenceSession | null = null;
    private readonly INPUT_SIZE = 640;
    private readonly CONF_THRESHOLD = 0.25;

    private readonly cocoClasses = [
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
            const modelPath = path.join(process.cwd(), 'src', 'models', 'yolo11n-seg-coco.onnx');
            this.logger.log(`🔍 Checking YOLOSeg model: ${modelPath}`);

            const fs = require('fs');
            if (!fs.existsSync(modelPath)) {
                this.logger.error('❌ YOLOSeg model not found!');
                return;
            }

            this.logger.log('⏳ Loading YOLOSeg model...');
            this.yoloSegSession = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['wasm'],
            });
            this.logger.log('✅ YOLOSeg model loaded successfully');
        } catch (error) {
            this.logger.error('❌ Error loading YOLOSeg model:', error);
        }
    }

    async segmentObjects(imagePath: string): Promise<{
        boxes: SegmentationBox[];
        originalWidth: number;
        originalHeight: number;
    }> {
        try {
            if (!this.yoloSegSession) {
                throw new Error('YOLOSeg model not initialized');
            }

            // Get original dimensions
            const metadata = await sharp(imagePath).metadata();
            const imgWidth = metadata.width!;
            const imgHeight = metadata.height!;

            this.logger.log(`📐 Original image size: ${imgWidth}x${imgHeight}`);

            // Preprocess: letterbox + resize
            const { resizedBuffer, padInfo } = await this.preprocessImage(imagePath);

            // Run inference
            const input = new ort.Tensor('float32', resizedBuffer, [1, 3, this.INPUT_SIZE, this.INPUT_SIZE]);
            const results = await this.yoloSegSession.run({ images: input });

            // Extract outputs
            const output0 = results['output0']; // Detections: [1, 116, 8400] (4 bbox + 80 classes + 32 mask)
            const output1 = results['output1']; // Masks: [1, 32, 160, 160]

            this.logger.log(`📊 Output0 shape: ${JSON.stringify(output0.dims)}`);
            this.logger.log(`📊 Output1 shape: ${JSON.stringify(output1.dims)}`);

            // Process detections with masks
            const boxes = this.processDetectionsWithMasks(
                output0.data as Float32Array,
                output0.dims as number[],
                output1.data as Float32Array,
                output1.dims as number[],
                imgWidth,
                imgHeight,
                padInfo,
            );

            return {
                boxes,
                originalWidth: imgWidth,
                originalHeight: imgHeight,
            };
        } catch (error) {
            this.logger.error('Error in segmentation:', error);
            throw error;
        }
    }

    private async preprocessImage(
        imagePath: string,
    ): Promise<{ resizedBuffer: Float32Array; padInfo: { padLeft: number; padTop: number; scale: number } }> {
        // Read image
        const image = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true });
        const { data, info } = image;

        const imgWidth = info.width;
        const imgHeight = info.height;

        // Letterbox resize
        const scale = Math.min(this.INPUT_SIZE / imgWidth, this.INPUT_SIZE / imgHeight);
        const newWidth = Math.round(imgWidth * scale);
        const newHeight = Math.round(imgHeight * scale);

        const padLeft = Math.round((this.INPUT_SIZE - newWidth) / 2);
        const padTop = Math.round((this.INPUT_SIZE - newHeight) / 2);

        // Resize image
        let resized = await sharp(imagePath)
            .resize(newWidth, newHeight, { fit: 'fill' })
            .removeAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });

        // Create padded image (640x640) with gray background (114,114,114)
        const paddedData = Buffer.alloc(this.INPUT_SIZE * this.INPUT_SIZE * 3);
        paddedData.fill(114); // Gray padding

        const resizedData = resized.data;
        for (let y = 0; y < newHeight; y++) {
            for (let x = 0; x < newWidth; x++) {
                const srcIdx = (y * newWidth + x) * 3;
                const dstIdx = ((padTop + y) * this.INPUT_SIZE + (padLeft + x)) * 3;
                paddedData[dstIdx] = resizedData[srcIdx];
                paddedData[dstIdx + 1] = resizedData[srcIdx + 1];
                paddedData[dstIdx + 2] = resizedData[srcIdx + 2];
            }
        }

        // Normalize: CHW format, [0,1]
        const inputTensor = new Float32Array(3 * this.INPUT_SIZE * this.INPUT_SIZE);
        for (let i = 0; i < this.INPUT_SIZE * this.INPUT_SIZE; i++) {
            const r = paddedData[i * 3] / 255.0;
            const g = paddedData[i * 3 + 1] / 255.0;
            const b = paddedData[i * 3 + 2] / 255.0;

            inputTensor[i] = r;
            inputTensor[this.INPUT_SIZE * this.INPUT_SIZE + i] = g;
            inputTensor[this.INPUT_SIZE * this.INPUT_SIZE * 2 + i] = b;
        }

        return {
            resizedBuffer: inputTensor,
            padInfo: { padLeft, padTop, scale },
        };
    }

    private processDetectionsWithMasks(
        detections: Float32Array,
        detDims: number[],
        masks: Float32Array,
        maskDims: number[],
        imgWidth: number,
        imgHeight: number,
        padInfo: { padLeft: number; padTop: number; scale: number },
    ): SegmentationBox[] {
        let boxes: SegmentationBox[] = [];

        // detDims: [1, 116, 8400] -> Transpose to [8400, 116]
        const numDetections = detDims[2]; // 8400
        const numChannels = detDims[1]; // 116

        this.logger.log(`🔍 Processing ${numDetections} detections with ${numChannels} channels...`);

        // Transpose: iterate over detections (8400)
        for (let d = 0; d < numDetections; d++) {
            // Extract bbox (xc, yc, w, h) - indices 0-3
            const xc = detections[0 * numDetections + d];
            const yc = detections[1 * numDetections + d];
            const w = detections[2 * numDetections + d];
            const h = detections[3 * numDetections + d];

            // Extract class scores (indices 4-83)
            let maxProb = 0;
            let classId = 0;

            for (let c = 4; c < 84; c++) {
                const prob = detections[c * numDetections + d];
                if (prob > maxProb) {
                    maxProb = prob;
                    classId = c - 4;
                }
            }

            const confidence = maxProb;

            // Filter by threshold
            if (confidence < this.CONF_THRESHOLD) {
                continue;
            }

            // Extract mask coefficients (indices 84-115, 32 values)
            const maskCoeffs = new Float32Array(32);
            for (let m = 0; m < 32; m++) {
                maskCoeffs[m] = detections[(84 + m) * numDetections + d];
            }

            // Convert from center format (xc, yc, w, h) to corner format (x1, y1, x2, y2)
            // Values are in 640x640 space
            const x1 = xc - w / 2;
            const y1 = yc - h / 2;
            const x2 = xc + w / 2;
            const y2 = yc + h / 2;

            if (w > 0 && h > 0) {
                boxes.push({
                    x: Math.max(0, Math.floor(x1)),
                    y: Math.max(0, Math.floor(y1)),
                    w: Math.floor(x2 - x1),
                    h: Math.floor(y2 - y1),
                    classId,
                    label: this.cocoClasses[classId] || `Class ${classId}`,
                    confidence,
                    mask: null,
                    maskCoefficients: maskCoeffs,
                });
            }
        }

        // Scale boxes back to original image size
        if (boxes.length > 0) {
            boxes = this.scaleBoxes(boxes, imgWidth, imgHeight, padInfo);
            boxes = this.processMasks(boxes, masks, maskDims, imgWidth, imgHeight);
        }

        this.logger.log(`✅ Found ${boxes.length} detections above ${this.CONF_THRESHOLD} threshold`);
        return boxes;
    }

    private scaleBoxes(
        boxes: SegmentationBox[],
        imgWidth: number,
        imgHeight: number,
        padInfo: { padLeft: number; padTop: number; scale: number },
    ): SegmentationBox[] {
        const { scale, padLeft: pad_x, padTop: pad_y } = padInfo;

        for (const box of boxes) {
            // Remove letterbox padding and scale to original image
            const x = (box.x - pad_x) / scale;
            const y = (box.y - pad_y) / scale;
            const w = box.w / scale;
            const h = box.h / scale;

            box.x = Math.max(0, Math.floor(x));
            box.y = Math.max(0, Math.floor(y));
            box.w = Math.floor(w);
            box.h = Math.floor(h);
        }

        return boxes;
    }

    private processMasks(
        boxes: SegmentationBox[],
        protoMasks: Float32Array,
        maskDims: number[],
        imgWidth: number,
        imgHeight: number,
    ): SegmentationBox[] {
        // maskDims: [1, 32, 160, 160]
        const maskChannels = maskDims[1]; // 32
        const maskHeight = maskDims[2]; // 160
        const maskWidth = maskDims[3]; // 160

        this.logger.log(`🎭 Processing ${boxes.length} masks (${maskWidth}x${maskHeight})...`);

        // Reshape proto masks: [1, 32, 160, 160] -> [32, 160*160]
        // Skip batch dimension and reshape to 2D
        const protoMasksReshaped = new Float32Array(maskChannels * maskHeight * maskWidth);
        for (let i = 0; i < protoMasks.length; i++) {
            protoMasksReshaped[i] = protoMasks[i];
        }

        for (const box of boxes) {
            if (!box.maskCoefficients) continue;

            // Reconstruct mask: coeff^T @ proto_masks = combined_mask
            // coeff: (32,), proto_masks: (32, H*W) -> result: (H*W,)
            const combinedMask = new Float32Array(maskHeight * maskWidth);

            for (let i = 0; i < maskHeight * maskWidth; i++) {
                let value = 0;
                for (let c = 0; c < maskChannels; c++) {
                    value += box.maskCoefficients[c] * protoMasksReshaped[c * maskHeight * maskWidth + i];
                }
                combinedMask[i] = value;
            }

            // Resize mask from 160x160 to 640x640, remove letterbox, then to original size
            const resizedMask = this.resizeMaskRemoveLetterbox(
                combinedMask,
                maskHeight,
                maskWidth,
                imgWidth,
                imgHeight,
            );

            // Extract box region from resized mask
            const boxMask = Buffer.alloc(box.w * box.h);
            let maskIdx = 0;

            for (let py = 0; py < box.h; py++) {
                for (let px = 0; px < box.w; px++) {
                    const imgY = box.y + py;
                    const imgX = box.x + px;

                    if (imgY >= 0 && imgY < imgHeight && imgX >= 0 && imgX < imgWidth) {
                        const resizedMaskIdx = imgY * imgWidth + imgX;
                        // Threshold at 0.5 and convert to 0-255
                        boxMask[maskIdx] = resizedMask[resizedMaskIdx] > 0.5 ? 255 : 0;
                    }
                    maskIdx++;
                }
            }

            box.mask = boxMask;
        }

        return boxes;
    }

    private resizeMaskRemoveLetterbox(
        mask: Float32Array,
        maskHeight: number,
        maskWidth: number,
        imgWidth: number,
        imgHeight: number,
    ): Float32Array {
        // Step 1: Resize from 160x160 to 640x640
        const resized640 = this.resizeImage(mask, maskHeight, maskWidth, this.INPUT_SIZE, this.INPUT_SIZE);

        // Step 2: Crop to remove letterbox padding
        const scale = Math.min(this.INPUT_SIZE / imgWidth, this.INPUT_SIZE / imgHeight);
        const unpadWidth = Math.round(imgWidth * scale);
        const unpadHeight = Math.round(imgHeight * scale);
        const padWidth = this.INPUT_SIZE - unpadWidth;
        const padHeight = this.INPUT_SIZE - unpadHeight;
        const padLeft = Math.floor(padWidth / 2);
        const padTop = Math.floor(padHeight / 2);

        const cropped = new Float32Array(unpadWidth * unpadHeight);
        for (let y = 0; y < unpadHeight; y++) {
            for (let x = 0; x < unpadWidth; x++) {
                const srcIdx = (padTop + y) * this.INPUT_SIZE + (padLeft + x);
                const dstIdx = y * unpadWidth + x;
                cropped[dstIdx] = resized640[srcIdx];
            }
        }

        // Step 3: Resize from unpadded size to original image size
        return this.resizeImage(cropped, unpadHeight, unpadWidth, imgHeight, imgWidth);
    }

    private resizeImage(
        src: Float32Array,
        srcHeight: number,
        srcWidth: number,
        dstHeight: number,
        dstWidth: number,
    ): Float32Array {
        // Bilinear interpolation resize
        const dst = new Float32Array(dstHeight * dstWidth);

        const scaleY = srcHeight / dstHeight;
        const scaleX = srcWidth / dstWidth;

        for (let y = 0; y < dstHeight; y++) {
            for (let x = 0; x < dstWidth; x++) {
                const srcX = (x + 0.5) * scaleX - 0.5;
                const srcY = (y + 0.5) * scaleY - 0.5;

                const x0 = Math.max(0, Math.floor(srcX));
                const x1 = Math.min(srcWidth - 1, Math.ceil(srcX));
                const y0 = Math.max(0, Math.floor(srcY));
                const y1 = Math.min(srcHeight - 1, Math.ceil(srcY));

                const wx1 = srcX - x0;
                const wx0 = 1 - wx1;
                const wy1 = srcY - y0;
                const wy0 = 1 - wy1;

                const v00 = src[y0 * srcWidth + x0] || 0;
                const v10 = src[y0 * srcWidth + x1] || 0;
                const v01 = src[y1 * srcWidth + x0] || 0;
                const v11 = src[y1 * srcWidth + x1] || 0;

                dst[y * dstWidth + x] = wy0 * (wx0 * v00 + wx1 * v10) + wy1 * (wx0 * v01 + wx1 * v11);
            }
        }

        return dst;
    }

    async drawBoxesWithMasks(
        imagePath: string,
        boxes: SegmentationBox[],
    ): Promise<Buffer> {
        try {
            let img = await sharp(imagePath).raw().toBuffer({ resolveWithObject: true });
            const { data, info } = img;

            const width = info.width;
            const height = info.height;
            const channels = info.channels;

            // Draw masks first (lower layer)
            for (const box of boxes) {
                if (box.mask) {
                    this.drawMask(data, width, height, channels, box);
                }
            }

            // Draw boxes on top
            for (const box of boxes) {
                const color = this.getColorForClass(box.classId);
                this.drawRectangle(data, width, height, channels, box, color);
                this.drawText(data, width, height, channels, box, color);
            }

            const output = await sharp(Buffer.from(data), {
                raw: { width, height, channels: 3 },
            })
                .png()
                .toBuffer();

            return output;
        } catch (error) {
            this.logger.error('Error drawing boxes with masks:', error);
            throw error;
        }
    }

    private drawMask(
        data: Buffer,
        width: number,
        height: number,
        channels: number,
        box: SegmentationBox,
    ) {
        const { x, y, w, h, mask } = box;

        if (!mask || w <= 0 || h <= 0) return;

        const alpha = 0.5;
        const [r, g, b] = this.getColorForClass(box.classId);

        for (let py = 0; py < h && y + py < height; py++) {
            for (let px = 0; px < w && x + px < width; px++) {
                const maskIdx = py * w + px;
                if (maskIdx < mask.length && mask[maskIdx] > 128) {
                    const imgIdx = ((y + py) * width + (x + px)) * channels;
                    data[imgIdx] = Math.round(data[imgIdx] * (1 - alpha) + r * alpha);
                    data[imgIdx + 1] = Math.round(data[imgIdx + 1] * (1 - alpha) + g * alpha);
                    data[imgIdx + 2] = Math.round(data[imgIdx + 2] * (1 - alpha) + b * alpha);
                }
            }
        }
    }

    private drawRectangle(
        data: Buffer,
        width: number,
        height: number,
        channels: number,
        box: SegmentationBox,
        color: [number, number, number],
    ) {
        const thickness = 2;
        const [r, g, b] = color;

        const { x, y, w, h } = box;

        // Top line
        for (let px = 0; px < w && x + px < width; px++) {
            for (let t = 0; t < thickness && y + t < height; t++) {
                const idx = ((y + t) * width + (x + px)) * channels;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        // Bottom line
        for (let px = 0; px < w && x + px < width; px++) {
            for (let t = 0; t < thickness; t++) {
                const lineY = Math.max(0, y + h - 1 - t);
                if (lineY < height) {
                    const idx = (lineY * width + (x + px)) * channels;
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                }
            }
        }

        // Left line
        for (let py = 0; py < h && y + py < height; py++) {
            for (let t = 0; t < thickness && x + t < width; t++) {
                const idx = ((y + py) * width + (x + t)) * channels;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }

        // Right line
        for (let py = 0; py < h && y + py < height; py++) {
            for (let t = 0; t < thickness; t++) {
                const lineX = Math.max(0, x + w - 1 - t);
                if (lineX < width) {
                    const idx = ((y + py) * width + lineX) * channels;
                    data[idx] = r;
                    data[idx + 1] = g;
                    data[idx + 2] = b;
                }
            }
        }
    }

    private drawText(
        data: Buffer,
        width: number,
        height: number,
        channels: number,
        box: SegmentationBox,
        color: [number, number, number],
    ) {
        const text = `${box.label} ${(box.confidence * 100).toFixed(1)}%`;
        const textX = box.x + 5;
        const textY = box.y - 5;

        const bgWidth = text.length * 6;
        const bgHeight = 15;

        // Draw white background for text
        for (let py = Math.max(textY - bgHeight, 0); py < Math.min(textY, height); py++) {
            for (let px = Math.max(textX, 0); px < Math.min(textX + bgWidth, width); px++) {
                const idx = (py * width + px) * channels;
                data[idx] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }
    }

    private getColorForClass(classId: number): [number, number, number] {
        const colors = [
            [255, 0, 0], // Red
            [0, 255, 0], // Green
            [0, 0, 255], // Blue
            [255, 255, 0], // Yellow
            [255, 0, 255], // Magenta
            [0, 255, 255], // Cyan
            [255, 128, 0], // Orange
            [128, 0, 255], // Purple
        ];
        return colors[classId % colors.length] as [number, number, number];
    }
    
}

