import { Controller, Post, UseInterceptors, UploadedFile, Res } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Response } from 'express';
import * as fs from 'fs';
import * as path from 'path';
import { YoloService } from '../services/yolo.service';

@Controller('api/detection')
export class DetectionController {
    private readonly tempDir = path.join(process.cwd(), 'temp-uploads');

    constructor(private readonly yoloService: YoloService) {
        if (!fs.existsSync(this.tempDir)) {
            fs.mkdirSync(this.tempDir, { recursive: true });
        }
    }

    @Post('detect')
    @UseInterceptors(FileInterceptor('image'))
    async detectObjects(@UploadedFile() file: Express.Multer.File, @Res() res: Response) {
        let tempPath: string | null = null;

        try {
            if (!file) {
                return res.status(400).json({ error: 'No image provided' });
            }

            // Guardar archivo temporalmente
            tempPath = path.join(this.tempDir, `${Date.now()}-${file.originalname}`);
            fs.writeFileSync(tempPath, file.buffer);

            // Detectar objetos
            const { boxes, originalWidth, originalHeight } = await this.yoloService.detectObjects(tempPath);

            // Dibujar boxes en la imagen
            const imageWithBoxes = await this.yoloService.drawBoxesOnImage(tempPath, boxes);

            // Convertir a base64
            const imageBase64 = imageWithBoxes.toString('base64');

            return res.json({
                success: true,
                boxes: boxes.map(b => ({
                    x1: b.x1,
                    y1: b.y1,
                    x2: b.x2,
                    y2: b.y2,
                    label: b.label,
                    confidence: parseFloat((b.confidence * 100).toFixed(2)),
                })),
                imageBase64,
                originalWidth,
                originalHeight,
            });
        } catch (error) {
            return res.status(500).json({
                error: error.message || 'Error processing image',
            });
        } finally {
            // Limpiar archivo temporal
            if (tempPath && fs.existsSync(tempPath)) {
                try {
                    fs.unlinkSync(tempPath);
                } catch (err) {
                    console.warn('Could not delete temp file:', tempPath);
                }
            }
        }
    }
}
