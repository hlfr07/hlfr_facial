import { Controller, Post, UseInterceptors, UploadedFile, Res } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Response } from 'express';
import * as fs from 'fs';
import * as path from 'path';
import { YoloSegService } from '../services/yoloseg.service';

@Controller('api/segmentation')
export class SegmentationController {
    private readonly tempDir = path.join(process.cwd(), 'temp-uploads');

    constructor(private readonly yoloSegService: YoloSegService) {
        if (!fs.existsSync(this.tempDir)) {
            fs.mkdirSync(this.tempDir, { recursive: true });
        }
    }

    @Post('segment')
    @UseInterceptors(FileInterceptor('image'))
    async segmentObjects(@UploadedFile() file: Express.Multer.File, @Res() res: Response) {
        let tempPath: string | null = null;

        try {
            if (!file) {
                return res.status(400).json({ error: 'No image provided' });
            }

            // Guardar archivo temporalmente
            tempPath = path.join(this.tempDir, `${Date.now()}-${file.originalname}`);
            fs.writeFileSync(tempPath, file.buffer);

            // Detectar y segmentar objetos
            const { boxes, originalWidth, originalHeight } = await this.yoloSegService.segmentObjects(tempPath);

            // Dibujar boxes con máscaras en la imagen
            const imageWithMasks = await this.yoloSegService.drawBoxesWithMasks(tempPath, boxes);

            // Convertir a base64
            const imageBase64 = imageWithMasks.toString('base64');

            return res.json({
                success: true,
                boxes: boxes.map(b => ({
                    x: b.x,
                    y: b.y,
                    w: b.w,
                    h: b.h,
                    label: b.label,
                    confidence: parseFloat((b.confidence * 100).toFixed(2)),
                    classId: b.classId,
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
