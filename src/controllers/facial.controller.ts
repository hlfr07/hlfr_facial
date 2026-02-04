import {
  Controller,
  Post,
  Get,
  Delete,
  Body,
  UseInterceptors,
  UploadedFile,
  BadRequestException,
  Param,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Express } from 'express';
import * as path from 'path';
import * as fs from 'fs';
import { FacialRecognitionService } from '../services/facial-recognition.service';

@Controller('api/faces')
export class FacialController {
  private uploadsDir = path.join(__dirname, '..', '..', 'temp-uploads');

  constructor(private readonly facialService: FacialRecognitionService) {
    if (!fs.existsSync(this.uploadsDir)) {
      fs.mkdirSync(this.uploadsDir, { recursive: true });
    }
  }

  // @Get('stats')
  // getDatabaseStats() {
  //   return this.facialService.getDatabaseStats();
  // }

  // @Post('register')
  // @UseInterceptors(FileInterceptor('image'))
  // async registerFace(
  //   @UploadedFile() file: Express.Multer.File,
  //   @Body('name') name: string,
  // ) {
  //   if (!file) {
  //     throw new BadRequestException('No image provided');
  //   }

  //   if (!name) {
  //     throw new BadRequestException('Name is required');
  //   }

  //   // Validar imagen
  //   if (!file.mimetype.startsWith('image/')) {
  //     throw new BadRequestException('File must be an image');
  //   }

  //   try {
  //     // Guardar temporalmente
  //     const tempPath = path.join(this.uploadsDir, `${Date.now()}-${file.originalname}`);
  //     fs.writeFileSync(tempPath, file.buffer);

  //     // Registrar rostro
  //     const result = await this.facialService.registerFace(tempPath, name);

  //     // Limpiar archivo temporal (ignorar errores de Windows lock)
  //     try {
  //       fs.unlinkSync(tempPath);
  //     } catch (err) {
  //       console.warn('Warning: Could not delete temp file:', tempPath);
  //     }

  //     if (result.success) {
  //       return {
  //         success: true,
  //         message: `Face registered successfully for: ${name}`,
  //         processing: {
  //           detectionBox: result.detectionBox,
  //           croppedImageBase64: result.croppedImageBase64,
  //           embeddingSize: result.embedding?.length,
  //           logs: result.logs,
  //         },
  //       };
  //     } else {
  //       return {
  //         success: false,
  //         message: 'Failed to register face. No face detected in image.',
  //         logs: result.logs,
  //       };
  //     }
  //   } catch (error) {
  //     return {
  //       success: false,
  //       message: error.message || 'Error registering face',
  //     };
  //   }
  // }

  // @Post('verify')
  // @UseInterceptors(FileInterceptor('image'))
  // async verifyFace(@UploadedFile() file: Express.Multer.File) {
  //   if (!file) {
  //     throw new BadRequestException('No image provided');
  //   }

  //   if (!file.mimetype.startsWith('image/')) {
  //     throw new BadRequestException('File must be an image');
  //   }

  //   try {
  //     // Guardar temporalmente
  //     const tempPath = path.join(this.uploadsDir, `${Date.now()}-${file.originalname}`);
  //     fs.writeFileSync(tempPath, file.buffer);

  //     // Verificar rostro
  //     const result = await this.facialService.verifyFace(tempPath);

  //     // Limpiar archivo temporal (ignorar errores de Windows lock)
  //     try {
  //       fs.unlinkSync(tempPath);
  //     } catch (err) {
  //       console.warn('Warning: Could not delete temp file:', tempPath);
  //     }

  //     if (result.verified) {
  //       return {
  //         verified: true,
  //         name: result.name,
  //         confidence: result.confidence,
  //         message: `✓ Access granted to ${result.name} (${result.confidence}% confidence)`,
  //         processing: {
  //           detectionBox: result.detectionBox,
  //           croppedImageBase64: result.croppedImageBase64,
  //           embeddingSize: result.embedding?.length,
  //           logs: result.logs,
  //         },
  //       };
  //     } else {
  //       return {
  //         verified: false,
  //         message: '✗ Face not recognized. Access denied.',
  //         processing: {
  //           detectionBox: result.detectionBox,
  //           croppedImageBase64: result.croppedImageBase64,
  //           embeddingSize: result.embedding?.length,
  //           logs: result.logs,
  //         },
  //       };
  //     }
  //   } catch (error) {
  //     return {
  //       verified: false,
  //       message: error.message || 'Error verifying face',
  //     };
  //   }
  // }

  // @Get('registered')
  // getRegisteredFaces() {
  //   const faces = this.facialService.getAllRegisteredFaces();
  //   return {
  //     total: faces.length,
  //     faces,
  //   };
  // }

  // @Delete('registered/:name')
  // deleteRegisteredFace(@Param('name') name: string) {
  //   const success = this.facialService.deleteRegisteredFace(name);
  //   return {
  //     success,
  //     message: success ? `Face deleted: ${name}` : 'Face not found',
  //   };
  // }
}
