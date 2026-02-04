import {
  Controller,
  Post,
  Get,
  UseInterceptors,
  UploadedFile,
  BadRequestException,
  Param,
  Delete,
  Res,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import type { Response } from 'express';
import type { Express } from 'express';
import { ImageService } from '../services/image.service';
import { DatabaseService } from '../services/database.service';

@Controller('api')
export class UploadController {
  constructor(
    private imageService: ImageService,
    private databaseService: DatabaseService,
  ) {}

  @Post('upload')
  @UseInterceptors(FileInterceptor('image'))
  uploadImage(@UploadedFile() file: Express.Multer.File) {
    if (!file) {
      throw new BadRequestException('No file uploaded');
    }

    // Validar que sea una imagen
    const allowedMimes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!allowedMimes.includes(file.mimetype)) {
      throw new BadRequestException('Only JPEG and PNG images are allowed');
    }

    const imageRecord = this.imageService.saveImage(file);
    return {
      message: 'Image uploaded successfully',
      data: imageRecord,
    };
  }

  @Get('images')
  getAllImages() {
    const images = this.databaseService.getAllImages();
    return {
      total: images.length,
      images,
    };
  }

  @Get('images/:id')
  getImageById(@Param('id') id: string) {
    const image = this.databaseService.getImageById(id);
    if (!image) {
      throw new BadRequestException('Image not found');
    }
    return image;
  }

  @Get('image/:filename')
  getImageFile(@Param('filename') filename: string, @Res() res: Response) {
    const imageBuffer = this.imageService.getImage(filename);
    if (!imageBuffer) {
      throw new BadRequestException('Image file not found');
    }

    // Determinar el tipo MIME basado en la extensión
    const ext = filename.split('.').pop()?.toLowerCase();
    const mimeType = ext === 'png' ? 'image/png' : 'image/jpeg';

    res.set('Content-Type', mimeType);
    res.send(imageBuffer);
  }

  @Delete('images/:id')
  deleteImage(@Param('id') id: string) {
    const image = this.databaseService.getImageById(id);
    if (!image) {
      throw new BadRequestException('Image not found');
    }

    this.imageService.deleteImage(image.filename);
    this.databaseService.deleteImage(id);

    return {
      message: 'Image deleted successfully',
      id,
    };
  }
}
