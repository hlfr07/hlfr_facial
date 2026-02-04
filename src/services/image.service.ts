import { Injectable, Logger } from '@nestjs/common';
import type { Express } from 'express';
import * as fs from 'fs';
import * as path from 'path';
import { DatabaseService } from './database.service';
import type { ImageRecord } from './database.service';

@Injectable()
export class ImageService {
  private readonly logger = new Logger(ImageService.name);
  private uploadsDir = path.join(__dirname, '..', '..', 'uploads');

  constructor(private databaseService: DatabaseService) {
    this.initializeUploadsDirectory();
  }

  private initializeUploadsDirectory() {
    if (!fs.existsSync(this.uploadsDir)) {
      fs.mkdirSync(this.uploadsDir, { recursive: true });
      this.logger.log('Uploads directory created');
    }
  }

  saveImage(file: Express.Multer.File): ImageRecord {
    const filename = `${Date.now()}-${file.originalname}`;
    const filepath = path.join(this.uploadsDir, filename);

    fs.writeFileSync(filepath, file.buffer);
    this.logger.log(`Image saved: ${filename}`);

    // Registrar en la base de datos
    const imageRecord = this.databaseService.addImage(filename);
    return imageRecord;
  }

  getImage(filename: string): Buffer | null {
    const filepath = path.join(this.uploadsDir, filename);
    if (fs.existsSync(filepath)) {
      return fs.readFileSync(filepath);
    }
    return null;
  }

  deleteImage(filename: string): boolean {
    const filepath = path.join(this.uploadsDir, filename);
    if (fs.existsSync(filepath)) {
      fs.unlinkSync(filepath);
      this.logger.log(`Image file deleted: ${filename}`);
      return true;
    }
    return false;
  }
}
