import { Injectable, Logger } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';

export interface ImageRecord {
  id: string;
  filename: string;
  uploadedAt: string;
  verified?: boolean;
}

interface Database {
  images: ImageRecord[];
}

@Injectable()
export class DatabaseService {
  private readonly logger = new Logger(DatabaseService.name);
  private dbPath = path.join(__dirname, '..', 'database.json');

  constructor() {
    this.initializeDatabase();
  }

  private initializeDatabase() {
    if (!fs.existsSync(this.dbPath)) {
      const initialDb: Database = { images: [] };
      fs.writeFileSync(this.dbPath, JSON.stringify(initialDb, null, 2));
      this.logger.log('Database initialized');
    }
  }

  private readDatabase(): Database {
    const data = fs.readFileSync(this.dbPath, 'utf-8');
    return JSON.parse(data);
  }

  private writeDatabase(data: Database) {
    fs.writeFileSync(this.dbPath, JSON.stringify(data, null, 2));
  }

  addImage(filename: string): ImageRecord {
    const db = this.readDatabase();
    const imageRecord: ImageRecord = {
      id: Date.now().toString(),
      filename,
      uploadedAt: new Date().toISOString(),
    };

    db.images.push(imageRecord);
    this.writeDatabase(db);

    this.logger.log(`Image added: ${filename}`);
    return imageRecord;
  }

  getAllImages(): ImageRecord[] {
    const db = this.readDatabase();
    return db.images;
  }

  getImageById(id: string): ImageRecord | null {
    const db = this.readDatabase();
    return db.images.find((img) => img.id === id) || null;
  }

  updateImageVerification(id: string, verified: boolean) {
    const db = this.readDatabase();
    const image = db.images.find((img) => img.id === id);
    if (image) {
      image.verified = verified;
      this.writeDatabase(db);
      this.logger.log(`Image ${id} verification updated: ${verified}`);
    }
  }

  deleteImage(id: string): boolean {
    const db = this.readDatabase();
    const index = db.images.findIndex((img) => img.id === id);
    if (index > -1) {
      db.images.splice(index, 1);
      this.writeDatabase(db);
      this.logger.log(`Image ${id} deleted`);
      return true;
    }
    return false;
  }
}
