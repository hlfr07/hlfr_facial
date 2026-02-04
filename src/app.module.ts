import { Module } from '@nestjs/common';
import { ServeStaticModule } from '@nestjs/serve-static';
import { join } from 'path';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UploadController } from './controllers/upload.controller';
import { FacialController } from './controllers/facial.controller';
import { DetectionController } from './controllers/detection.controller';
import { DatabaseService } from './services/database.service';
import { ImageService } from './services/image.service';
import { FacialRecognitionService } from './services/facial-recognition.service';
import { YoloService } from './services/yolo.service';

@Module({
  imports: [
    // ServeStaticModule.forRoot({
    //   rootPath: join(__dirname, '..', 'public'),
    //   serveRoot: '/',
    // }),
  ],
  controllers: [AppController, UploadController, FacialController, DetectionController],
  providers: [AppService, DatabaseService, ImageService, FacialRecognitionService, YoloService],
})
export class AppModule {}
