import { Controller, Get, Res } from '@nestjs/common';
import { AppService } from './app.service';
import * as fs from 'fs';
import * as path from 'path';
import type { Response } from 'express';
import { get } from 'http';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) { }

  @Get()
  gethome(@Res() res: Response) {
    const homePath = path.join(__dirname, '..', 'public', 'index.html');
    const content = fs.readFileSync(homePath, 'utf-8');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.send(content);
  }

  @Get('camera')
  getCamera(@Res() res: Response) {
    const cameraPath = path.join(__dirname, '..', 'public', 'camera.html');
    const content = fs.readFileSync(cameraPath, 'utf-8');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.send(content);
  }

  @Get('detect')
  getDetect(@Res() res: Response) {
    const detectPath = path.join(__dirname, '..', 'public', 'detection.html');
    const content = fs.readFileSync(detectPath, 'utf-8');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.send(content);
  }

  // Comentado para permitir que ServeStaticModule sirva index.html
  // @Get()
  // getHello(): string {
  //   return this.appService.getHello();
  // }
}
