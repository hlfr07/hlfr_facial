import { Controller, Get } from '@nestjs/common';
import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  // Comentado para permitir que ServeStaticModule sirva index.html
  // @Get()
  // getHello(): string {
  //   return this.appService.getHello();
  // }
}
