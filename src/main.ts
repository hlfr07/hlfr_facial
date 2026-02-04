import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { Logger } from '@nestjs/common';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Aumentar límite de tamaño de archivo
  app.use((req, res, next) => {
    if (req.path === '/api/upload') {
      req.socket.setMaxListeners(0);
    }
    next();
  });

  await app.listen(process.env.PORT ?? 5000);

  const logger = new Logger('Bootstrap');

  logger.log(`Server running on http://localhost:${process.env.PORT ?? 5000}`);
  logger.log(`Upload page available at http://localhost:${process.env.PORT ?? 5000}/`);
}
bootstrap();
