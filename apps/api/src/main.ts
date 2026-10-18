import {
  NestFactory,
} from '@nestjs/core';

import {
  AppModule,
} from './app.module';


async function bootstrap() {
  const app =
    await NestFactory.create(
      AppModule,
    );

  app.enableCors({
    origin: [
      'http://localhost:3000',
      'http://127.0.0.1:3000',
      ...(process.env.CORS_ORIGIN ?? '').split(',').filter(Boolean),
    ],

    methods: [
      'GET',
      'POST',
      'PUT',
      'PATCH',
      'DELETE',
      'OPTIONS',
    ],

    allowedHeaders: [
      'Content-Type',
    ],

    credentials: false,
  });

  const port =
    Number(
      process.env.PORT ?? process.env.API_PORT,
    ) || 4000;

  await app.listen(
    port,
    '0.0.0.0',
  );

  console.log(
    `SecureVision API running on http://127.0.0.1:${port}`,
  );

  console.log(
    `AI service: ${
      process.env
        .AI_SERVICE_URL ??
      'http://127.0.0.1:8000'
    }`,
  );
}

bootstrap();