import {
  HttpException,
  Injectable,
  ServiceUnavailableException,
} from '@nestjs/common';

export interface UploadedVideoFile {
  originalname: string;
  mimetype: string;
  size: number;
  buffer: Buffer;
}

@Injectable()
export class AiService {
  private readonly aiBaseUrl =
    process.env.AI_SERVICE_URL ??
    'http://127.0.0.1:8000';

  private readonly aiApiKey =
    process.env.SECUREVISION_AI_API_KEY ??
    'securevision-dev-key';

  private async handleResponse(
    response: Response,
  ): Promise<any> {
    const text = await response.text();

    let payload: any;

    try {
      payload = text
        ? JSON.parse(text)
        : {};
    } catch {
      payload = {
        raw: text,
      };
    }

    if (!response.ok) {
      throw new HttpException(
        {
          message:
            'SecureVision AI service returned an error.',
          ai_status_code:
            response.status,
          ai_response:
            payload,
        },
        response.status,
      );
    }

    return payload;
  }

  private async fetchWithTimeout(
    url: string,
    options: RequestInit,
    timeoutMs = 120_000,
  ): Promise<Response> {
    const controller =
      new AbortController();

    const timeout =
      setTimeout(() => {
        controller.abort();
      }, timeoutMs);

    try {
      return await fetch(
        url,
        {
          ...options,
          signal:
            controller.signal,
        },
      );
    } catch (error) {
      throw new ServiceUnavailableException(
        {
          message:
            'Unable to connect to SecureVision AI service.',
          ai_service_url:
            this.aiBaseUrl,
          error:
            error instanceof Error
              ? error.message
              : String(error),
        },
      );
    } finally {
      clearTimeout(
        timeout,
      );
    }
  }

  async submitFullAnalysis(
    file: UploadedVideoFile,
  ): Promise<any> {
    const formData =
      new FormData();

    const bytes =
      Uint8Array.from(
        file.buffer,
      );

    const blob =
      new Blob(
        [bytes],
        {
          type:
            file.mimetype ||
            'video/quicktime',
        },
      );

    formData.append(
      'file',
      blob,
      file.originalname,
    );

    const response =
      await this.fetchWithTimeout(
        `${this.aiBaseUrl}/api/v1/full/analyze`,
        {
          method: 'POST',

          headers: {
            'X-API-Key':
              this.aiApiKey,
          },

          body:
            formData,
        },
      );

    return this.handleResponse(
      response,
    );
  }

  async getFullAnalysisJob(
    jobId: string,
  ): Promise<any> {
    const response =
      await this.fetchWithTimeout(
        `${this.aiBaseUrl}/api/v1/full/jobs/${encodeURIComponent(
          jobId,
        )}`,
        {
          method: 'GET',

          headers: {
            'X-API-Key':
              this.aiApiKey,
          },
        },
        30_000,
      );

    return this.handleResponse(
      response,
    );
  }

  async submitCashAnalysis(
    file: UploadedVideoFile,
  ): Promise<any> {
    const formData =
      new FormData();

    const bytes =
      Uint8Array.from(
        file.buffer,
      );

    const blob =
      new Blob(
        [bytes],
        {
          type:
            file.mimetype ||
            'video/quicktime',
        },
      );

    formData.append(
      'file',
      blob,
      file.originalname,
    );

    const response =
      await this.fetchWithTimeout(
        `${this.aiBaseUrl}/api/v1/analyze`,
        {
          method: 'POST',

          headers: {
            'X-API-Key':
              this.aiApiKey,
          },

          body:
            formData,
        },
      );

    return this.handleResponse(
      response,
    );
  }

  async getCashAnalysisJob(
    jobId: string,
  ): Promise<any> {
    const response =
      await this.fetchWithTimeout(
        `${this.aiBaseUrl}/api/v1/jobs/${encodeURIComponent(
          jobId,
        )}`,
        {
          method: 'GET',

          headers: {
            'X-API-Key':
              this.aiApiKey,
          },
        },
        30_000,
      );

    return this.handleResponse(
      response,
    );
  }

  async submitWorkplaceAnalysis(
    file: UploadedVideoFile,
  ): Promise<any> {
    const formData =
      new FormData();

    const bytes =
      Uint8Array.from(
        file.buffer,
      );

    const blob =
      new Blob(
        [bytes],
        {
          type:
            file.mimetype ||
            'video/quicktime',
        },
      );

    formData.append(
      'file',
      blob,
      file.originalname,
    );

    const response =
      await this.fetchWithTimeout(
        `${this.aiBaseUrl}/api/v1/workplace/analyze`,
        {
          method: 'POST',

          headers: {
            'X-API-Key':
              this.aiApiKey,
          },

          body:
            formData,
        },
      );

    return this.handleResponse(
      response,
    );
  }

  async getWorkplaceAnalysisJob(
    jobId: string,
  ): Promise<any> {
    const response =
      await this.fetchWithTimeout(
        `${this.aiBaseUrl}/api/v1/workplace/jobs/${encodeURIComponent(
          jobId,
        )}`,
        {
          method: 'GET',

          headers: {
            'X-API-Key':
              this.aiApiKey,
          },
        },
        30_000,
      );

    return this.handleResponse(
      response,
    );
  }
}