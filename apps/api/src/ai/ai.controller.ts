import {
  BadRequestException,
  Controller,
  Get,
  Param,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';

import {
  FileInterceptor,
} from '@nestjs/platform-express';

import {
  extname,
} from 'node:path';

import {
  AiService,
} from './ai.service';

import type {
  UploadedVideoFile,
} from './ai.service';


const MAX_FILE_SIZE =
  150 *
  1024 *
  1024;


@Controller('api/analysis')
export class AiController {
  constructor(
    private readonly aiService:
      AiService,
  ) {}

  private validateFile(
    file:
      UploadedVideoFile
      | undefined,
  ): asserts file is UploadedVideoFile {
    if (!file) {
      throw new BadRequestException(
        'MOV video file is required.',
      );
    }

    const extension =
      extname(
        file.originalname,
      ).toLowerCase();

    if (extension !== '.mov') {
      throw new BadRequestException(
        'Only .mov files are supported.',
      );
    }

    if (file.size <= 0) {
      throw new BadRequestException(
        'Uploaded file is empty.',
      );
    }
  }

  // ==========================================
  // FULL SECUREVISION ANALYSIS
  // Cash Discipline + Workplace Analytics
  // ==========================================

  @Post('full')
  @UseInterceptors(
    FileInterceptor(
      'file',
      {
        limits: {
          fileSize:
            MAX_FILE_SIZE,
        },
      },
    ),
  )
  async submitFullAnalysis(
    @UploadedFile()
    file: UploadedVideoFile,
  ) {
    this.validateFile(
      file,
    );

    return this.aiService
      .submitFullAnalysis(
        file,
      );
  }

  @Get('full/:jobId')
  async getFullAnalysis(
    @Param('jobId')
    jobId: string,
  ) {
    return this.aiService
      .getFullAnalysisJob(
        jobId,
      );
  }

  // ==========================================
  // CASH DISCIPLINE ONLY
  // ==========================================

  @Post('cash')
  @UseInterceptors(
    FileInterceptor(
      'file',
      {
        limits: {
          fileSize:
            MAX_FILE_SIZE,
        },
      },
    ),
  )
  async submitCashAnalysis(
    @UploadedFile()
    file: UploadedVideoFile,
  ) {
    this.validateFile(
      file,
    );

    return this.aiService
      .submitCashAnalysis(
        file,
      );
  }

  @Get('cash/:jobId')
  async getCashAnalysis(
    @Param('jobId')
    jobId: string,
  ) {
    return this.aiService
      .getCashAnalysisJob(
        jobId,
      );
  }

  // ==========================================
  // WORKPLACE ONLY
  // ==========================================

  @Post('workplace')
  @UseInterceptors(
    FileInterceptor(
      'file',
      {
        limits: {
          fileSize:
            MAX_FILE_SIZE,
        },
      },
    ),
  )
  async submitWorkplaceAnalysis(
    @UploadedFile()
    file: UploadedVideoFile,
  ) {
    this.validateFile(
      file,
    );

    return this.aiService
      .submitWorkplaceAnalysis(
        file,
      );
  }

  @Get('workplace/:jobId')
  async getWorkplaceAnalysis(
    @Param('jobId')
    jobId: string,
  ) {
    return this.aiService
      .getWorkplaceAnalysisJob(
        jobId,
      );
  }
}